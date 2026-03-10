"""
Tests for C037: SMT Solver
Composes C035 (SAT Solver) with Simplex theory solver for Linear Integer Arithmetic.
"""

import pytest
import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
from smt_solver import (
    SMTSolver, SMTResult, Term, Var, IntConst, BoolConst, App, Op,
    LinearExpr, Constraint, ConstraintKind, SimplexSolver, CongruenceClosure,
    FuncDecl, UFApp, BOOL, INT, REAL, parse_smtlib2,
)


# ===== LinearExpr Tests =====

class TestLinearExpr:
    def test_constant(self):
        e = LinearExpr(const=Fraction(5))
        assert e.is_constant()
        assert e.const == 5

    def test_single_var(self):
        e = LinearExpr(coeffs={'x': Fraction(3)}, const=Fraction(2))
        assert not e.is_constant()
        assert e.evaluate({'x': 4}) == Fraction(14)

    def test_addition(self):
        a = LinearExpr(coeffs={'x': Fraction(1)}, const=Fraction(2))
        b = LinearExpr(coeffs={'y': Fraction(3)}, const=Fraction(1))
        c = a + b
        assert c.coeffs == {'x': Fraction(1), 'y': Fraction(3)}
        assert c.const == Fraction(3)

    def test_subtraction(self):
        a = LinearExpr(coeffs={'x': Fraction(5)})
        b = LinearExpr(coeffs={'x': Fraction(2)})
        c = a - b
        assert c.coeffs == {'x': Fraction(3)}

    def test_scalar_mul(self):
        a = LinearExpr(coeffs={'x': Fraction(2), 'y': Fraction(3)}, const=Fraction(1))
        b = a * Fraction(3)
        assert b.coeffs == {'x': Fraction(6), 'y': Fraction(9)}
        assert b.const == Fraction(3)

    def test_negation(self):
        a = LinearExpr(coeffs={'x': Fraction(1)}, const=Fraction(2))
        b = -a
        assert b.coeffs == {'x': Fraction(-1)}
        assert b.const == Fraction(-2)

    def test_zero_coeff_removal(self):
        a = LinearExpr(coeffs={'x': Fraction(3)})
        b = LinearExpr(coeffs={'x': Fraction(3)})
        c = a - b
        assert 'x' not in c.coeffs

    def test_add_scalar(self):
        a = LinearExpr(coeffs={'x': Fraction(1)})
        b = a + 5
        assert b.const == Fraction(5)

    def test_sub_scalar(self):
        a = LinearExpr(coeffs={'x': Fraction(1)}, const=Fraction(10))
        b = a - 3
        assert b.const == Fraction(7)

    def test_evaluate_multivar(self):
        e = LinearExpr(coeffs={'x': Fraction(2), 'y': Fraction(-1)}, const=Fraction(3))
        assert e.evaluate({'x': 5, 'y': 2}) == Fraction(11)


# ===== SimplexSolver Tests =====

class TestSimplex:
    def test_simple_feasible(self):
        s = SimplexSolver()
        s.add_variable('x')
        # x <= 5
        e = LinearExpr(coeffs={'x': Fraction(1)}, const=Fraction(-5))
        s.add_constraint(e, ConstraintKind.LE)
        assert s.check()

    def test_simple_infeasible(self):
        s = SimplexSolver()
        s.add_variable('x')
        # x <= -1
        e1 = LinearExpr(coeffs={'x': Fraction(1)}, const=Fraction(1))
        s.add_constraint(e1, ConstraintKind.LE)
        # x >= 1 means -x <= -1
        e2 = LinearExpr(coeffs={'x': Fraction(-1)}, const=Fraction(1))
        s.add_constraint(e2, ConstraintKind.LE)
        # x <= -1 AND x >= 1 is infeasible
        assert not s.check()

    def test_equality(self):
        s = SimplexSolver()
        s.add_variable('x')
        # x == 3
        e = LinearExpr(coeffs={'x': Fraction(1)}, const=Fraction(-3))
        s.add_constraint(e, ConstraintKind.EQ)
        assert s.check()
        assert s.get_value('x') == Fraction(3)

    def test_two_vars(self):
        s = SimplexSolver()
        s.add_variable('x')
        s.add_variable('y')
        # x + y <= 10
        e1 = LinearExpr(coeffs={'x': Fraction(1), 'y': Fraction(1)}, const=Fraction(-10))
        s.add_constraint(e1, ConstraintKind.LE)
        # x >= 3
        e2 = LinearExpr(coeffs={'x': Fraction(-1)}, const=Fraction(3))
        s.add_constraint(e2, ConstraintKind.LE)
        # y >= 3
        e3 = LinearExpr(coeffs={'y': Fraction(-1)}, const=Fraction(3))
        s.add_constraint(e3, ConstraintKind.LE)
        assert s.check()
        x = s.get_value('x')
        y = s.get_value('y')
        assert x >= 3
        assert y >= 3
        assert x + y <= 10

    def test_infeasible_two_vars(self):
        s = SimplexSolver()
        s.add_variable('x')
        s.add_variable('y')
        # x + y <= 4
        e1 = LinearExpr(coeffs={'x': Fraction(1), 'y': Fraction(1)}, const=Fraction(-4))
        s.add_constraint(e1, ConstraintKind.LE)
        # x >= 3
        e2 = LinearExpr(coeffs={'x': Fraction(-1)}, const=Fraction(3))
        s.add_constraint(e2, ConstraintKind.LE)
        # y >= 3
        e3 = LinearExpr(coeffs={'y': Fraction(-1)}, const=Fraction(3))
        s.add_constraint(e3, ConstraintKind.LE)
        assert not s.check()

    def test_strict_inequality(self):
        s = SimplexSolver()
        s.add_variable('x')
        # x < 1 (i.e. x <= 0 for integers)
        e = LinearExpr(coeffs={'x': Fraction(1)}, const=Fraction(-1))
        s.add_constraint(e, ConstraintKind.LT)
        # x >= 0
        e2 = LinearExpr(coeffs={'x': Fraction(-1)})
        s.add_constraint(e2, ConstraintKind.LE)
        assert s.check()
        assert s.get_value('x') == Fraction(0)

    def test_save_restore(self):
        s = SimplexSolver()
        s.add_variable('x')
        e = LinearExpr(coeffs={'x': Fraction(1)}, const=Fraction(-10))
        s.add_constraint(e, ConstraintKind.LE)
        state = s.save_state()

        # Add conflicting constraint
        e2 = LinearExpr(coeffs={'x': Fraction(-1)}, const=Fraction(20))
        s.add_constraint(e2, ConstraintKind.LE)
        assert not s.check()

        # Restore
        s.restore_state(state)
        assert s.check()

    def test_multiple_constraints(self):
        s = SimplexSolver()
        s.add_variable('x')
        s.add_variable('y')
        s.add_variable('z')
        # x + y + z <= 15
        e1 = LinearExpr(coeffs={'x': Fraction(1), 'y': Fraction(1), 'z': Fraction(1)}, const=Fraction(-15))
        s.add_constraint(e1, ConstraintKind.LE)
        # x >= 1
        e2 = LinearExpr(coeffs={'x': Fraction(-1)}, const=Fraction(1))
        s.add_constraint(e2, ConstraintKind.LE)
        # y >= 2
        e3 = LinearExpr(coeffs={'y': Fraction(-1)}, const=Fraction(2))
        s.add_constraint(e3, ConstraintKind.LE)
        # z >= 3
        e4 = LinearExpr(coeffs={'z': Fraction(-1)}, const=Fraction(3))
        s.add_constraint(e4, ConstraintKind.LE)
        assert s.check()


# ===== CongruenceClosure Tests =====

class TestCongruenceClosure:
    def test_basic_equality(self):
        cc = CongruenceClosure()
        cc.add_term(1)
        cc.add_term(2)
        assert not cc.are_equal(1, 2)
        cc.merge(1, 2)
        assert cc.are_equal(1, 2)

    def test_transitivity(self):
        cc = CongruenceClosure()
        cc.add_term(1)
        cc.add_term(2)
        cc.add_term(3)
        cc.merge(1, 2)
        cc.merge(2, 3)
        assert cc.are_equal(1, 3)

    def test_congruence(self):
        cc = CongruenceClosure()
        cc.add_term(1)  # a
        cc.add_term(2)  # b
        cc.add_term(3, 'f', [1])  # f(a)
        cc.add_term(4, 'f', [2])  # f(b)
        assert not cc.are_equal(3, 4)
        cc.merge(1, 2)  # a == b
        assert cc.are_equal(3, 4)  # f(a) == f(b)

    def test_disequality(self):
        cc = CongruenceClosure()
        cc.add_term(1)
        cc.add_term(2)
        cc.add_disequality(1, 2, 100)
        assert cc.check()
        cc.merge(1, 2)
        assert not cc.check()

    def test_nested_congruence(self):
        cc = CongruenceClosure()
        cc.add_term(1)  # a
        cc.add_term(2)  # b
        cc.add_term(3, 'f', [1])  # f(a)
        cc.add_term(4, 'f', [2])  # f(b)
        cc.add_term(5, 'g', [3])  # g(f(a))
        cc.add_term(6, 'g', [4])  # g(f(b))
        cc.merge(1, 2)
        assert cc.are_equal(5, 6)

    def test_save_restore(self):
        cc = CongruenceClosure()
        cc.add_term(1)
        cc.add_term(2)
        state = cc.save_state()
        cc.merge(1, 2)
        assert cc.are_equal(1, 2)
        cc.restore_state(state)
        assert not cc.are_equal(1, 2)


# ===== SMTSolver Term Construction =====

class TestTermConstruction:
    def test_int_var(self):
        s = SMTSolver()
        x = s.Int('x')
        assert isinstance(x, Var)
        assert x.sort == INT

    def test_bool_var(self):
        s = SMTSolver()
        b = s.Bool('b')
        assert isinstance(b, Var)
        assert b.sort == BOOL

    def test_arithmetic(self):
        s = SMTSolver()
        x = s.Int('x')
        y = s.Int('y')
        t = x + y
        assert isinstance(t, App)
        assert t.op == Op.ADD

    def test_comparison(self):
        s = SMTSolver()
        x = s.Int('x')
        t = x <= 5
        assert isinstance(t, App)
        assert t.op == Op.LE

    def test_boolean_ops(self):
        s = SMTSolver()
        x = s.Int('x')
        a = x <= 5
        b = x >= 0
        t = s.And(a, b)
        assert isinstance(t, App)
        assert t.op == Op.AND

    def test_negation(self):
        s = SMTSolver()
        x = s.Int('x')
        t = -x
        assert isinstance(t, App)
        assert t.op == Op.NEG

    def test_subtraction(self):
        s = SMTSolver()
        x = s.Int('x')
        y = s.Int('y')
        t = x - y
        assert isinstance(t, App)
        assert t.op == Op.SUB

    def test_mul_const(self):
        s = SMTSolver()
        x = s.Int('x')
        t = 3 * x
        assert isinstance(t, App)
        assert t.op == Op.MUL

    def test_ints_helper(self):
        s = SMTSolver()
        x, y, z = s.Ints('x y z')
        assert x.name == 'x'
        assert y.name == 'y'
        assert z.name == 'z'

    def test_bools_helper(self):
        s = SMTSolver()
        a, b = s.Bools('a b')
        assert a.name == 'a'
        assert b.name == 'b'

    def test_distinct(self):
        s = SMTSolver()
        x, y, z = s.Ints('x y z')
        d = s.Distinct(x, y, z)
        assert isinstance(d, App)

    def test_intval(self):
        s = SMTSolver()
        v = s.IntVal(42)
        assert isinstance(v, IntConst)
        assert v.value == 42

    def test_boolval(self):
        s = SMTSolver()
        v = s.BoolVal(True)
        assert isinstance(v, BoolConst)
        assert v.value is True

    def test_implies(self):
        s = SMTSolver()
        a = s.Bool('a')
        b = s.Bool('b')
        t = s.Implies(a, b)
        assert isinstance(t, App)
        assert t.op == Op.IMPLIES

    def test_iff(self):
        s = SMTSolver()
        a = s.Bool('a')
        b = s.Bool('b')
        t = s.Iff(a, b)
        assert isinstance(t, App)
        assert t.op == Op.IFF

    def test_if_then_else(self):
        s = SMTSolver()
        c = s.Bool('c')
        x = s.Int('x')
        y = s.Int('y')
        t = s.If(c, x, y)
        assert isinstance(t, App)
        assert t.op == Op.ITE


# ===== Basic SMT Solving =====

class TestBasicSolving:
    def test_simple_sat(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x <= 10)
        s.add(x >= 5)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert 5 <= m['x'] <= 10

    def test_simple_unsat(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x <= 3)
        s.add(x >= 5)
        assert s.check() == SMTResult.UNSAT

    def test_equality(self):
        s = SMTSolver()
        x = s.Int('x')
        y = s.Int('y')
        s.add(x == y)
        s.add(x >= 5)
        s.add(y <= 10)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] == m['y']
        assert m['x'] >= 5

    def test_two_var_constraint(self):
        s = SMTSolver()
        x = s.Int('x')
        y = s.Int('y')
        s.add(x + y <= 10)
        s.add(x >= 3)
        s.add(y >= 3)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] + m['y'] <= 10
        assert m['x'] >= 3
        assert m['y'] >= 3

    def test_two_var_unsat(self):
        s = SMTSolver()
        x = s.Int('x')
        y = s.Int('y')
        s.add(x + y <= 4)
        s.add(x >= 3)
        s.add(y >= 3)
        assert s.check() == SMTResult.UNSAT

    def test_strict_less_than(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x < 5)
        s.add(x > 3)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] == 4

    def test_strict_less_than_unsat(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x < 5)
        s.add(x > 4)
        # x < 5 means x <= 4, x > 4 means x >= 5 -- unsat
        assert s.check() == SMTResult.UNSAT

    def test_neg_constraint(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 0)
        s.add(x <= 100)
        s.add(-x <= -10)  # x >= 10
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] >= 10

    def test_linear_combination(self):
        s = SMTSolver()
        x = s.Int('x')
        y = s.Int('y')
        s.add(2 * x + 3 * y <= 12)
        s.add(x >= 0)
        s.add(y >= 0)
        s.add(x + y >= 3)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert 2 * m['x'] + 3 * m['y'] <= 12
        assert m['x'] + m['y'] >= 3

    def test_three_vars(self):
        s = SMTSolver()
        x, y, z = s.Ints('x y z')
        s.add(x + y + z <= 10)
        s.add(x >= 1)
        s.add(y >= 2)
        s.add(z >= 3)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] >= 1
        assert m['y'] >= 2
        assert m['z'] >= 3
        assert m['x'] + m['y'] + m['z'] <= 10


# ===== Boolean Reasoning =====

class TestBooleanReasoning:
    def test_pure_bool_sat(self):
        s = SMTSolver()
        a = s.Bool('a')
        b = s.Bool('b')
        s.add(s.Or(a, b))
        assert s.check() == SMTResult.SAT

    def test_pure_bool_unsat(self):
        s = SMTSolver()
        a = s.Bool('a')
        s.add(a)
        s.add(s.Not(a))
        assert s.check() == SMTResult.UNSAT

    def test_implication(self):
        s = SMTSolver()
        a = s.Bool('a')
        b = s.Bool('b')
        s.add(s.Implies(a, b))
        s.add(a)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['b'] is True

    def test_iff_sat(self):
        s = SMTSolver()
        a = s.Bool('a')
        b = s.Bool('b')
        s.add(s.Iff(a, b))
        s.add(a)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['a'] == m['b']

    def test_and_constraint(self):
        s = SMTSolver()
        a, b, c = s.Bools('a b c')
        s.add(s.And(a, b, c))
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['a'] is True
        assert m['b'] is True
        assert m['c'] is True

    def test_or_constraint(self):
        s = SMTSolver()
        a, b = s.Bools('a b')
        s.add(s.Or(a, b))
        s.add(s.Not(a))
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['b'] is True

    def test_bool_arithmetic_mix(self):
        s = SMTSolver()
        x = s.Int('x')
        b = s.Bool('b')
        # If b then x > 0 else x < 0
        s.add(s.Implies(b, x > 0))
        s.add(s.Implies(s.Not(b), x < 0))
        s.add(b)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] > 0


# ===== Disjunctive Constraints (OR of arithmetic) =====

class TestDisjunctive:
    def test_or_arithmetic(self):
        s = SMTSolver()
        x = s.Int('x')
        # x <= 0 OR x >= 10
        s.add(s.Or(x <= 0, x >= 10))
        s.add(x >= -5)
        s.add(x <= 20)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] <= 0 or m['x'] >= 10

    def test_or_arithmetic_unsat(self):
        s = SMTSolver()
        x = s.Int('x')
        # x <= 0 OR x >= 10
        s.add(s.Or(x <= 0, x >= 10))
        # x >= 3 AND x <= 7 -- conflict with both branches
        s.add(x >= 3)
        s.add(x <= 7)
        assert s.check() == SMTResult.UNSAT

    def test_multiple_disjuncts(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(s.Or(x == s.IntVal(1), x == s.IntVal(2), x == s.IntVal(3)))
        s.add(x >= 2)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] in (2, 3)

    def test_nested_or_and(self):
        s = SMTSolver()
        x, y = s.Ints('x y')
        # (x >= 0 AND y >= 0) OR (x <= -1 AND y <= -1)
        s.add(s.Or(
            s.And(x >= 0, y >= 0),
            s.And(x <= -1, y <= -1)
        ))
        s.add(x + y >= 0)
        assert s.check() == SMTResult.SAT


# ===== Not-Equal (!=) =====

class TestNotEqual:
    def test_neq_sat(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x != s.IntVal(5))
        s.add(x >= 4)
        s.add(x <= 6)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] != 5
        assert 4 <= m['x'] <= 6

    def test_neq_forces_value(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 0)
        s.add(x <= 2)
        s.add(x != s.IntVal(0))
        s.add(x != s.IntVal(2))
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] == 1


# ===== Uninterpreted Functions =====

class TestUninterpretedFunctions:
    def test_uf_basic(self):
        s = SMTSolver()
        f = s.Function('f', INT, INT)
        x = s.Int('x')
        y = s.Int('y')
        s.add(x == y)
        s.add(f(x) != f(y))
        # f(x) should equal f(y) if x == y -- unsat
        assert s.check() == SMTResult.UNSAT

    def test_uf_sat(self):
        s = SMTSolver()
        f = s.Function('f', INT, INT)
        x = s.Int('x')
        y = s.Int('y')
        s.add(x != y)
        s.add(f(x) != f(y))
        # Different inputs can have different outputs
        assert s.check() == SMTResult.SAT

    def test_uf_transitivity(self):
        s = SMTSolver()
        f = s.Function('f', INT, INT)
        x, y, z = s.Ints('x y z')
        s.add(x == y)
        s.add(y == z)
        s.add(f(x) != f(z))
        assert s.check() == SMTResult.UNSAT


# ===== Push/Pop =====

class TestPushPop:
    def test_basic_push_pop(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 0)
        s.add(x <= 10)
        s.push()
        s.add(x >= 20)  # Conflict
        assert s.check() == SMTResult.UNSAT
        s.pop()
        assert s.check() == SMTResult.SAT

    def test_nested_push_pop(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 0)
        s.push()
        s.add(x <= 5)
        s.push()
        s.add(x >= 10)  # Conflict
        assert s.check() == SMTResult.UNSAT
        s.pop()
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert 0 <= m['x'] <= 5
        s.pop()
        assert s.check() == SMTResult.SAT

    def test_pop_error(self):
        s = SMTSolver()
        with pytest.raises(ValueError):
            s.pop()


# ===== ITE (If-Then-Else) =====

class TestITE:
    def test_ite_bool(self):
        s = SMTSolver()
        c = s.Bool('c')
        x = s.Int('x')
        y = s.Int('y')
        result = s.If(c, x, y)
        s.add(x >= 10)
        s.add(y <= 0)
        s.add(c)
        s.add(result >= 5)
        assert s.check() == SMTResult.SAT


# ===== Model Validation =====

class TestModelValidation:
    def test_model_satisfies_constraints(self):
        s = SMTSolver()
        x, y = s.Ints('x y')
        s.add(x + y <= 20)
        s.add(x >= 5)
        s.add(y >= 5)
        s.add(x - y <= 3)
        s.add(y - x <= 3)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] + m['y'] <= 20
        assert m['x'] >= 5
        assert m['y'] >= 5
        assert m['x'] - m['y'] <= 3
        assert m['y'] - m['x'] <= 3

    def test_model_equality_chain(self):
        s = SMTSolver()
        x, y, z = s.Ints('x y z')
        s.add(x == y)
        s.add(y == z)
        s.add(z >= 10)
        s.add(z <= 10)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] == m['y'] == m['z'] == 10

    def test_zero_solution(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 0)
        s.add(x <= 0)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] == 0

    def test_negative_solution(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x <= -5)
        s.add(x >= -10)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert -10 <= m['x'] <= -5


# ===== Linear Arithmetic Edge Cases =====

class TestLIAEdgeCases:
    def test_many_constraints(self):
        s = SMTSolver()
        vars = [s.Int(f'x{i}') for i in range(5)]
        for v in vars:
            s.add(v >= 0)
            s.add(v <= 10)
        # Sum <= 20
        total = vars[0]
        for v in vars[1:]:
            total = total + v
        s.add(total <= 20)
        assert s.check() == SMTResult.SAT

    def test_coefficients(self):
        s = SMTSolver()
        x, y = s.Ints('x y')
        s.add(3 * x + 2 * y <= 12)
        s.add(x >= 0)
        s.add(y >= 0)
        s.add(x + y >= 4)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert 3 * m['x'] + 2 * m['y'] <= 12
        assert m['x'] + m['y'] >= 4

    def test_large_constant(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 1000000)
        s.add(x <= 1000001)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] in (1000000, 1000001)

    def test_symmetric_constraints(self):
        s = SMTSolver()
        x, y = s.Ints('x y')
        s.add(x + y <= 10)
        s.add(x - y <= 2)
        s.add(y - x <= 2)
        s.add(x >= 0)
        s.add(y >= 0)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert abs(m['x'] - m['y']) <= 2


# ===== Multiple check() calls =====

class TestMultipleChecks:
    def test_recheck_after_add(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 0)
        assert s.check() == SMTResult.SAT

        s.add(x <= 5)
        assert s.check() == SMTResult.SAT

        s.add(x >= 10)
        assert s.check() == SMTResult.UNSAT

    def test_independent_solvers(self):
        s1 = SMTSolver()
        s2 = SMTSolver()
        x1 = s1.Int('x')
        x2 = s2.Int('x')
        s1.add(x1 >= 0)
        s2.add(x2 <= -1)
        assert s1.check() == SMTResult.SAT
        assert s2.check() == SMTResult.SAT


# ===== Harder Problems =====

class TestHarderProblems:
    def test_pigeonhole_2_1(self):
        """2 pigeons, 1 hole -- must be UNSAT"""
        s = SMTSolver()
        # Pigeon i in hole j: p_i_j is boolean
        p = [[s.Bool(f'p_{i}_{j}') for j in range(1)] for i in range(2)]
        # Each pigeon must be in some hole
        for i in range(2):
            s.add(s.Or(*p[i]))
        # No two pigeons in same hole
        for j in range(1):
            for i1 in range(2):
                for i2 in range(i1 + 1, 2):
                    s.add(s.Not(s.And(p[i1][j], p[i2][j])))
        assert s.check() == SMTResult.UNSAT

    def test_pigeonhole_2_2(self):
        """2 pigeons, 2 holes -- SAT"""
        s = SMTSolver()
        p = [[s.Bool(f'p_{i}_{j}') for j in range(2)] for i in range(2)]
        for i in range(2):
            s.add(s.Or(*p[i]))
        for j in range(2):
            for i1 in range(2):
                for i2 in range(i1 + 1, 2):
                    s.add(s.Not(s.And(p[i1][j], p[i2][j])))
        assert s.check() == SMTResult.SAT

    def test_queens_2(self):
        """2-queens on 2x2: UNSAT"""
        s = SMTSolver()
        # q[i] = column of queen in row i
        q = [s.Int(f'q{i}') for i in range(2)]
        for qi in q:
            s.add(qi >= 0)
            s.add(qi <= 1)
        # Different columns
        s.add(q[0] != q[1])
        # Different diagonals
        # |q0 - q1| != |0 - 1| = 1
        s.add(q[0] - q[1] != s.IntVal(1))
        s.add(q[1] - q[0] != s.IntVal(1))
        assert s.check() == SMTResult.UNSAT

    def test_system_of_equations(self):
        s = SMTSolver()
        x, y, z = s.Ints('x y z')
        s.add(x + y == s.IntVal(10))
        s.add(y + z == s.IntVal(15))
        s.add(x + z == s.IntVal(11))
        # Solution: x=3, y=7, z=8
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] + m['y'] == 10
        assert m['y'] + m['z'] == 15
        assert m['x'] + m['z'] == 11

    def test_absolute_value(self):
        s = SMTSolver()
        x = s.Int('x')
        a = s.Int('a')  # |x| = a
        # a == x if x >= 0, else a == -x
        s.add(s.Or(
            s.And(x >= 0, a == x),
            s.And(x < 0, a == -x)
        ))
        s.add(a <= 5)
        s.add(a >= 3)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert 3 <= abs(m['x']) <= 5


# ===== SMT-LIB2 Parsing =====

class TestSMTLIB2:
    def test_parse_simple(self):
        text = """
        (declare-const x Int)
        (declare-const y Int)
        (assert (>= x 0))
        (assert (<= x 10))
        (assert (>= y 0))
        (assert (<= (+ x y) 15))
        (check-sat)
        """
        solver = parse_smtlib2(text)
        assert solver.check() == SMTResult.SAT

    def test_parse_unsat(self):
        text = """
        (declare-const x Int)
        (assert (<= x 0))
        (assert (>= x 5))
        (check-sat)
        """
        solver = parse_smtlib2(text)
        assert solver.check() == SMTResult.UNSAT

    def test_parse_nested(self):
        text = """
        (declare-const x Int)
        (declare-const y Int)
        (assert (and (>= x 0) (<= x 10)))
        (assert (or (<= y 0) (>= y 5)))
        (check-sat)
        """
        solver = parse_smtlib2(text)
        assert solver.check() == SMTResult.SAT

    def test_parse_bool(self):
        text = """
        (declare-const a Bool)
        (declare-const b Bool)
        (assert (or a b))
        (assert (not a))
        (check-sat)
        """
        solver = parse_smtlib2(text)
        result = solver.check()
        assert result == SMTResult.SAT

    def test_parse_multiplication(self):
        text = """
        (declare-const x Int)
        (assert (>= x 0))
        (assert (<= (* 2 x) 10))
        (check-sat)
        """
        solver = parse_smtlib2(text)
        assert solver.check() == SMTResult.SAT

    def test_parse_distinct(self):
        text = """
        (declare-const x Int)
        (declare-const y Int)
        (assert (>= x 0))
        (assert (<= x 1))
        (assert (>= y 0))
        (assert (<= y 1))
        (assert (distinct x y))
        (check-sat)
        """
        solver = parse_smtlib2(text)
        assert solver.check() == SMTResult.SAT


# ===== Composition with C035 =====

class TestC035Composition:
    def test_sat_solver_imported(self):
        """Verify C035 SAT solver is properly imported and usable."""
        from sat_solver import Solver, SolverResult
        s = Solver()
        v1 = s.new_var()
        v2 = s.new_var()
        s.add_clause([v1, v2])
        s.add_clause([-v1, v2])
        assert s.solve() == SolverResult.SAT

    def test_theory_lemma_feedback(self):
        """Verify theory lemmas are fed back to SAT solver."""
        s = SMTSolver()
        x = s.Int('x')
        y = s.Int('y')
        # Boolean: pick x >= 10 OR y >= 10
        # Arithmetic: x + y <= 8
        s.add(s.Or(x >= 10, y >= 10))
        s.add(x + y <= 8)
        s.add(x >= 0)
        s.add(y >= 0)
        # Both branches of the OR are infeasible with x+y<=8 when both >= 0
        assert s.check() == SMTResult.UNSAT


# ===== Solver Result API =====

class TestSolverAPI:
    def test_result_method(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 0)
        s.check()
        assert s.result() == SMTResult.SAT

    def test_model_none_before_check(self):
        s = SMTSolver()
        assert s.model() is None

    def test_model_none_after_unsat(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 10)
        s.add(x <= 0)
        s.check()
        assert s.model() is None

    def test_repr(self):
        s = SMTSolver()
        s.Int('x')
        s.add(s._vars['x'] >= 0)
        r = repr(s)
        assert 'SMTSolver' in r

    def test_function_decl(self):
        s = SMTSolver()
        f = s.Function('f', INT, INT)
        assert f.name == 'f'
        assert f.domain == [INT]
        assert f.range_sort == INT


# ===== Edge Cases =====

class TestEdgeCases:
    def test_empty_solver(self):
        s = SMTSolver()
        assert s.check() == SMTResult.SAT

    def test_single_true_assertion(self):
        s = SMTSolver()
        s.add(s.BoolVal(True))
        assert s.check() == SMTResult.SAT

    def test_single_false_assertion(self):
        s = SMTSolver()
        s.add(s.BoolVal(False))
        assert s.check() == SMTResult.UNSAT

    def test_duplicate_var_name(self):
        s = SMTSolver()
        x1 = s.Int('x')
        x2 = s.Int('x')
        assert x1 is x2  # Same variable returned

    def test_constant_equality(self):
        s = SMTSolver()
        s.add(s.IntVal(5) == s.IntVal(5))
        assert s.check() == SMTResult.SAT

    def test_constant_inequality(self):
        s = SMTSolver()
        s.add(s.IntVal(5) == s.IntVal(6))
        assert s.check() == SMTResult.UNSAT

    def test_no_int_vars_bool_only(self):
        s = SMTSolver()
        a, b, c = s.Bools('a b c')
        s.add(s.Or(a, b))
        s.add(s.Or(s.Not(a), c))
        s.add(s.Not(c))
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m.get('a', False) is False or m.get('c', False) is True

    def test_var_reuse_across_assertions(self):
        s = SMTSolver()
        x = s.Int('x')
        s.add(x >= 0)
        s.add(x <= 10)
        s.add(x >= 5)
        s.add(x <= 7)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert 5 <= m['x'] <= 7


# ===== Practical Problems =====

class TestPracticalProblems:
    def test_scheduling(self):
        """Simple scheduling: 3 tasks with durations, single resource."""
        s = SMTSolver()
        # Start times
        t1, t2, t3 = s.Ints('t1 t2 t3')
        d1, d2, d3 = 2, 3, 1  # durations

        # All start after time 0
        s.add(t1 >= 0)
        s.add(t2 >= 0)
        s.add(t3 >= 0)

        # No overlap (disjunctive)
        # t1 + d1 <= t2 OR t2 + d2 <= t1
        s.add(s.Or(t1 + d1 <= t2, t2 + d2 <= t1))
        s.add(s.Or(t1 + d1 <= t3, t3 + d3 <= t1))
        s.add(s.Or(t2 + d2 <= t3, t3 + d3 <= t2))

        # All finish by time 10
        s.add(t1 + d1 <= 10)
        s.add(t2 + d2 <= 10)
        s.add(t3 + d3 <= 10)

        assert s.check() == SMTResult.SAT
        m = s.model()
        # Verify no overlaps
        intervals = [(m['t1'], m['t1'] + d1), (m['t2'], m['t2'] + d2), (m['t3'], m['t3'] + d3)]
        for i in range(3):
            for j in range(i+1, 3):
                assert intervals[i][1] <= intervals[j][0] or intervals[j][1] <= intervals[i][0]

    def test_bounded_addition(self):
        """Verify x + y doesn't overflow a bound."""
        s = SMTSolver()
        x, y, z = s.Ints('x y z')
        s.add(x >= 0)
        s.add(x <= 100)
        s.add(y >= 0)
        s.add(y <= 100)
        s.add(z == x + y)
        s.add(z > 200)  # Impossible since max is 100+100=200
        assert s.check() == SMTResult.UNSAT

    def test_resource_allocation(self):
        """Allocate resources to tasks with constraints."""
        s = SMTSolver()
        # Resources allocated to 3 tasks
        r1, r2, r3 = s.Ints('r1 r2 r3')
        # At least 1 unit each
        s.add(r1 >= 1)
        s.add(r2 >= 1)
        s.add(r3 >= 1)
        # Total budget = 10
        s.add(r1 + r2 + r3 <= 10)
        # Task 2 needs more than task 1
        s.add(r2 > r1)
        # Task 3 needs at least twice task 1
        s.add(r3 >= 2 * r1)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['r2'] > m['r1']
        assert m['r3'] >= 2 * m['r1']
        assert m['r1'] + m['r2'] + m['r3'] <= 10


# ===== Stress Tests =====

class TestStress:
    def test_many_vars_feasible(self):
        s = SMTSolver()
        n = 10
        vs = [s.Int(f'v{i}') for i in range(n)]
        for v in vs:
            s.add(v >= 0)
            s.add(v <= 100)
        assert s.check() == SMTResult.SAT

    def test_chain_equalities(self):
        s = SMTSolver()
        n = 8
        vs = [s.Int(f'v{i}') for i in range(n)]
        for i in range(n - 1):
            s.add(vs[i] == vs[i + 1])
        s.add(vs[0] >= 5)
        s.add(vs[-1] <= 5)
        assert s.check() == SMTResult.SAT
        m = s.model()
        for i in range(n):
            assert m[f'v{i}'] == 5

    def test_tight_bounds(self):
        s = SMTSolver()
        x, y = s.Ints('x y')
        s.add(x + y == s.IntVal(10))
        s.add(x >= 4)
        s.add(x <= 6)
        s.add(y >= 4)
        s.add(y <= 6)
        assert s.check() == SMTResult.SAT
        m = s.model()
        assert m['x'] + m['y'] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
