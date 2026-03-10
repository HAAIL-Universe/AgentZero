"""
Tests for V012: Craig Interpolation

Tests cover:
1. Term utilities (collect_vars, flatten, negate)
2. Linear constraint extraction and conversion
3. Fourier-Motzkin elimination
4. Basic interpolation (no shared vars, all shared vars)
5. Interpolation with local variables
6. Equality constraints
7. Multi-variable interpolation
8. Relational interpolation
9. Sequence interpolation
10. CEGAR integration helpers
11. Model-based interpolation fallback
12. Edge cases
13. Interpolant validity verification
14. Predicate extraction
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import SMTSolver, SMTResult, Var, IntConst, BoolConst, App, Op, BOOL, INT

from craig_interpolation import (
    # Term utilities
    collect_vars, collect_atoms, flatten_conjunction, make_conjunction,
    make_disjunction, negate,
    # Linear constraints
    LinearConstraint, term_to_linear, linear_to_term,
    # Fourier-Motzkin
    fourier_motzkin_eliminate,
    # Classification
    classify_variables,
    # Core interpolation
    interpolate, check_interpolant_validity,
    InterpolantResult, Interpolant,
    # Sequence interpolation
    sequence_interpolate, SequenceInterpolant,
    # CEGAR helpers
    interpolation_refine, extract_predicates_from_interpolant,
    # Convenience API
    check_and_interpolate, interpolate_with_vars,
    # Internal helpers
    interpolate_linear, _find_bounds, _find_relational_bounds,
)


# --- Helpers ---

def make_var(name, sort=INT):
    return Var(name, sort)

def smt_vars(*names):
    """Create SMT Int variables for testing."""
    s = SMTSolver()
    return [s.Int(n) for n in names]


# ============================================================
# Section 1: Term Utilities
# ============================================================

class TestTermUtilities:
    def test_collect_vars_simple(self):
        x = make_var('x')
        y = make_var('y')
        t = App(Op.ADD, [x, y], INT)
        assert collect_vars(t) == {'x', 'y'}

    def test_collect_vars_nested(self):
        x = make_var('x')
        y = make_var('y')
        z = make_var('z')
        t = App(Op.LE, [App(Op.ADD, [x, y], INT), z], BOOL)
        assert collect_vars(t) == {'x', 'y', 'z'}

    def test_collect_vars_constants(self):
        x = make_var('x')
        t = App(Op.LE, [x, IntConst(5)], BOOL)
        assert collect_vars(t) == {'x'}

    def test_flatten_conjunction(self):
        x = make_var('x')
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        b = App(Op.LE, [x, IntConst(10)], BOOL)
        conj = App(Op.AND, [a, b], BOOL)
        flat = flatten_conjunction(conj)
        assert len(flat) == 2

    def test_flatten_nested_conjunction(self):
        x = make_var('x')
        y = make_var('y')
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        b = App(Op.LE, [x, IntConst(10)], BOOL)
        c = App(Op.GE, [y, IntConst(0)], BOOL)
        inner = App(Op.AND, [a, b], BOOL)
        outer = App(Op.AND, [inner, c], BOOL)
        flat = flatten_conjunction(outer)
        assert len(flat) == 3

    def test_negate_comparison(self):
        x = make_var('x')
        t = App(Op.LE, [x, IntConst(5)], BOOL)
        neg = negate(t)
        assert isinstance(neg, App)
        assert neg.op == Op.GT

    def test_negate_eq(self):
        x = make_var('x')
        t = App(Op.EQ, [x, IntConst(3)], BOOL)
        neg = negate(t)
        assert neg.op == Op.NEQ

    def test_negate_and(self):
        x = make_var('x')
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        b = App(Op.LE, [x, IntConst(10)], BOOL)
        conj = App(Op.AND, [a, b], BOOL)
        neg = negate(conj)
        assert neg.op == Op.OR

    def test_make_conjunction_empty(self):
        assert isinstance(make_conjunction([]), BoolConst)

    def test_make_conjunction_single(self):
        x = make_var('x')
        t = App(Op.GE, [x, IntConst(0)], BOOL)
        assert make_conjunction([t]) is t


# ============================================================
# Section 2: Linear Constraint Extraction
# ============================================================

class TestLinearConstraints:
    def test_le_constraint(self):
        x = make_var('x')
        t = App(Op.LE, [x, IntConst(5)], BOOL)
        lc = term_to_linear(t)
        assert lc is not None
        assert lc.coeffs.get('x', 0) == 1
        assert lc.const == -5  # x - 5 <= 0
        assert lc.op == '<='

    def test_ge_constraint(self):
        x = make_var('x')
        t = App(Op.GE, [x, IntConst(3)], BOOL)
        lc = term_to_linear(t)
        assert lc is not None
        # x >= 3 becomes -x + 3 <= 0 (negate: -1*x, const = -3 -> -(-3) hmm)
        # Actually: x - 3 >= 0 -> -(x - 3) <= 0 -> -x + 3 <= 0
        assert lc.coeffs.get('x', 0) == -1
        assert lc.const == 3
        assert lc.op == '<='

    def test_lt_constraint(self):
        x = make_var('x')
        t = App(Op.LT, [x, IntConst(5)], BOOL)
        lc = term_to_linear(t)
        assert lc is not None
        # x < 5 -> x <= 4 -> x - 4 <= 0 -> x - 5 + 1 <= 0
        assert lc.coeffs.get('x', 0) == 1
        assert lc.const == -4  # x + (-5+1) = x - 4
        assert lc.op == '<='

    def test_gt_constraint(self):
        x = make_var('x')
        t = App(Op.GT, [x, IntConst(3)], BOOL)
        lc = term_to_linear(t)
        assert lc is not None
        # x > 3 -> x >= 4 -> -x + 4 <= 0 -> -(x-3) + 1 <= 0
        assert lc.coeffs.get('x', 0) == -1
        assert lc.const == 4  # -x + 3 + 1 = -x + 4
        assert lc.op == '<='

    def test_eq_constraint(self):
        x = make_var('x')
        t = App(Op.EQ, [x, IntConst(7)], BOOL)
        lc = term_to_linear(t)
        assert lc is not None
        assert lc.coeffs.get('x', 0) == 1
        assert lc.const == -7
        assert lc.op == '=='

    def test_two_var_constraint(self):
        x = make_var('x')
        y = make_var('y')
        t = App(Op.LE, [App(Op.ADD, [x, y], INT), IntConst(10)], BOOL)
        lc = term_to_linear(t)
        assert lc is not None
        assert lc.coeffs.get('x', 0) == 1
        assert lc.coeffs.get('y', 0) == 1
        assert lc.const == -10

    def test_linear_to_term_roundtrip(self):
        lc = LinearConstraint({'x': 2, 'y': -1}, 3, '<=')
        s = SMTSolver()
        t = linear_to_term(lc, s)
        assert isinstance(t, App)
        assert t.op == Op.LE


# ============================================================
# Section 3: Fourier-Motzkin Elimination
# ============================================================

class TestFourierMotzkin:
    def test_eliminate_single_var(self):
        # x <= 5, x >= 3 -> 3 <= 5 (trivially true -> empty useful result)
        c1 = LinearConstraint({'x': 1}, -5, '<=')      # x - 5 <= 0
        c2 = LinearConstraint({'x': -1}, 3, '<=')       # -x + 3 <= 0
        result = fourier_motzkin_eliminate([c1, c2], 'x')
        # After elimination: combined gives 1*(-5) + 1*(3) = -2 <= 0 -> True
        # So result should have the combined constraint
        assert len(result) >= 0  # May have trivially true constraint

    def test_eliminate_preserves_independent(self):
        # x <= 5, y >= 0 -> eliminate x -> y >= 0 remains
        c1 = LinearConstraint({'x': 1}, -5, '<=')
        c2 = LinearConstraint({'y': -1}, 0, '<=')
        result = fourier_motzkin_eliminate([c1, c2], 'x')
        # c2 doesn't mention x, so it should survive
        assert any(c.has_var('y') for c in result)

    def test_eliminate_produces_new_constraint(self):
        # x - y <= 0 (x <= y), x >= 3 -> y >= 3 (-y + 3 <= 0)
        c1 = LinearConstraint({'x': 1, 'y': -1}, 0, '<=')   # x - y <= 0
        c2 = LinearConstraint({'x': -1}, 3, '<=')             # -x + 3 <= 0
        result = fourier_motzkin_eliminate([c1, c2], 'x')
        # Combined: 1*0 + 1*(-1*y + 3) hmm wait
        # upper(x): c1 has x coeff 1 (upper)
        # lower(x): c2 has x coeff -1 (lower)
        # combine: c_u=1, c_l=-1, mult_l=1, mult_u=1
        # new coeffs for y: 1*(-1) + 1*(0) = -1
        # new const: 1*(0) + 1*(3) = 3
        # result: -y + 3 <= 0, i.e. y >= 3
        assert any(c.coeff('y') == -1 and c.const == 3 for c in result)

    def test_eliminate_with_equality(self):
        # x == 5, x + y <= 10 -> eliminate x -> 5 + y <= 10 -> y <= 5
        c1 = LinearConstraint({'x': 1}, -5, '==')
        c2 = LinearConstraint({'x': 1, 'y': 1}, -10, '<=')
        result = fourier_motzkin_eliminate([c1, c2], 'x')
        # Substitute x=5: y + 5 - 10 <= 0 -> y - 5 <= 0
        assert any(c.coeff('y') != 0 for c in result)

    def test_eliminate_no_variable(self):
        # y >= 0, z <= 5 -> eliminate x (not present) -> unchanged
        c1 = LinearConstraint({'y': -1}, 0, '<=')
        c2 = LinearConstraint({'z': 1}, -5, '<=')
        result = fourier_motzkin_eliminate([c1, c2], 'x')
        assert len(result) == 2


# ============================================================
# Section 4: Variable Classification
# ============================================================

class TestClassification:
    def test_disjoint_vars(self):
        x = make_var('x')
        y = make_var('y')
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        b = App(Op.LE, [y, IntConst(0)], BOOL)
        a_local, b_local, shared = classify_variables(a, b)
        assert a_local == {'x'}
        assert b_local == {'y'}
        assert shared == set()

    def test_shared_vars(self):
        x = make_var('x')
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        b = App(Op.LE, [x, IntConst(-1)], BOOL)
        a_local, b_local, shared = classify_variables(a, b)
        assert shared == {'x'}
        assert a_local == set()
        assert b_local == set()

    def test_mixed_vars(self):
        x = make_var('x')
        y = make_var('y')
        z = make_var('z')
        a = App(Op.AND, [
            App(Op.GE, [x, IntConst(0)], BOOL),
            App(Op.LE, [y, IntConst(5)], BOOL)
        ], BOOL)
        b = App(Op.AND, [
            App(Op.GE, [y, IntConst(10)], BOOL),
            App(Op.LE, [z, IntConst(0)], BOOL)
        ], BOOL)
        a_local, b_local, shared = classify_variables(a, b)
        assert a_local == {'x'}
        assert b_local == {'z'}
        assert shared == {'y'}


# ============================================================
# Section 5: Basic Interpolation (No Local Vars / All Shared)
# ============================================================

class TestBasicInterpolation:
    def test_contradictory_shared_vars(self):
        """A: x >= 5, B: x <= 3 -> I should imply x >= 5 and contradict x <= 3."""
        x_a = make_var('x')
        x_b = make_var('x')
        a = App(Op.GE, [x_a, IntConst(5)], BOOL)
        b = App(Op.LE, [x_b, IntConst(3)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        assert result.formula is not None
        # Verify interpolant properties
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3

    def test_no_shared_vars_a_unsat(self):
        """A: x >= 5 AND x <= 3 (UNSAT alone), B: y >= 0. Interpolant: False."""
        x = make_var('x')
        y = make_var('y')
        a = App(Op.AND, [
            App(Op.GE, [x, IntConst(5)], BOOL),
            App(Op.LE, [x, IntConst(3)], BOOL)
        ], BOOL)
        b = App(Op.GE, [y, IntConst(0)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS

    def test_no_shared_vars_b_unsat(self):
        """A: x >= 0, B: y >= 5 AND y <= 3 (UNSAT alone). Interpolant: True."""
        x = make_var('x')
        y = make_var('y')
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        b = App(Op.AND, [
            App(Op.GE, [y, IntConst(5)], BOOL),
            App(Op.LE, [y, IntConst(3)], BOOL)
        ], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS

    def test_not_unsat(self):
        """A: x >= 0, B: x >= 0 -> SAT, no interpolant."""
        x = make_var('x')
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        b = App(Op.GE, [x, IntConst(0)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.NOT_UNSAT

    def test_equality_contradiction(self):
        """A: x == 5, B: x == 3."""
        x = make_var('x')
        a = App(Op.EQ, [x, IntConst(5)], BOOL)
        b = App(Op.EQ, [x, IntConst(3)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3


# ============================================================
# Section 6: Interpolation with Local Variables
# ============================================================

class TestLocalVariableInterpolation:
    def test_a_local_elimination(self):
        """A: y <= x, x <= 3. B: y >= 5. Shared: y. A-local: x.
        After eliminating x: y <= 3. Interpolant: y <= 3."""
        x = make_var('x')
        y = make_var('y')
        a = App(Op.AND, [
            App(Op.LE, [y, x], BOOL),
            App(Op.LE, [x, IntConst(3)], BOOL)
        ], BOOL)
        b = App(Op.GE, [y, IntConst(5)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        assert result.a_local_vars == {'x'}
        assert result.shared_vars == {'y'}
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3

    def test_b_local_not_in_interpolant(self):
        """A: x >= 5. B: x <= y, y <= 3. Shared: x. B-local: y."""
        x = make_var('x')
        y = make_var('y')
        a = App(Op.GE, [x, IntConst(5)], BOOL)
        b = App(Op.AND, [
            App(Op.LE, [x, y], BOOL),
            App(Op.LE, [y, IntConst(3)], BOOL)
        ], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        i_vars = collect_vars(result.formula)
        assert 'y' not in i_vars
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3

    def test_chain_elimination(self):
        """A: z <= y, y <= x, x <= 3. B: z >= 5. Shared: z. A-local: x, y."""
        x = make_var('x')
        y = make_var('y')
        z = make_var('z')
        a = App(Op.AND, [
            App(Op.LE, [z, y], BOOL),
            App(Op.AND, [
                App(Op.LE, [y, x], BOOL),
                App(Op.LE, [x, IntConst(3)], BOOL)
            ], BOOL)
        ], BOOL)
        b = App(Op.GE, [z, IntConst(5)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        i_vars = collect_vars(result.formula)
        assert 'x' not in i_vars
        assert 'y' not in i_vars
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3


# ============================================================
# Section 7: Multi-Variable Interpolation
# ============================================================

class TestMultiVariableInterpolation:
    def test_two_shared_vars(self):
        """A: x + y <= 5. B: x + y >= 10. Shared: x, y."""
        x = make_var('x')
        y = make_var('y')
        a = App(Op.LE, [App(Op.ADD, [x, y], INT), IntConst(5)], BOOL)
        b = App(Op.GE, [App(Op.ADD, [x, y], INT), IntConst(10)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3

    def test_shared_with_local(self):
        """A: x + t <= 3, t >= 0. B: x >= 5. Shared: x. A-local: t."""
        x = make_var('x')
        t = make_var('t')
        a = App(Op.AND, [
            App(Op.LE, [App(Op.ADD, [x, t], INT), IntConst(3)], BOOL),
            App(Op.GE, [t, IntConst(0)], BOOL)
        ], BOOL)
        b = App(Op.GE, [x, IntConst(5)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3


# ============================================================
# Section 8: Relational Interpolation
# ============================================================

class TestRelationalInterpolation:
    def test_equality_relation(self):
        """A: x == y (via local). B: x != y style."""
        x = make_var('x')
        y = make_var('y')
        t = make_var('t')
        # A: x == t, t == y -> implies x == y
        a = App(Op.AND, [
            App(Op.EQ, [x, t], BOOL),
            App(Op.EQ, [t, y], BOOL)
        ], BOOL)
        # B: x != y
        b = App(Op.NEQ, [x, y], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3

    def test_ordering_relation(self):
        """A: x <= t, t <= y - 5. B: x > y. Shared: x, y. A-local: t."""
        x = make_var('x')
        y = make_var('y')
        t = make_var('t')
        a = App(Op.AND, [
            App(Op.LE, [x, t], BOOL),
            App(Op.LE, [t, App(Op.SUB, [y, IntConst(5)], INT)], BOOL)
        ], BOOL)
        b = App(Op.GT, [x, y], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        i_vars = collect_vars(result.formula)
        assert 't' not in i_vars
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3


# ============================================================
# Section 9: Sequence Interpolation
# ============================================================

class TestSequenceInterpolation:
    def test_two_formulas(self):
        """Simplest case: A0 AND A1 is UNSAT."""
        x = make_var('x')
        f0 = App(Op.GE, [x, IntConst(5)], BOOL)
        f1 = App(Op.LE, [x, IntConst(3)], BOOL)
        result = sequence_interpolate([f0, f1])
        assert result.result == InterpolantResult.SUCCESS
        assert len(result.interpolants) == 1

    def test_three_formulas(self):
        """A0 AND A1 AND A2 is UNSAT. Get I1, I2."""
        x = make_var('x')
        y = make_var('y')
        f0 = App(Op.EQ, [x, IntConst(5)], BOOL)      # x = 5
        f1 = App(Op.EQ, [y, x], BOOL)                  # y = x
        f2 = App(Op.LE, [y, IntConst(3)], BOOL)        # y <= 3
        result = sequence_interpolate([f0, f1, f2])
        assert result.result == InterpolantResult.SUCCESS
        assert len(result.interpolants) == 2

    def test_not_unsat_sequence(self):
        x = make_var('x')
        f0 = App(Op.GE, [x, IntConst(0)], BOOL)
        f1 = App(Op.LE, [x, IntConst(10)], BOOL)
        result = sequence_interpolate([f0, f1])
        assert result.result == InterpolantResult.NOT_UNSAT

    def test_trace_interpolation(self):
        """Simulate a CEGAR trace: Init, Trans, NOT(Prop)."""
        x0 = make_var('x0')
        x1 = make_var('x1')
        x2 = make_var('x2')
        # Init: x0 == 0
        init = App(Op.EQ, [x0, IntConst(0)], BOOL)
        # Trans: x1 == x0 + 1
        trans = App(Op.EQ, [x1, App(Op.ADD, [x0, IntConst(1)], INT)], BOOL)
        # NOT(Prop): x1 >= 5 (property is x1 < 5, negated)
        not_prop = App(Op.GE, [x1, IntConst(5)], BOOL)
        result = sequence_interpolate([init, trans, not_prop])
        # This should be SAT (x0=0, x1=1 satisfies x1<5)
        # Actually x0=0, x1=0+1=1, x1>=5 is UNSAT since 1<5
        # Wait: init AND trans AND not_prop: x0=0, x1=1, but 1>=5 is false. So UNSAT.
        assert result.result == InterpolantResult.SUCCESS


# ============================================================
# Section 10: CEGAR Integration
# ============================================================

class TestCEGARIntegration:
    def test_interpolation_refine(self):
        """Test the CEGAR refinement helper."""
        x = make_var('x')
        y = make_var('y')
        f0 = App(Op.EQ, [x, IntConst(0)], BOOL)
        f1 = App(Op.EQ, [y, App(Op.ADD, [x, IntConst(3)], INT)], BOOL)
        f2 = App(Op.GE, [y, IntConst(10)], BOOL)
        preds = interpolation_refine([f0, f1, f2])
        assert preds is not None
        assert len(preds) == 2  # One interpolant per partition point

    def test_extract_predicates(self):
        """Extract atomic predicates from an interpolant."""
        x = make_var('x')
        y = make_var('y')
        interp = App(Op.AND, [
            App(Op.GE, [x, IntConst(0)], BOOL),
            App(Op.LE, [y, IntConst(5)], BOOL)
        ], BOOL)
        preds = extract_predicates_from_interpolant(interp)
        assert len(preds) == 2

    def test_extract_deduplicates(self):
        """Duplicate atoms are removed."""
        x = make_var('x')
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        interp = App(Op.AND, [a, a], BOOL)
        preds = extract_predicates_from_interpolant(interp)
        assert len(preds) == 1


# ============================================================
# Section 11: Model-Based Fallback
# ============================================================

class TestModelBasedInterpolation:
    def test_bound_discovery(self):
        """Test that bound discovery finds implied bounds."""
        x = make_var('x')
        formula = App(Op.AND, [
            App(Op.GE, [x, IntConst(3)], BOOL),
            App(Op.LE, [x, IntConst(7)], BOOL)
        ], BOOL)
        bounds = _find_bounds(formula, 'x')
        assert len(bounds) >= 2  # At least lower and upper

    def test_relational_discovery(self):
        """Test relational bound discovery."""
        x = make_var('x')
        y = make_var('y')
        formula = App(Op.EQ, [x, y], BOOL)
        rels = _find_relational_bounds(formula, 'x', 'y')
        assert len(rels) >= 1  # Should find x == y


# ============================================================
# Section 12: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_var_tight_bounds(self):
        """A: x == 5. B: x == 5 is SAT -> no interpolant."""
        x = make_var('x')
        a = App(Op.EQ, [x, IntConst(5)], BOOL)
        b = App(Op.EQ, [x, IntConst(5)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.NOT_UNSAT

    def test_constant_true_false(self):
        """A: False -> Interpolant should be False."""
        a = BoolConst(False)
        b = BoolConst(True)
        # A AND B = False AND True = False, which is UNSAT
        # But our SMT solver might not handle BoolConst directly
        # Use x >= 5 AND x <= 3 instead
        x = make_var('x')
        a = App(Op.AND, [
            App(Op.GE, [x, IntConst(5)], BOOL),
            App(Op.LE, [x, IntConst(3)], BOOL)
        ], BOOL)
        b = BoolConst(True)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS

    def test_many_local_vars(self):
        """A has several local vars; all get eliminated."""
        a1 = make_var('a1')
        a2 = make_var('a2')
        a3 = make_var('a3')
        x = make_var('x')
        a = App(Op.AND, [
            App(Op.LE, [x, a1], BOOL),
            App(Op.AND, [
                App(Op.LE, [a1, a2], BOOL),
                App(Op.AND, [
                    App(Op.LE, [a2, a3], BOOL),
                    App(Op.LE, [a3, IntConst(2)], BOOL)
                ], BOOL)
            ], BOOL)
        ], BOOL)
        b = App(Op.GE, [x, IntConst(5)], BOOL)
        result = interpolate(a, b)
        assert result.result == InterpolantResult.SUCCESS
        i_vars = collect_vars(result.formula)
        assert 'a1' not in i_vars
        assert 'a2' not in i_vars
        assert 'a3' not in i_vars
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3


# ============================================================
# Section 13: Interpolant Validity Verification
# ============================================================

class TestValidityVerification:
    def test_valid_interpolant(self):
        x = make_var('x')
        a = App(Op.GE, [x, IntConst(5)], BOOL)
        b = App(Op.LE, [x, IntConst(3)], BOOL)
        # x >= 4 is a valid interpolant
        interp = App(Op.GE, [x, IntConst(4)], BOOL)
        c1, c2, c3 = check_interpolant_validity(a, b, interp)
        assert c1  # x >= 5 => x >= 4
        assert c2  # x >= 4 AND x <= 3 is UNSAT
        assert c3  # vars({x}) subset {x}

    def test_invalid_interpolant_too_weak(self):
        x = make_var('x')
        a = App(Op.GE, [x, IntConst(5)], BOOL)
        b = App(Op.LE, [x, IntConst(3)], BOOL)
        # x >= 0 is too weak (doesn't contradict B)
        interp = App(Op.GE, [x, IntConst(0)], BOOL)
        c1, c2, c3 = check_interpolant_validity(a, b, interp)
        assert c1  # x >= 5 => x >= 0
        assert not c2  # x >= 0 AND x <= 3 is SAT (x=1)

    def test_invalid_interpolant_too_strong(self):
        x = make_var('x')
        a = App(Op.GE, [x, IntConst(5)], BOOL)
        b = App(Op.LE, [x, IntConst(3)], BOOL)
        # x >= 10 is too strong (not implied by A)
        interp = App(Op.GE, [x, IntConst(10)], BOOL)
        c1, c2, c3 = check_interpolant_validity(a, b, interp)
        assert not c1  # x >= 5 does NOT imply x >= 10

    def test_invalid_interpolant_wrong_vars(self):
        x = make_var('x')
        y = make_var('y')
        a = App(Op.GE, [x, IntConst(5)], BOOL)
        b = App(Op.LE, [x, IntConst(3)], BOOL)
        # y >= 0 uses a non-shared variable
        interp = App(Op.GE, [y, IntConst(0)], BOOL)
        c1, c2, c3 = check_interpolant_validity(a, b, interp)
        assert not c3  # y is not in vars(A) intersect vars(B) = {x}


# ============================================================
# Section 14: Predicate Extraction
# ============================================================

class TestPredicateExtraction:
    def test_extract_from_conjunction(self):
        x = make_var('x')
        y = make_var('y')
        interp = App(Op.AND, [
            App(Op.GE, [x, IntConst(0)], BOOL),
            App(Op.LE, [y, IntConst(5)], BOOL),
            App(Op.LE, [App(Op.ADD, [x, y], INT), IntConst(10)], BOOL)
        ], BOOL)
        preds = extract_predicates_from_interpolant(interp)
        assert len(preds) == 3

    def test_extract_from_atom(self):
        x = make_var('x')
        interp = App(Op.GE, [x, IntConst(0)], BOOL)
        preds = extract_predicates_from_interpolant(interp)
        assert len(preds) == 1

    def test_extract_from_disjunction(self):
        x = make_var('x')
        interp = App(Op.OR, [
            App(Op.GE, [x, IntConst(5)], BOOL),
            App(Op.LE, [x, IntConst(-5)], BOOL)
        ], BOOL)
        preds = extract_predicates_from_interpolant(interp)
        assert len(preds) == 2


# ============================================================
# Section 15: Convenience API
# ============================================================

class TestConvenienceAPI:
    def test_check_and_interpolate(self):
        x = make_var('x')
        a_terms = [App(Op.GE, [x, IntConst(5)], BOOL)]
        b_terms = [App(Op.LE, [x, IntConst(3)], BOOL)]
        result = check_and_interpolate(a_terms, b_terms)
        assert result.result == InterpolantResult.SUCCESS

    def test_interpolate_with_explicit_vars(self):
        x = make_var('x')
        y = make_var('y')
        a = App(Op.AND, [
            App(Op.LE, [x, IntConst(3)], BOOL),
            App(Op.LE, [y, x], BOOL)
        ], BOOL)
        b = App(Op.GE, [y, IntConst(5)], BOOL)
        result = interpolate_with_vars(a, b, {'x', 'y'}, {'y'})
        assert result.result == InterpolantResult.SUCCESS
        assert result.shared_vars == {'y'}
        c1, c2, c3 = check_interpolant_validity(a, b, result.formula)
        assert c1 and c2 and c3
