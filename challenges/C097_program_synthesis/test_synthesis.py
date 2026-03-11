"""Tests for C097: Program Synthesis Engine."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from synthesis import (
    # DSL
    Expr, IntConst, BoolConst, VarExpr, UnaryOp, BinOp, IfExpr,
    Type, evaluate, EvalError, expr_size, expr_depth, uses_variable, get_variables,
    # Spec
    IOExample, SynthesisSpec, SynthesisResult,
    # OE
    ObservationalEquivalence,
    # Synthesizers
    EnumerativeSynthesizer, ConstraintSynthesizer, CEGISSynthesizer,
    ComponentSynthesizer, OracleSynthesizer, ConditionalSynthesizer,
    # Components
    Component, ARITHMETIC_COMPONENTS, COMPARISON_COMPONENTS,
    CONDITIONAL_COMPONENTS, EXTENDED_COMPONENTS,
    # Utilities
    programs_equivalent, simplify, pretty_print, synthesize,
)


# ============================================================
# DSL Expression Tests
# ============================================================

class TestExpressions:
    def test_int_const(self):
        e = IntConst(42)
        assert evaluate(e, {}) == 42
        assert repr(e) == "42"

    def test_bool_const(self):
        e = BoolConst(True)
        assert evaluate(e, {}) is True
        assert repr(e) == "true"

    def test_bool_const_false(self):
        e = BoolConst(False)
        assert evaluate(e, {}) is False
        assert repr(e) == "false"

    def test_var_expr(self):
        e = VarExpr('x')
        assert evaluate(e, {'x': 10}) == 10
        assert repr(e) == "x"

    def test_var_undefined(self):
        e = VarExpr('y')
        with pytest.raises(EvalError):
            evaluate(e, {'x': 1})

    def test_unary_neg(self):
        e = UnaryOp('neg', IntConst(5))
        assert evaluate(e, {}) == -5

    def test_unary_not(self):
        e = UnaryOp('not', BoolConst(True))
        assert evaluate(e, {}) is False

    def test_unary_abs(self):
        e = UnaryOp('abs', IntConst(-7))
        assert evaluate(e, {}) == 7

    def test_binary_add(self):
        e = BinOp('+', VarExpr('x'), IntConst(1))
        assert evaluate(e, {'x': 5}) == 6

    def test_binary_sub(self):
        e = BinOp('-', VarExpr('x'), VarExpr('y'))
        assert evaluate(e, {'x': 10, 'y': 3}) == 7

    def test_binary_mul(self):
        e = BinOp('*', IntConst(3), IntConst(4))
        assert evaluate(e, {}) == 12

    def test_binary_div(self):
        e = BinOp('/', IntConst(7), IntConst(2))
        assert evaluate(e, {}) == 3

    def test_binary_div_negative(self):
        # Division toward zero
        e = BinOp('/', IntConst(-7), IntConst(2))
        assert evaluate(e, {}) == -3

    def test_binary_div_by_zero(self):
        e = BinOp('/', IntConst(5), IntConst(0))
        with pytest.raises(EvalError):
            evaluate(e, {})

    def test_binary_mod(self):
        e = BinOp('%', IntConst(7), IntConst(3))
        assert evaluate(e, {}) == 1

    def test_binary_mod_by_zero(self):
        e = BinOp('%', IntConst(5), IntConst(0))
        with pytest.raises(EvalError):
            evaluate(e, {})

    def test_binary_max(self):
        e = BinOp('max', IntConst(3), IntConst(7))
        assert evaluate(e, {}) == 7

    def test_binary_min(self):
        e = BinOp('min', IntConst(3), IntConst(7))
        assert evaluate(e, {}) == 3

    def test_comparison_eq(self):
        e = BinOp('==', IntConst(3), IntConst(3))
        assert evaluate(e, {}) is True

    def test_comparison_neq(self):
        e = BinOp('!=', IntConst(3), IntConst(4))
        assert evaluate(e, {}) is True

    def test_comparison_lt(self):
        e = BinOp('<', IntConst(3), IntConst(4))
        assert evaluate(e, {}) is True

    def test_comparison_le(self):
        e = BinOp('<=', IntConst(4), IntConst(4))
        assert evaluate(e, {}) is True

    def test_comparison_gt(self):
        e = BinOp('>', IntConst(5), IntConst(3))
        assert evaluate(e, {}) is True

    def test_comparison_ge(self):
        e = BinOp('>=', IntConst(3), IntConst(3))
        assert evaluate(e, {}) is True

    def test_boolean_and(self):
        e = BinOp('and', BoolConst(True), BoolConst(False))
        assert evaluate(e, {}) is False

    def test_boolean_or(self):
        e = BinOp('or', BoolConst(False), BoolConst(True))
        assert evaluate(e, {}) is True

    def test_if_expr_true(self):
        e = IfExpr(BoolConst(True), IntConst(1), IntConst(2))
        assert evaluate(e, {}) == 1

    def test_if_expr_false(self):
        e = IfExpr(BoolConst(False), IntConst(1), IntConst(2))
        assert evaluate(e, {}) == 2

    def test_if_expr_with_comparison(self):
        e = IfExpr(BinOp('<', VarExpr('x'), IntConst(0)),
                   UnaryOp('neg', VarExpr('x')),
                   VarExpr('x'))
        assert evaluate(e, {'x': -5}) == 5
        assert evaluate(e, {'x': 3}) == 3

    def test_nested_expr(self):
        # (x + 1) * (y - 2)
        e = BinOp('*', BinOp('+', VarExpr('x'), IntConst(1)),
                       BinOp('-', VarExpr('y'), IntConst(2)))
        assert evaluate(e, {'x': 3, 'y': 5}) == 12

    def test_repr_binop(self):
        e = BinOp('+', VarExpr('x'), IntConst(1))
        assert repr(e) == "(x + 1)"

    def test_repr_max(self):
        e = BinOp('max', VarExpr('x'), VarExpr('y'))
        assert repr(e) == "max(x, y)"

    def test_repr_if(self):
        e = IfExpr(BoolConst(True), IntConst(1), IntConst(2))
        assert "if" in repr(e)


# ============================================================
# Expression Utility Tests
# ============================================================

class TestExprUtils:
    def test_expr_size_leaf(self):
        assert expr_size(IntConst(1)) == 1
        assert expr_size(VarExpr('x')) == 1

    def test_expr_size_unary(self):
        e = UnaryOp('neg', VarExpr('x'))
        assert expr_size(e) == 2

    def test_expr_size_binary(self):
        e = BinOp('+', VarExpr('x'), IntConst(1))
        assert expr_size(e) == 3

    def test_expr_size_nested(self):
        e = BinOp('+', BinOp('*', VarExpr('x'), IntConst(2)), IntConst(1))
        assert expr_size(e) == 5

    def test_expr_size_if(self):
        e = IfExpr(BoolConst(True), IntConst(1), IntConst(2))
        assert expr_size(e) == 4

    def test_expr_depth_leaf(self):
        assert expr_depth(IntConst(1)) == 0

    def test_expr_depth_unary(self):
        assert expr_depth(UnaryOp('neg', VarExpr('x'))) == 1

    def test_expr_depth_binary(self):
        assert expr_depth(BinOp('+', VarExpr('x'), IntConst(1))) == 1

    def test_expr_depth_nested(self):
        e = BinOp('+', BinOp('*', VarExpr('x'), IntConst(2)), IntConst(1))
        assert expr_depth(e) == 2

    def test_uses_variable_yes(self):
        e = BinOp('+', VarExpr('x'), IntConst(1))
        assert uses_variable(e, 'x') is True

    def test_uses_variable_no(self):
        e = BinOp('+', VarExpr('x'), IntConst(1))
        assert uses_variable(e, 'y') is False

    def test_get_variables(self):
        e = BinOp('+', VarExpr('x'), BinOp('*', VarExpr('y'), VarExpr('x')))
        assert get_variables(e) == {'x', 'y'}

    def test_get_variables_const(self):
        assert get_variables(IntConst(5)) == set()

    def test_get_variables_if(self):
        e = IfExpr(BinOp('<', VarExpr('a'), IntConst(0)), VarExpr('b'), VarExpr('c'))
        assert get_variables(e) == {'a', 'b', 'c'}


# ============================================================
# Observational Equivalence Tests
# ============================================================

class TestObservationalEquivalence:
    def test_basic_signature(self):
        examples = [IOExample({'x': 1}, 2), IOExample({'x': 2}, 3)]
        oe = ObservationalEquivalence(examples)
        sig = oe.signature(BinOp('+', VarExpr('x'), IntConst(1)))
        assert sig == (2, 3)

    def test_equivalent_programs_same_sig(self):
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 1}, 1)]
        oe = ObservationalEquivalence(examples)

        # x and x+0 are equivalent
        sig1 = oe.signature(VarExpr('x'))
        sig2 = oe.signature(BinOp('+', VarExpr('x'), IntConst(0)))
        assert sig1 == sig2

    def test_is_new_first(self):
        examples = [IOExample({'x': 1}, 2)]
        oe = ObservationalEquivalence(examples)
        assert oe.is_new(VarExpr('x')) is True

    def test_is_new_duplicate(self):
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 1}, 1)]
        oe = ObservationalEquivalence(examples)
        oe.add(VarExpr('x'))
        # x+0 is equivalent to x, so not new
        assert oe.is_new(BinOp('+', VarExpr('x'), IntConst(0))) is False

    def test_keeps_smallest(self):
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 1}, 1)]
        oe = ObservationalEquivalence(examples)
        oe.add(VarExpr('x'))  # size 1
        sig = oe.signature(VarExpr('x'))
        rep = oe.get_representative(sig)
        assert isinstance(rep, VarExpr)

    def test_num_classes(self):
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 1}, 1)]
        oe = ObservationalEquivalence(examples)
        oe.add(VarExpr('x'))        # class 1: (0, 1)
        oe.add(IntConst(0))         # class 2: (0, 0)
        oe.add(IntConst(1))         # class 3: (1, 1)
        assert oe.num_classes == 3

    def test_eval_error_returns_none(self):
        examples = [IOExample({'x': 0}, 0)]
        oe = ObservationalEquivalence(examples)
        # Division by zero -> signature is None -> not new
        expr = BinOp('/', IntConst(1), IntConst(0))
        assert oe.signature(expr) is None
        assert oe.is_new(expr) is False


# ============================================================
# Enumerative Synthesis Tests
# ============================================================

class TestEnumerativeSynthesis:
    def test_identity(self):
        """Synthesize f(x) = x"""
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 5}, 5), IOExample({'x': -3}, -3)]
        spec = SynthesisSpec(examples, ['x'])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 7}) == 7

    def test_increment(self):
        """Synthesize f(x) = x + 1"""
        examples = [IOExample({'x': 0}, 1), IOExample({'x': 5}, 6), IOExample({'x': -1}, 0)]
        spec = SynthesisSpec(examples, ['x'])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 10}) == 11

    def test_double(self):
        """Synthesize f(x) = 2*x"""
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 1}, 2),
                     IOExample({'x': 3}, 6), IOExample({'x': -2}, -4)]
        spec = SynthesisSpec(examples, ['x'], constants=[0, 1, 2])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        for v in [-5, 0, 5, 10]:
            assert evaluate(result.program, {'x': v}) == 2 * v

    def test_add_two_vars(self):
        """Synthesize f(x, y) = x + y"""
        examples = [
            IOExample({'x': 1, 'y': 2}, 3),
            IOExample({'x': 0, 'y': 0}, 0),
            IOExample({'x': -1, 'y': 1}, 0),
            IOExample({'x': 3, 'y': 4}, 7),
        ]
        spec = SynthesisSpec(examples, ['x', 'y'])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 10, 'y': 20}) == 30

    def test_subtract(self):
        """Synthesize f(x, y) = x - y"""
        examples = [
            IOExample({'x': 5, 'y': 3}, 2),
            IOExample({'x': 0, 'y': 0}, 0),
            IOExample({'x': 1, 'y': 5}, -4),
        ]
        spec = SynthesisSpec(examples, ['x', 'y'])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 10, 'y': 3}) == 7

    def test_square(self):
        """Synthesize f(x) = x * x"""
        examples = [
            IOExample({'x': 0}, 0),
            IOExample({'x': 1}, 1),
            IOExample({'x': 2}, 4),
            IOExample({'x': 3}, 9),
            IOExample({'x': -2}, 4),
        ]
        spec = SynthesisSpec(examples, ['x'])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 5}) == 25

    def test_constant_function(self):
        """Synthesize f(x) = 0"""
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 1}, 0), IOExample({'x': -5}, 0)]
        spec = SynthesisSpec(examples, ['x'])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 999}) == 0

    def test_result_metadata(self):
        examples = [IOExample({'x': 0}, 1)]
        spec = SynthesisSpec(examples, ['x'])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        assert result.method == "enumerative"
        assert result.candidates_explored > 0

    def test_x_plus_y_times_z(self):
        """Synthesize f(x,y) = x + y (different examples)"""
        examples = [
            IOExample({'x': 2, 'y': 3}, 5),
            IOExample({'x': 0, 'y': 0}, 0),
            IOExample({'x': -1, 'y': -2}, -3),
        ]
        spec = SynthesisSpec(examples, ['x', 'y'])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success

    def test_negation(self):
        """Synthesize f(x) = -x"""
        examples = [
            IOExample({'x': 0}, 0),
            IOExample({'x': 1}, -1),
            IOExample({'x': -3}, 3),
        ]
        spec = SynthesisSpec(examples, ['x'], components=['+', '-', '*'])
        synth = EnumerativeSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 7}) == -7


# ============================================================
# Constraint-Based Synthesis Tests
# ============================================================

class TestConstraintSynthesis:
    def test_identity_constraint(self):
        """Synthesize f(x) = x via SMT constraints."""
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 5}, 5), IOExample({'x': -3}, -3)]
        spec = SynthesisSpec(examples, ['x'])
        synth = ConstraintSynthesizer(spec, max_nodes=3)
        result = synth.synthesize()
        assert result.success
        assert result.method == "constraint"
        # Verify the synthesized program works
        assert evaluate(result.program, {'x': 7}) == 7

    def test_increment_constraint(self):
        """Synthesize f(x) = x + 1 via SMT."""
        examples = [IOExample({'x': 0}, 1), IOExample({'x': 3}, 4), IOExample({'x': -1}, 0)]
        spec = SynthesisSpec(examples, ['x'])
        synth = ConstraintSynthesizer(spec, max_nodes=4)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 10}) == 11

    def test_sum_constraint(self):
        """Synthesize f(x,y) = x + y via SMT."""
        examples = [
            IOExample({'x': 1, 'y': 2}, 3),
            IOExample({'x': 0, 'y': 0}, 0),
            IOExample({'x': 3, 'y': -1}, 2),
        ]
        spec = SynthesisSpec(examples, ['x', 'y'])
        synth = ConstraintSynthesizer(spec, max_nodes=4)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 5, 'y': 5}) == 10

    def test_constant_constraint(self):
        """Synthesize f(x) = 1 via SMT."""
        examples = [IOExample({'x': 0}, 1), IOExample({'x': 5}, 1)]
        spec = SynthesisSpec(examples, ['x'])
        synth = ConstraintSynthesizer(spec, max_nodes=3)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 999}) == 1


# ============================================================
# CEGIS Tests
# ============================================================

class TestCEGIS:
    def test_cegis_with_oracle_identity(self):
        """CEGIS with oracle for f(x) = x."""
        oracle = lambda inputs: inputs['x']
        examples = [IOExample({'x': 0}, 0)]
        spec = SynthesisSpec(examples, ['x'])
        synth = CEGISSynthesizer(spec, oracle=oracle, max_size=6)
        result = synth.synthesize()
        assert result.success
        assert result.method == "cegis"

    def test_cegis_with_oracle_double(self):
        """CEGIS with oracle for f(x) = 2*x."""
        oracle = lambda inputs: 2 * inputs['x']
        examples = [IOExample({'x': 1}, 2)]
        spec = SynthesisSpec(examples, ['x'], constants=[0, 1, 2])
        synth = CEGISSynthesizer(spec, oracle=oracle, max_size=6)
        result = synth.synthesize()
        assert result.success
        # Verify on wider range
        for v in range(-10, 11):
            assert evaluate(result.program, {'x': v}) == 2 * v

    def test_cegis_with_oracle_sum(self):
        """CEGIS with oracle for f(x,y) = x + y."""
        oracle = lambda inputs: inputs['x'] + inputs['y']
        examples = [IOExample({'x': 1, 'y': 2}, 3)]
        spec = SynthesisSpec(examples, ['x', 'y'])
        synth = CEGISSynthesizer(spec, oracle=oracle, max_size=6)
        result = synth.synthesize()
        assert result.success

    def test_cegis_with_verifier(self):
        """CEGIS with custom verifier."""
        # Synthesize x + 1, verifier checks specific inputs
        test_cases = [
            ({'x': 0}, 1), ({'x': 5}, 6), ({'x': -3}, -2),
            ({'x': 10}, 11), ({'x': -10}, -9),
        ]
        def verifier(candidate):
            for inputs, expected in test_cases:
                try:
                    actual = evaluate(candidate, inputs)
                    if actual != expected:
                        return IOExample(inputs, expected)
                except EvalError:
                    return IOExample(inputs, expected)
            return None

        examples = [IOExample({'x': 0}, 1)]
        spec = SynthesisSpec(examples, ['x'])
        synth = CEGISSynthesizer(spec, verifier=verifier, max_size=6)
        result = synth.synthesize()
        assert result.success

    def test_cegis_iterations(self):
        """CEGIS may need multiple iterations."""
        oracle = lambda inputs: inputs['x'] * inputs['x']
        examples = [IOExample({'x': 0}, 0)]
        spec = SynthesisSpec(examples, ['x'])
        synth = CEGISSynthesizer(spec, oracle=oracle, max_size=6)
        result = synth.synthesize()
        assert result.success
        assert result.iterations >= 1


# ============================================================
# Component-Based Synthesis Tests
# ============================================================

class TestComponentSynthesis:
    def test_component_identity(self):
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 5}, 5)]
        spec = SynthesisSpec(examples, ['x'])
        synth = ComponentSynthesizer(spec, ARITHMETIC_COMPONENTS)
        result = synth.synthesize()
        assert result.success
        assert result.method == "component"

    def test_component_with_comparisons(self):
        """Synthesize a comparison expression."""
        examples = [
            IOExample({'x': 0}, True),
            IOExample({'x': 1}, False),
            IOExample({'x': -1}, True),
        ]
        spec = SynthesisSpec(examples, ['x'], constants=[0, 1])
        comps = ARITHMETIC_COMPONENTS + COMPARISON_COMPONENTS
        synth = ComponentSynthesizer(spec, comps)
        result = synth.synthesize()
        assert result.success

    def test_component_increment(self):
        examples = [IOExample({'x': 0}, 1), IOExample({'x': 5}, 6)]
        spec = SynthesisSpec(examples, ['x'])
        synth = ComponentSynthesizer(spec, ARITHMETIC_COMPONENTS)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 10}) == 11

    def test_component_with_ite(self):
        """Synthesize conditional using ITE component."""
        examples = [
            IOExample({'x': -3}, 3),
            IOExample({'x': 0}, 0),
            IOExample({'x': 5}, 5),
        ]
        spec = SynthesisSpec(examples, ['x'], constants=[0])
        comps = ARITHMETIC_COMPONENTS + COMPARISON_COMPONENTS + CONDITIONAL_COMPONENTS
        synth = ComponentSynthesizer(spec, comps, max_depth=3, max_size=10)
        result = synth.synthesize()
        assert result.success
        # Should synthesize abs(x) or if(x<0, -x, x)
        assert evaluate(result.program, {'x': -7}) == 7
        assert evaluate(result.program, {'x': 3}) == 3

    def test_component_extended(self):
        """Use extended components (max, min, abs)."""
        examples = [
            IOExample({'x': 3, 'y': 7}, 7),
            IOExample({'x': 10, 'y': 2}, 10),
            IOExample({'x': -1, 'y': -5}, -1),
        ]
        spec = SynthesisSpec(examples, ['x', 'y'])
        synth = ComponentSynthesizer(spec, EXTENDED_COMPONENTS)
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 100, 'y': 1}) == 100


# ============================================================
# Oracle-Guided Synthesis Tests
# ============================================================

class TestOracleSynthesis:
    def test_oracle_identity(self):
        oracle = lambda inputs: inputs['x']
        synth = OracleSynthesizer(oracle, ['x'])
        result = synth.synthesize()
        assert result.success
        assert result.method == "oracle"

    def test_oracle_double(self):
        oracle = lambda inputs: 2 * inputs['x']
        synth = OracleSynthesizer(oracle, ['x'], constants=[0, 1, 2])
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 7}) == 14

    def test_oracle_sum(self):
        oracle = lambda inputs: inputs['x'] + inputs['y']
        synth = OracleSynthesizer(oracle, ['x', 'y'])
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 3, 'y': 4}) == 7

    def test_oracle_subtract(self):
        oracle = lambda inputs: inputs['x'] - inputs['y']
        synth = OracleSynthesizer(oracle, ['x', 'y'])
        result = synth.synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 10, 'y': 3}) == 7


# ============================================================
# Conditional Synthesis Tests
# ============================================================

class TestConditionalSynthesis:
    def test_abs_value(self):
        """Synthesize abs(x) via conditional."""
        examples = [
            IOExample({'x': -5}, 5),
            IOExample({'x': 0}, 0),
            IOExample({'x': 3}, 3),
            IOExample({'x': -1}, 1),
        ]
        spec = SynthesisSpec(examples, ['x'], components=['+', '-', '*', '<', '<=', '>', '>=', '=='])
        synth = ConditionalSynthesizer(spec)
        result = synth.synthesize()
        assert result.success
        assert result.method == "conditional"

    def test_max_two(self):
        """Synthesize max(x, y) via conditional."""
        examples = [
            IOExample({'x': 1, 'y': 2}, 2),
            IOExample({'x': 5, 'y': 3}, 5),
            IOExample({'x': -1, 'y': -3}, -1),
            IOExample({'x': 0, 'y': 0}, 0),
        ]
        spec = SynthesisSpec(examples, ['x', 'y'],
                              components=['+', '-', '*', '<', '<=', '>', '>='])
        synth = ConditionalSynthesizer(spec)
        result = synth.synthesize()
        assert result.success

    def test_simple_no_conditional_needed(self):
        """If no conditional needed, still succeeds."""
        examples = [IOExample({'x': 1}, 2), IOExample({'x': 3}, 4)]
        spec = SynthesisSpec(examples, ['x'])
        synth = ConditionalSynthesizer(spec)
        result = synth.synthesize()
        assert result.success


# ============================================================
# Simplification Tests
# ============================================================

class TestSimplification:
    def test_constant_fold_add(self):
        e = BinOp('+', IntConst(2), IntConst(3))
        s = simplify(e)
        assert isinstance(s, IntConst) and s.value == 5

    def test_constant_fold_mul(self):
        e = BinOp('*', IntConst(3), IntConst(4))
        s = simplify(e)
        assert isinstance(s, IntConst) and s.value == 12

    def test_identity_add_zero(self):
        e = BinOp('+', VarExpr('x'), IntConst(0))
        s = simplify(e)
        assert isinstance(s, VarExpr) and s.name == 'x'

    def test_identity_zero_add(self):
        e = BinOp('+', IntConst(0), VarExpr('x'))
        s = simplify(e)
        assert isinstance(s, VarExpr) and s.name == 'x'

    def test_identity_sub_zero(self):
        e = BinOp('-', VarExpr('x'), IntConst(0))
        s = simplify(e)
        assert isinstance(s, VarExpr)

    def test_sub_self(self):
        e = BinOp('-', VarExpr('x'), VarExpr('x'))
        s = simplify(e)
        assert isinstance(s, IntConst) and s.value == 0

    def test_mul_one(self):
        e = BinOp('*', VarExpr('x'), IntConst(1))
        s = simplify(e)
        assert isinstance(s, VarExpr)

    def test_mul_zero(self):
        e = BinOp('*', VarExpr('x'), IntConst(0))
        s = simplify(e)
        assert isinstance(s, IntConst) and s.value == 0

    def test_double_neg(self):
        e = UnaryOp('neg', UnaryOp('neg', VarExpr('x')))
        s = simplify(e)
        assert isinstance(s, VarExpr) and s.name == 'x'

    def test_double_not(self):
        e = UnaryOp('not', UnaryOp('not', BoolConst(True)))
        s = simplify(e)
        assert isinstance(s, BoolConst) and s.value is True

    def test_neg_const(self):
        e = UnaryOp('neg', IntConst(5))
        s = simplify(e)
        assert isinstance(s, IntConst) and s.value == -5

    def test_abs_const(self):
        e = UnaryOp('abs', IntConst(-7))
        s = simplify(e)
        assert isinstance(s, IntConst) and s.value == 7

    def test_if_true(self):
        e = IfExpr(BoolConst(True), IntConst(1), IntConst(2))
        s = simplify(e)
        assert isinstance(s, IntConst) and s.value == 1

    def test_if_false(self):
        e = IfExpr(BoolConst(False), IntConst(1), IntConst(2))
        s = simplify(e)
        assert isinstance(s, IntConst) and s.value == 2

    def test_if_same_branches(self):
        e = IfExpr(VarExpr('c'), IntConst(5), IntConst(5))
        s = simplify(e)
        assert isinstance(s, IntConst) and s.value == 5

    def test_nested_simplify(self):
        # (x + 0) * 1 -> x
        e = BinOp('*', BinOp('+', VarExpr('x'), IntConst(0)), IntConst(1))
        s = simplify(e)
        assert isinstance(s, VarExpr) and s.name == 'x'

    def test_constant_fold_comparison(self):
        e = BinOp('<', IntConst(3), IntConst(5))
        s = simplify(e)
        assert isinstance(s, BoolConst) and s.value is True


# ============================================================
# Program Equivalence Tests
# ============================================================

class TestProgramEquivalence:
    def test_equivalent_identity(self):
        e1 = VarExpr('x')
        e2 = BinOp('+', VarExpr('x'), IntConst(0))
        assert programs_equivalent(e1, e2, ['x']) is True

    def test_not_equivalent(self):
        e1 = VarExpr('x')
        e2 = BinOp('+', VarExpr('x'), IntConst(1))
        assert programs_equivalent(e1, e2, ['x']) is False

    def test_equivalent_two_vars(self):
        e1 = BinOp('+', VarExpr('x'), VarExpr('y'))
        e2 = BinOp('+', VarExpr('y'), VarExpr('x'))
        assert programs_equivalent(e1, e2, ['x', 'y']) is True

    def test_equivalent_algebraic(self):
        # 2*x vs x+x
        e1 = BinOp('*', IntConst(2), VarExpr('x'))
        e2 = BinOp('+', VarExpr('x'), VarExpr('x'))
        assert programs_equivalent(e1, e2, ['x']) is True


# ============================================================
# Pretty Printer Tests
# ============================================================

class TestPrettyPrint:
    def test_leaf(self):
        assert pretty_print(IntConst(42)).strip() == "42"

    def test_binop(self):
        e = BinOp('+', VarExpr('x'), IntConst(1))
        pp = pretty_print(e)
        assert '+' in pp
        assert 'x' in pp

    def test_if_expr(self):
        e = IfExpr(BoolConst(True), IntConst(1), IntConst(2))
        pp = pretty_print(e)
        assert 'if' in pp

    def test_nested(self):
        e = BinOp('+', BinOp('*', VarExpr('x'), IntConst(2)), IntConst(1))
        pp = pretty_print(e, indent=0)
        assert '*' in pp
        assert '+' in pp


# ============================================================
# High-Level synthesize() API Tests
# ============================================================

class TestSynthesizeAPI:
    def test_enumerative_api(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 2), ({'x': 3}, 6)]
        result = synthesize(examples, ['x'], method="enumerative", constants=[0, 1, 2])
        assert result.success

    def test_component_api(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1), ({'x': -3}, -3)]
        result = synthesize(examples, ['x'], method="component",
                           component_list=ARITHMETIC_COMPONENTS)
        assert result.success

    def test_conditional_api(self):
        examples = [({'x': -2}, 2), ({'x': 0}, 0), ({'x': 3}, 3)]
        result = synthesize(examples, ['x'], method="conditional",
                           components=['+', '-', '*', '<', '>', '=='])
        assert result.success

    def test_cegis_api(self):
        oracle = lambda inputs: inputs['x'] + 1
        examples = [({'x': 0}, 1)]
        result = synthesize(examples, ['x'], method="cegis", oracle=oracle)
        assert result.success

    def test_io_example_input(self):
        """Can pass IOExample directly."""
        examples = [IOExample({'x': 1}, 2), IOExample({'x': 2}, 3)]
        result = synthesize(examples, ['x'], method="enumerative")
        assert result.success

    def test_constraint_api(self):
        examples = [({'x': 0}, 1), ({'x': 3}, 4)]
        result = synthesize(examples, ['x'], method="constraint", max_nodes=4)
        assert result.success

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            synthesize([({'x': 0}, 0)], ['x'], method="quantum")


# ============================================================
# Edge Cases and Stress Tests
# ============================================================

class TestEdgeCases:
    def test_single_example(self):
        """Synthesis with single example."""
        examples = [IOExample({'x': 5}, 6)]
        spec = SynthesisSpec(examples, ['x'])
        result = EnumerativeSynthesizer(spec).synthesize()
        assert result.success

    def test_no_solution_small_space(self):
        """No solution in given search space."""
        # Asking for x^3 with max_depth=1 and only +,-,* won't work
        examples = [
            IOExample({'x': 0}, 0),
            IOExample({'x': 1}, 1),
            IOExample({'x': 2}, 8),
            IOExample({'x': 3}, 27),
        ]
        spec = SynthesisSpec(examples, ['x'])
        result = EnumerativeSynthesizer(spec, max_size=5, max_depth=1).synthesize()
        # May or may not find it depending on depth, but should not crash
        assert isinstance(result, SynthesisResult)

    def test_negative_constants(self):
        """Can use negative constants."""
        examples = [IOExample({'x': 0}, -1), IOExample({'x': 1}, 0)]
        spec = SynthesisSpec(examples, ['x'], constants=[-1, 0, 1])
        result = EnumerativeSynthesizer(spec).synthesize()
        assert result.success

    def test_frozen_exprs(self):
        """Expressions are frozen (hashable)."""
        e1 = BinOp('+', VarExpr('x'), IntConst(1))
        e2 = BinOp('+', VarExpr('x'), IntConst(1))
        assert e1 == e2
        assert hash(e1) == hash(e2)
        s = {e1, e2}
        assert len(s) == 1

    def test_empty_vars_constant(self):
        """Synthesize with no input variables (constant function)."""
        examples = [IOExample({}, 42)]
        spec = SynthesisSpec(examples, [], constants=[42])
        result = EnumerativeSynthesizer(spec).synthesize()
        assert result.success
        assert evaluate(result.program, {}) == 42

    def test_large_constants(self):
        """Works with larger constants."""
        examples = [IOExample({'x': 0}, 100), IOExample({'x': 1}, 101)]
        spec = SynthesisSpec(examples, ['x'], constants=[0, 1, 100])
        result = EnumerativeSynthesizer(spec).synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 5}) == 105

    def test_multiple_correct_programs(self):
        """Multiple programs may satisfy spec -- any correct one is fine."""
        # f(x) where f(0)=0, f(1)=1 -- could be x, x*1, x+0, etc.
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 1}, 1)]
        spec = SynthesisSpec(examples, ['x'])
        result = EnumerativeSynthesizer(spec).synthesize()
        assert result.success
        # Just verify it works on the examples
        for ex in examples:
            assert evaluate(result.program, ex.inputs) == ex.output


# ============================================================
# Integration Tests (compose multiple features)
# ============================================================

class TestIntegration:
    def test_synthesize_then_simplify(self):
        """Synthesize and then simplify the result."""
        examples = [({'x': 0}, 0), ({'x': 1}, 2), ({'x': 3}, 6)]
        result = synthesize(examples, ['x'], constants=[0, 1, 2])
        assert result.success
        simplified = simplify(result.program)
        # Should still be correct
        for inputs, output in examples:
            assert evaluate(simplified, inputs) == output

    def test_equivalence_of_synthesized(self):
        """Two synthesized programs for same spec should be equivalent."""
        examples = [IOExample({'x': 0}, 1), IOExample({'x': 1}, 2), IOExample({'x': -1}, 0)]
        spec = SynthesisSpec(examples, ['x'])
        r1 = EnumerativeSynthesizer(spec).synthesize()
        assert r1.success
        # Verify equivalence with x+1
        expected = BinOp('+', VarExpr('x'), IntConst(1))
        assert programs_equivalent(r1.program, expected, ['x'])

    def test_oracle_then_verify(self):
        """Oracle synthesis + equivalence check."""
        oracle = lambda inputs: inputs['x'] * inputs['x']
        synth = OracleSynthesizer(oracle, ['x'])
        result = synth.synthesize()
        assert result.success
        expected = BinOp('*', VarExpr('x'), VarExpr('x'))
        assert programs_equivalent(result.program, expected, ['x'])

    def test_conditional_then_simplify(self):
        """Conditional synthesis + simplification."""
        examples = [
            IOExample({'x': -3}, 3),
            IOExample({'x': 0}, 0),
            IOExample({'x': 5}, 5),
        ]
        spec = SynthesisSpec(examples, ['x'], components=['+', '-', '*', '<', '<=', '>', '>=', '=='])
        result = ConditionalSynthesizer(spec).synthesize()
        assert result.success
        simplified = simplify(result.program)
        for ex in examples:
            assert evaluate(simplified, ex.inputs) == ex.output

    def test_cegis_square(self):
        """CEGIS for x^2."""
        oracle = lambda inputs: inputs['x'] ** 2
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 1}, 1)]
        spec = SynthesisSpec(examples, ['x'])
        result = CEGISSynthesizer(spec, oracle=oracle).synthesize()
        assert result.success
        for v in range(-5, 6):
            assert evaluate(result.program, {'x': v}) == v * v

    def test_multiple_methods_same_result(self):
        """Different methods should all find a correct program."""
        examples = [
            IOExample({'x': 0}, 1),
            IOExample({'x': 1}, 2),
            IOExample({'x': -1}, 0),
        ]
        spec = SynthesisSpec(examples, ['x'])
        for method in ["enumerative", "component"]:
            if method == "component":
                result = synthesize(
                    [(ex.inputs, ex.output) for ex in examples],
                    ['x'], method=method,
                    component_list=ARITHMETIC_COMPONENTS
                )
            else:
                result = synthesize(
                    [(ex.inputs, ex.output) for ex in examples],
                    ['x'], method=method
                )
            assert result.success, f"Method {method} failed"
            for ex in examples:
                assert evaluate(result.program, ex.inputs) == ex.output


# ============================================================
# Specification Tests
# ============================================================

class TestSpec:
    def test_spec_defaults(self):
        spec = SynthesisSpec([], ['x'])
        assert spec.constants == [0, 1]
        assert spec.components == ['+', '-', '*']

    def test_spec_custom(self):
        spec = SynthesisSpec([], ['x'], constants=[0, 1, 2], components=['+', '*'])
        assert spec.constants == [0, 1, 2]
        assert spec.components == ['+', '*']

    def test_io_example(self):
        ex = IOExample({'x': 5, 'y': 3}, 8)
        assert ex.inputs == {'x': 5, 'y': 3}
        assert ex.output == 8

    def test_result_fields(self):
        r = SynthesisResult(True, IntConst(1), 3, 100, "test")
        assert r.success is True
        assert r.iterations == 3
        assert r.candidates_explored == 100
        assert r.method == "test"


# ============================================================
# Component Object Tests
# ============================================================

class TestComponents:
    def test_component_apply_binary(self):
        comp = ARITHMETIC_COMPONENTS[0]  # +
        result = comp.apply(VarExpr('x'), IntConst(1))
        assert isinstance(result, BinOp)
        assert result.op == '+'

    def test_component_apply_unary(self):
        comp = ARITHMETIC_COMPONENTS[3]  # neg
        result = comp.apply(VarExpr('x'))
        assert isinstance(result, UnaryOp)
        assert result.op == 'neg'

    def test_comparison_components(self):
        for comp in COMPARISON_COMPONENTS:
            assert comp.output_type == Type.BOOL
            assert comp.arity == 2

    def test_conditional_component(self):
        comp = CONDITIONAL_COMPONENTS[0]
        assert comp.name == 'ite'
        assert comp.arity == 3
        result = comp.apply(BoolConst(True), IntConst(1), IntConst(2))
        assert isinstance(result, IfExpr)

    def test_extended_components(self):
        for comp in EXTENDED_COMPONENTS:
            assert comp.output_type == Type.INT


# ============================================================
# Harder Synthesis Problems
# ============================================================

class TestHarderProblems:
    def test_polynomial_2x_plus_1(self):
        """Synthesize f(x) = 2*x + 1."""
        examples = [
            IOExample({'x': 0}, 1),
            IOExample({'x': 1}, 3),
            IOExample({'x': 2}, 5),
            IOExample({'x': -1}, -1),
        ]
        spec = SynthesisSpec(examples, ['x'], constants=[0, 1, 2])
        result = EnumerativeSynthesizer(spec, max_size=8).synthesize()
        assert result.success
        assert evaluate(result.program, {'x': 10}) == 21

    def test_difference_of_squares(self):
        """Synthesize f(x,y) = x*x - y*y (or (x+y)*(x-y))."""
        examples = [
            IOExample({'x': 2, 'y': 1}, 3),
            IOExample({'x': 3, 'y': 2}, 5),
            IOExample({'x': 0, 'y': 0}, 0),
            IOExample({'x': 1, 'y': 1}, 0),
            IOExample({'x': 4, 'y': 3}, 7),
        ]
        spec = SynthesisSpec(examples, ['x', 'y'])
        result = EnumerativeSynthesizer(spec, max_size=10, max_depth=3).synthesize()
        assert result.success
        for v1 in range(-3, 4):
            for v2 in range(-3, 4):
                assert evaluate(result.program, {'x': v1, 'y': v2}) == v1*v1 - v2*v2

    def test_sign_function(self):
        """Synthesize sign(x): -1, 0, or 1."""
        examples = [
            IOExample({'x': -5}, -1),
            IOExample({'x': -1}, -1),
            IOExample({'x': 0}, 0),
            IOExample({'x': 1}, 1),
            IOExample({'x': 3}, 1),
        ]
        spec = SynthesisSpec(examples, ['x'], constants=[-1, 0, 1],
                              components=['+', '-', '*', '<', '>', '=='])
        result = ConditionalSynthesizer(spec, max_branches=3).synthesize()
        assert result.success

    def test_clamp(self):
        """Synthesize clamp(x, 0, 10) = max(0, min(x, 10))."""
        examples = [
            IOExample({'x': -5}, 0),
            IOExample({'x': 0}, 0),
            IOExample({'x': 5}, 5),
            IOExample({'x': 10}, 10),
            IOExample({'x': 15}, 10),
        ]
        spec = SynthesisSpec(examples, ['x'], constants=[0, 10],
                              components=['+', '-', '*', '<', '>', '>=', '<='])
        result = ConditionalSynthesizer(spec, max_branches=3).synthesize()
        assert result.success

    def test_three_way_max(self):
        """Synthesize max(x, y) from examples."""
        examples = [
            IOExample({'x': 1, 'y': 2}, 2),
            IOExample({'x': 5, 'y': 3}, 5),
            IOExample({'x': -1, 'y': -3}, -1),
            IOExample({'x': 4, 'y': 4}, 4),
        ]
        spec = SynthesisSpec(examples, ['x', 'y'],
                              components=['+', '-', '*', '<', '<=', '>', '>='])
        result = ConditionalSynthesizer(spec).synthesize()
        assert result.success
        for v1 in range(-3, 4):
            for v2 in range(-3, 4):
                assert evaluate(result.program, {'x': v1, 'y': v2}) == max(v1, v2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
