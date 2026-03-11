"""Tests for V135: Effect-Typed Program Synthesis"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from effect_typed_synthesis import (
    effect_typed_synthesize, synthesize_pure, synthesize_total,
    synthesize_safe, synthesize_with_effects,
    verify_synthesized_effects, source_level_effect_check,
    compare_with_unrestricted, effect_synthesis_summary,
    EffectSpec, EffectSynthesisResult, EffectConstraint,
    _expr_effects, _expr_to_source, _check_expr_effect_constraint,
    PURE_ARITHMETIC, EFFECTFUL_ARITHMETIC,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V040_effect_systems'))
from effect_systems import EffectKind, EffectSet, PURE, IO, State, Exn, Effect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C097_program_synthesis'))
from synthesis import (
    IntConst, BoolConst, VarExpr, BinOp, UnaryOp, IfExpr,
    IOExample, evaluate, expr_size,
)


# ===== Section 1: Pure synthesis - identity/constant =====

class TestPureSynthesisBasic:
    def test_synthesize_identity(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1), ({'x': 5}, 5)]
        result = synthesize_pure(examples, ['x'])
        assert result.success
        assert result.effect_satisfied
        assert result.io_satisfied

    def test_synthesize_constant(self):
        examples = [({'x': 0}, 42), ({'x': 1}, 42), ({'x': 5}, 42)]
        result = synthesize_pure(examples, ['x'], constants=[0, 1, 42])
        assert result.success
        assert result.effect_satisfied

    def test_synthesize_increment(self):
        examples = [({'x': 0}, 1), ({'x': 1}, 2), ({'x': 5}, 6)]
        result = synthesize_pure(examples, ['x'])
        assert result.success
        assert result.io_satisfied

    def test_pure_result_is_pure(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_pure(examples, ['x'])
        assert result.inferred_effects is not None
        assert result.inferred_effects.is_pure


# ===== Section 2: Pure synthesis - arithmetic =====

class TestPureSynthesisArithmetic:
    def test_synthesize_double(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 2), ({'x': 3}, 6)]
        result = synthesize_pure(examples, ['x'])
        assert result.success
        assert result.effect_satisfied

    def test_synthesize_negate(self):
        examples = [({'x': 0}, 0), ({'x': 1}, -1), ({'x': 5}, -5)]
        result = synthesize_pure(examples, ['x'])
        assert result.success
        assert result.effect_satisfied

    def test_synthesize_sum(self):
        examples = [({'x': 1, 'y': 2}, 3), ({'x': 0, 'y': 5}, 5), ({'x': 3, 'y': 4}, 7)]
        result = synthesize_pure(examples, ['x', 'y'])
        assert result.success
        assert result.effect_satisfied

    def test_synthesize_difference(self):
        examples = [({'x': 5, 'y': 2}, 3), ({'x': 10, 'y': 3}, 7), ({'x': 0, 'y': 0}, 0)]
        result = synthesize_pure(examples, ['x', 'y'])
        assert result.success


# ===== Section 3: Effect spec construction =====

class TestEffectSpec:
    def test_pure_spec(self):
        spec = EffectSpec.pure()
        assert EffectKind.STATE in spec.forbidden_kinds
        assert EffectKind.IO in spec.forbidden_kinds
        assert EffectKind.EXN in spec.forbidden_kinds

    def test_no_io_spec(self):
        spec = EffectSpec.no_io()
        assert EffectKind.IO in spec.forbidden_kinds
        assert EffectKind.STATE not in spec.forbidden_kinds

    def test_no_exn_spec(self):
        spec = EffectSpec.no_exn()
        assert EffectKind.EXN in spec.forbidden_kinds
        assert EffectKind.STATE not in spec.forbidden_kinds

    def test_unrestricted_spec(self):
        spec = EffectSpec.unrestricted()
        assert len(spec.forbidden_kinds) == 0

    def test_total_spec(self):
        spec = EffectSpec.total()
        assert EffectKind.DIV in spec.forbidden_kinds

    def test_spec_check_pure_effects(self):
        spec = EffectSpec.pure()
        assert spec.check_effects(EffectSet.pure())
        assert not spec.check_effects(EffectSet.of(IO))


# ===== Section 4: Expression effect inference =====

class TestExprEffects:
    def test_constant_is_pure(self):
        effects = _expr_effects(IntConst(42))
        assert effects.is_pure

    def test_variable_is_pure(self):
        effects = _expr_effects(VarExpr('x'))
        assert effects.is_pure

    def test_addition_is_pure(self):
        expr = BinOp('+', VarExpr('x'), IntConst(1))
        effects = _expr_effects(expr)
        assert effects.is_pure

    def test_division_has_exn(self):
        expr = BinOp('/', VarExpr('x'), VarExpr('y'))
        effects = _expr_effects(expr)
        assert not effects.is_pure
        assert any(e.kind == EffectKind.EXN for e in effects.effects)

    def test_modulo_has_exn(self):
        expr = BinOp('%', VarExpr('x'), IntConst(2))
        effects = _expr_effects(expr)
        assert any(e.kind == EffectKind.EXN for e in effects.effects)

    def test_if_combines_effects(self):
        expr = IfExpr(
            BoolConst(True),
            BinOp('+', VarExpr('x'), IntConst(1)),
            BinOp('/', VarExpr('x'), VarExpr('y'))
        )
        effects = _expr_effects(expr)
        assert any(e.kind == EffectKind.EXN for e in effects.effects)

    def test_nested_pure_is_pure(self):
        expr = BinOp('+', BinOp('*', VarExpr('x'), IntConst(2)), IntConst(1))
        effects = _expr_effects(expr)
        assert effects.is_pure

    def test_unary_preserves_purity(self):
        expr = UnaryOp('neg', VarExpr('x'))
        effects = _expr_effects(expr)
        assert effects.is_pure


# ===== Section 5: Effect constraint checking =====

class TestEffectConstraintChecking:
    def test_pure_expr_passes_pure_spec(self):
        expr = BinOp('+', VarExpr('x'), IntConst(1))
        assert _check_expr_effect_constraint(expr, EffectSpec.pure())

    def test_division_fails_pure_spec(self):
        expr = BinOp('/', VarExpr('x'), VarExpr('y'))
        assert not _check_expr_effect_constraint(expr, EffectSpec.pure())

    def test_division_passes_unrestricted(self):
        expr = BinOp('/', VarExpr('x'), VarExpr('y'))
        assert _check_expr_effect_constraint(expr, EffectSpec.unrestricted())

    def test_division_fails_no_exn(self):
        expr = BinOp('/', VarExpr('x'), VarExpr('y'))
        assert not _check_expr_effect_constraint(expr, EffectSpec.no_exn())


# ===== Section 6: Component filtering =====

class TestComponentFiltering:
    def test_pure_excludes_division(self):
        from effect_typed_synthesis import _filter_components
        filtered = _filter_components(['+',' -', '*', '/'], EffectSpec.pure())
        assert '/' not in filtered

    def test_pure_excludes_modulo(self):
        from effect_typed_synthesis import _filter_components
        filtered = _filter_components(['+', '-', '%'], EffectSpec.pure())
        assert '%' not in filtered

    def test_unrestricted_keeps_all(self):
        from effect_typed_synthesis import _filter_components
        components = ['+', '-', '*', '/', '%']
        filtered = _filter_components(components, EffectSpec.unrestricted())
        assert filtered == components

    def test_default_pure_components(self):
        from effect_typed_synthesis import _get_default_components
        components = _get_default_components(EffectSpec.pure())
        assert '/' not in components
        assert '+' in components


# ===== Section 7: Safe synthesis (no exceptions) =====

class TestSafeSynthesis:
    def test_safe_identity(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_safe(examples, ['x'])
        assert result.success
        assert result.effect_satisfied

    def test_safe_arithmetic(self):
        examples = [({'x': 1, 'y': 2}, 3), ({'x': 3, 'y': 4}, 7)]
        result = synthesize_safe(examples, ['x', 'y'])
        assert result.success
        assert result.effect_satisfied


# ===== Section 8: Effect-typed synthesis with explicit effects =====

class TestSynthesizeWithEffects:
    def test_allow_all(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_with_effects(
            examples, ['x'],
            allowed_effects=EffectSet.of(IO, State("x"), Exn(), Effect(EffectKind.DIV))
        )
        assert result.success

    def test_pure_constraint(self):
        examples = [({'x': 0}, 1), ({'x': 1}, 2)]
        result = synthesize_with_effects(
            examples, ['x'],
            allowed_effects=EffectSet.pure()
        )
        assert result.success
        assert result.effect_satisfied


# ===== Section 9: Expression to source conversion =====

class TestExprToSource:
    def test_int_const(self):
        assert _expr_to_source(IntConst(42)) == "42"

    def test_var(self):
        assert _expr_to_source(VarExpr('x')) == "x"

    def test_binop(self):
        expr = BinOp('+', VarExpr('x'), IntConst(1))
        assert _expr_to_source(expr) == "(x + 1)"

    def test_neg(self):
        expr = UnaryOp('neg', VarExpr('x'))
        assert _expr_to_source(expr) == "(0 - x)"

    def test_if_expr(self):
        expr = IfExpr(BoolConst(True), IntConst(1), IntConst(0))
        s = _expr_to_source(expr)
        assert "if" in s
        assert "1" in s
        assert "0" in s

    def test_max_min(self):
        expr = BinOp('max', VarExpr('x'), VarExpr('y'))
        assert _expr_to_source(expr) == "max(x, y)"


# ===== Section 10: Verify synthesized effects =====

class TestVerifySynthesizedEffects:
    def test_pure_program(self):
        expr = BinOp('+', VarExpr('x'), IntConst(1))
        result = verify_synthesized_effects(expr, EffectSpec.pure())
        assert result["satisfied"]
        assert result["is_pure"]
        assert result["effect_count"] == 0

    def test_effectful_program(self):
        expr = BinOp('/', VarExpr('x'), VarExpr('y'))
        result = verify_synthesized_effects(expr, EffectSpec.pure())
        assert not result["satisfied"]
        assert len(result["violations"]) > 0

    def test_effectful_passes_unrestricted(self):
        expr = BinOp('/', VarExpr('x'), VarExpr('y'))
        result = verify_synthesized_effects(expr, EffectSpec.unrestricted())
        assert result["satisfied"]


# ===== Section 11: Effect synthesis result structure =====

class TestEffectSynthesisResult:
    def test_result_has_program(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_pure(examples, ['x'])
        assert result.program is not None
        assert result.program_str is not None

    def test_result_has_effects(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_pure(examples, ['x'])
        assert result.inferred_effects is not None

    def test_result_method(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_pure(examples, ['x'])
        assert result.method != ""

    def test_failed_result(self):
        # Impossible to synthesize with tiny size
        examples = [({'x': 0}, 0), ({'x': 1}, 1), ({'x': 2}, 8)]
        result = synthesize_pure(examples, ['x'], max_size=1)
        assert not result.success


# ===== Section 12: Summary API =====

class TestSummaryAPI:
    def test_summary_returns_dict(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_pure(examples, ['x'])
        s = effect_synthesis_summary(result)
        assert isinstance(s, dict)
        assert s["success"]
        assert s["effect_satisfied"]

    def test_summary_has_fields(self):
        examples = [({'x': 0}, 1), ({'x': 1}, 2)]
        result = synthesize_pure(examples, ['x'])
        s = effect_synthesis_summary(result)
        assert "program" in s
        assert "effects" in s
        assert "is_pure" in s
        assert "method" in s

    def test_summary_pure_effects(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_pure(examples, ['x'])
        s = effect_synthesis_summary(result)
        assert s["is_pure"]
        assert s["effects"] == []


# ===== Section 13: Comparison API =====

class TestComparisonAPI:
    def test_compare_returns_dict(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = compare_with_unrestricted(examples, ['x'], EffectSpec.pure())
        assert isinstance(result, dict)
        assert "unrestricted" in result
        assert "constrained" in result

    def test_compare_both_succeed(self):
        examples = [({'x': 0}, 1), ({'x': 1}, 2)]
        result = compare_with_unrestricted(examples, ['x'], EffectSpec.pure())
        assert result["both_succeeded"]

    def test_compare_has_constraint_spec(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = compare_with_unrestricted(examples, ['x'], EffectSpec.pure())
        assert "constraint_spec" in result
        assert len(result["constraint_spec"]) > 0


# ===== Section 14: Two-variable pure synthesis =====

class TestTwoVariablePure:
    def test_sum(self):
        examples = [({'x': 1, 'y': 2}, 3), ({'x': 0, 'y': 5}, 5)]
        result = synthesize_pure(examples, ['x', 'y'])
        assert result.success
        assert result.effect_satisfied

    def test_difference(self):
        examples = [({'x': 5, 'y': 3}, 2), ({'x': 10, 'y': 7}, 3)]
        result = synthesize_pure(examples, ['x', 'y'])
        assert result.success

    def test_product(self):
        examples = [({'x': 2, 'y': 3}, 6), ({'x': 0, 'y': 5}, 0), ({'x': 1, 'y': 1}, 1)]
        result = synthesize_pure(examples, ['x', 'y'])
        assert result.success


# ===== Section 15: Effect spec is_allowed =====

class TestEffectSpecIsAllowed:
    def test_pure_disallows_state(self):
        spec = EffectSpec.pure()
        assert not spec.is_allowed(EffectKind.STATE)

    def test_pure_disallows_io(self):
        spec = EffectSpec.pure()
        assert not spec.is_allowed(EffectKind.IO)

    def test_no_io_allows_state(self):
        spec = EffectSpec.no_io()
        assert spec.is_allowed(EffectKind.STATE)

    def test_unrestricted_allows_all(self):
        spec = EffectSpec.unrestricted()
        assert spec.is_allowed(EffectKind.STATE)
        assert spec.is_allowed(EffectKind.IO)
        assert spec.is_allowed(EffectKind.EXN)


# ===== Section 16: IOExample normalization =====

class TestIOExampleNormalization:
    def test_tuple_examples(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_pure(examples, ['x'])
        assert result.success

    def test_ioexample_objects(self):
        examples = [IOExample({'x': 0}, 0), IOExample({'x': 1}, 1)]
        result = synthesize_pure(examples, ['x'])
        assert result.success


# ===== Section 17: Multiple methods =====

class TestMultipleMethods:
    def test_enumerative(self):
        examples = [({'x': 0}, 1), ({'x': 1}, 2)]
        result = effect_typed_synthesize(examples, ['x'], method="enumerative")
        assert result.success

    def test_constraint(self):
        examples = [({'x': 0}, 1), ({'x': 1}, 2), ({'x': 3}, 4)]
        result = effect_typed_synthesize(examples, ['x'], method="constraint")
        assert result.success

    def test_conditional(self):
        examples = [({'x': -1}, 1), ({'x': 0}, 0), ({'x': 1}, 1)]
        result = effect_typed_synthesize(
            examples, ['x'], method="conditional",
            max_size=15, max_depth=5
        )
        # Conditional synthesis may or may not find abs
        # Just check it returns a result without crashing
        assert isinstance(result, EffectSynthesisResult)


# ===== Section 18: Edge cases =====

class TestEdgeCases:
    def test_single_example(self):
        examples = [({'x': 5}, 5)]
        result = synthesize_pure(examples, ['x'])
        assert result.success

    def test_negative_values(self):
        examples = [({'x': -1}, -2), ({'x': -3}, -4)]
        result = synthesize_pure(examples, ['x'])
        assert result.success

    def test_zero(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 0), ({'x': 2}, 0)]
        result = synthesize_pure(examples, ['x'], constants=[0])
        assert result.success


# ===== Section 19: Total synthesis (no divergence) =====

class TestTotalSynthesis:
    def test_total_identity(self):
        examples = [({'x': 0}, 0), ({'x': 1}, 1)]
        result = synthesize_total(examples, ['x'])
        assert result.success
        assert result.effect_satisfied

    def test_total_arithmetic(self):
        examples = [({'x': 1, 'y': 2}, 3)]
        result = synthesize_total(examples, ['x', 'y'])
        assert result.success


# ===== Section 20: Effect-typed with custom constants =====

class TestCustomConstants:
    def test_with_custom_constants(self):
        examples = [({'x': 0}, 10), ({'x': 1}, 11)]
        result = synthesize_pure(examples, ['x'], constants=[0, 1, 10])
        assert result.success
        assert result.io_satisfied

    def test_with_negative_constants(self):
        examples = [({'x': 0}, -1), ({'x': 1}, 0)]
        result = synthesize_pure(examples, ['x'], constants=[0, 1, -1])
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
