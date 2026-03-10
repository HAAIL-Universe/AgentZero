"""Tests for V011: Refinement Type Checking (Liquid Types)"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from refinement_types import (
    RefinedType, RefinedFuncType, RefinementEnv, RefinementChecker,
    SubtypeResult, CheckResult, RefinementError,
    check_subtype, check_refinements, check_program_refinements,
    check_function_refinements, check_subtype_valid, infer_refinement,
    subst_sexpr, selfify, negate_sexpr, strip_annotations,
    extract_refinement_specs, parse_source,
    refined_int, unrefined, nat_type, pos_type, range_type, eq_type,
    _type_to_expr
)

# Import SExpr types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
from vc_gen import SVar, SInt, SBool, SBinOp, SUnaryOp, SAnd, SOr, SNot, SImplies, s_and, s_or

# Import base types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C013_type_checker'))
from type_checker import INT, BOOL, FLOAT, STRING


# ===================================================================
# Section 1: Refinement Type Construction
# ===================================================================

class TestRefinementTypeConstruction:
    def test_basic_refined_type(self):
        rt = RefinedType(INT, 'v', SBinOp('>=', SVar('v'), SInt(0)))
        assert rt.base is INT
        assert rt.binder == 'v'

    def test_unrefined(self):
        rt = unrefined(INT)
        assert rt.base is INT
        assert isinstance(rt.predicate, SBool) and rt.predicate.value == True

    def test_nat_type(self):
        rt = nat_type()
        assert rt.base is INT
        assert isinstance(rt.predicate, SBinOp)
        assert rt.predicate.op == '>='

    def test_pos_type(self):
        rt = pos_type()
        assert rt.predicate.op == '>'

    def test_range_type(self):
        rt = range_type(1, 10)
        assert isinstance(rt.predicate, SAnd)

    def test_eq_type(self):
        rt = eq_type(42)
        assert rt.predicate.op == '=='

    def test_refined_func_type(self):
        ft = RefinedFuncType(
            params=[('x', nat_type()), ('y', pos_type())],
            ret=nat_type()
        )
        assert len(ft.params) == 2
        assert ft.ret.base is INT


# ===================================================================
# Section 2: SExpr Substitution
# ===================================================================

class TestSubstitution:
    def test_subst_var(self):
        expr = SVar('v')
        result = subst_sexpr(expr, 'v', SVar('x'))
        assert isinstance(result, SVar) and result.name == 'x'

    def test_subst_no_match(self):
        expr = SVar('w')
        result = subst_sexpr(expr, 'v', SVar('x'))
        assert isinstance(result, SVar) and result.name == 'w'

    def test_subst_binop(self):
        expr = SBinOp('>=', SVar('v'), SInt(0))
        result = subst_sexpr(expr, 'v', SVar('x'))
        assert isinstance(result, SBinOp)
        assert isinstance(result.left, SVar) and result.left.name == 'x'

    def test_subst_nested(self):
        expr = SBinOp('+', SBinOp('*', SVar('v'), SInt(2)), SVar('v'))
        result = subst_sexpr(expr, 'v', SVar('y'))
        assert isinstance(result.left.left, SVar) and result.left.left.name == 'y'
        assert isinstance(result.right, SVar) and result.right.name == 'y'

    def test_selfify(self):
        rt = nat_type()  # {v: int | v >= 0}
        pred = selfify(rt, 'x')  # x >= 0
        assert isinstance(pred, SBinOp)
        assert pred.left.name == 'x'

    def test_subst_constant(self):
        expr = SInt(5)
        result = subst_sexpr(expr, 'v', SVar('x'))
        assert isinstance(result, SInt) and result.value == 5


# ===================================================================
# Section 3: Negation
# ===================================================================

class TestNegation:
    def test_negate_bool(self):
        assert negate_sexpr(SBool(True)) == SBool(False)
        assert negate_sexpr(SBool(False)) == SBool(True)

    def test_negate_comparison(self):
        expr = SBinOp('>=', SVar('x'), SInt(0))
        neg = negate_sexpr(expr)
        assert isinstance(neg, SBinOp)
        assert neg.op == '<'

    def test_negate_eq(self):
        expr = SBinOp('==', SVar('x'), SInt(5))
        neg = negate_sexpr(expr)
        assert neg.op == '!='

    def test_negate_and(self):
        expr = SBinOp('and', SVar('a'), SVar('b'))
        neg = negate_sexpr(expr)
        # Should be OR of negations (De Morgan)
        assert isinstance(neg, SOr)

    def test_double_negation(self):
        expr = SBinOp('>=', SVar('x'), SInt(0))
        neg = negate_sexpr(expr)
        double_neg = negate_sexpr(neg)
        assert double_neg.op == '>='


# ===================================================================
# Section 4: Subtype Checking
# ===================================================================

class TestSubtypeChecking:
    def test_nat_subtype_of_int(self):
        """nat <: int (any nat is an int, trivially)"""
        result = check_subtype(nat_type(), unrefined(INT))
        assert result.is_subtype

    def test_pos_subtype_of_nat(self):
        """pos <: nat (any positive int is non-negative)"""
        result = check_subtype(pos_type(), nat_type())
        assert result.is_subtype

    def test_nat_not_subtype_of_pos(self):
        """nat is NOT a subtype of pos (0 is nat but not pos)"""
        result = check_subtype(nat_type(), pos_type())
        assert not result.is_subtype

    def test_eq_subtype_of_nat(self):
        """{v == 5} <: {v >= 0}"""
        result = check_subtype(eq_type(5), nat_type())
        assert result.is_subtype

    def test_eq_negative_not_subtype_of_nat(self):
        """{v == -3} is NOT <: {v >= 0}"""
        result = check_subtype(eq_type(-3), nat_type())
        assert not result.is_subtype

    def test_range_subtype_of_nat(self):
        """{1 <= v <= 10} <: {v >= 0}"""
        result = check_subtype(range_type(1, 10), nat_type())
        assert result.is_subtype

    def test_range_not_subtype_of_pos_when_includes_zero(self):
        """{0 <= v <= 10} is NOT <: {v > 0}"""
        result = check_subtype(range_type(0, 10), pos_type())
        assert not result.is_subtype

    def test_reflexive(self):
        """nat <: nat"""
        result = check_subtype(nat_type(), nat_type())
        assert result.is_subtype

    def test_subtype_with_assumptions(self):
        """Under x > 0, {v == x} <: {v > 0}"""
        sub = RefinedType(INT, 'v', SBinOp('==', SVar('v'), SVar('x')))
        sup = pos_type()
        assumptions = [SBinOp('>', SVar('x'), SInt(0))]
        result = check_subtype(sub, sup, assumptions)
        assert result.is_subtype

    def test_counterexample_provided(self):
        """Failed subtype check provides counterexample"""
        result = check_subtype(nat_type(), pos_type())
        assert not result.is_subtype
        assert result.counterexample is not None


# ===================================================================
# Section 5: Refinement Environment
# ===================================================================

class TestRefinementEnv:
    def test_bind_and_lookup(self):
        env = RefinementEnv()
        env.bind('x', nat_type('x'))
        rt = env.lookup('x')
        assert rt is not None
        assert rt.base is INT

    def test_parent_lookup(self):
        parent = RefinementEnv()
        parent.bind('x', nat_type('x'))
        child = parent.child()
        assert child.lookup('x') is not None

    def test_child_overrides_parent(self):
        parent = RefinementEnv()
        parent.bind('x', nat_type('x'))
        child = parent.child()
        child.bind('x', pos_type('x'))
        rt = child.lookup('x')
        assert rt.predicate.op == '>'

    def test_assumptions(self):
        env = RefinementEnv()
        env.assume(SBinOp('>', SVar('x'), SInt(0)))
        child = env.child()
        child.assume(SBinOp('<', SVar('x'), SInt(10)))
        all_asms = child.all_assumptions()
        assert len(all_asms) == 2


# ===================================================================
# Section 6: Refinement Inference for Expressions
# ===================================================================

class TestRefinementInference:
    def test_infer_int_literal(self):
        rt = infer_refinement("let x = 42;", 'x')
        assert rt is not None
        # Should be {x: int | x == 42}
        assert rt.base is INT

    def test_infer_addition(self):
        rt = infer_refinement("let a = 3; let b = 4; let c = (a + b);", 'c')
        assert rt is not None
        assert rt.base is INT

    def test_infer_negation(self):
        rt = infer_refinement("let x = 5; let y = (0 - x);", 'y')
        assert rt is not None

    def test_infer_reassignment(self):
        rt = infer_refinement("let x = 5; x = 10;", 'x')
        assert rt is not None
        # After reassignment, should reflect new value


# ===================================================================
# Section 7: Function Checking with Specs
# ===================================================================

class TestFunctionChecking:
    def test_identity_nat(self):
        """fn id(x) { return x; } with nat -> nat spec should pass"""
        source = "fn id(x) { return x; }"
        result = check_function_refinements(
            source, 'id',
            params=[('x', nat_type())],
            ret=nat_type()
        )
        assert result.ok

    def test_increment_nat_to_pos(self):
        """fn inc(x) { return x + 1; } with nat -> pos should pass"""
        source = "fn inc(x) { let r = (x + 1); return r; }"
        result = check_function_refinements(
            source, 'inc',
            params=[('x', nat_type())],
            ret=pos_type()
        )
        assert result.ok

    def test_increment_int_to_nat_fails(self):
        """fn inc(x) { return x + 1; } with int -> nat should fail (x could be -2)"""
        source = "fn inc(x) { let r = (x + 1); return r; }"
        result = check_function_refinements(
            source, 'inc',
            params=[('x', unrefined(INT))],
            ret=nat_type()
        )
        assert not result.ok

    def test_abs_returns_nat(self):
        """fn abs(x) { if (x >= 0) { return x; } else { return 0-x; } }"""
        source = """
        fn abs(x) {
            if (x >= 0) {
                return x;
            } else {
                let neg = (0 - x);
                return neg;
            }
        }
        """
        result = check_function_refinements(
            source, 'abs',
            params=[('x', unrefined(INT))],
            ret=nat_type()
        )
        assert result.ok

    def test_max_returns_geq_both(self):
        """fn max(x, y) returns {v >= x && v >= y}"""
        source = """
        fn max(x, y) {
            if (x >= y) {
                return x;
            } else {
                return y;
            }
        }
        """
        ret_type = RefinedType(INT, 'v', s_and(
            SBinOp('>=', SVar('v'), SVar('x')),
            SBinOp('>=', SVar('v'), SVar('y'))
        ))
        result = check_function_refinements(
            source, 'max',
            params=[('x', unrefined(INT)), ('y', unrefined(INT))],
            ret=ret_type
        )
        assert result.ok


# ===================================================================
# Section 8: Path-Sensitive Refinement
# ===================================================================

class TestPathSensitive:
    def test_if_then_refines(self):
        """In 'if (x > 0)', the then-branch knows x > 0"""
        source = """
        fn check(x) {
            if (x > 0) {
                return x;
            } else {
                return (0 - x);
            }
        }
        """
        # Return type: nat (both branches should produce >= 0)
        result = check_function_refinements(
            source, 'check',
            params=[('x', unrefined(INT))],
            ret=nat_type()
        )
        # Then branch: x > 0 => x >= 0 (ok)
        # Else branch: x <= 0 => 0-x >= 0 (ok)
        assert result.ok

    def test_nested_if_refines(self):
        """Nested conditionals accumulate path conditions"""
        source = """
        fn classify(x) {
            if (x > 0) {
                if (x > 10) {
                    return x;
                } else {
                    return x;
                }
            } else {
                return (0 - x);
            }
        }
        """
        result = check_function_refinements(
            source, 'classify',
            params=[('x', unrefined(INT))],
            ret=nat_type()
        )
        assert result.ok


# ===================================================================
# Section 9: Source-Level Annotation API
# ===================================================================

class TestAnnotationAPI:
    def test_extract_specs(self):
        source = """
        fn abs(x) {
            ensures(result >= 0);
            if (x >= 0) {
                return x;
            } else {
                let neg = (0 - x);
                return neg;
            }
        }
        """
        stmts = parse_source(source)
        specs = extract_refinement_specs(stmts)
        assert 'abs' in specs
        assert specs['abs'].ret.predicate is not None

    def test_check_program_with_annotations(self):
        source = """
        fn abs(x) {
            ensures(result >= 0);
            if (x >= 0) {
                return x;
            } else {
                let neg = (0 - x);
                return neg;
            }
        }
        """
        result = check_program_refinements(source)
        assert result.ok

    def test_requires_annotation(self):
        source = """
        fn safe_div_approx(x, y) {
            requires(y > 0);
            ensures(result >= 0);
            if (x >= 0) {
                return x;
            } else {
                return (0 - x);
            }
        }
        """
        result = check_program_refinements(source)
        assert result.ok

    def test_strip_annotations(self):
        source = """
        fn f(x) {
            requires(x > 0);
            ensures(result > 0);
            return x;
        }
        """
        stmts = parse_source(source)
        clean = strip_annotations(stmts)
        # Function body should not contain requires/ensures
        fn = clean[0]
        for s in fn.body.stmts:
            if isinstance(s, type(stmts[0])):
                continue
            assert not (hasattr(s, 'callee') and s.callee in ('requires', 'ensures'))


# ===================================================================
# Section 10: Call-Site Checking
# ===================================================================

class TestCallSiteChecking:
    def test_valid_call(self):
        """Calling abs(x) where x satisfies nat -> ok"""
        source = """
        fn abs(x) {
            if (x >= 0) { return x; } else { let n = (0 - x); return n; }
        }
        fn main() {
            let a = 5;
            let b = abs(a);
            return b;
        }
        """
        abs_spec = RefinedFuncType(
            params=[('x', unrefined(INT))],
            ret=nat_type()
        )
        main_spec = RefinedFuncType(
            params=[],
            ret=nat_type()
        )
        result = check_refinements(source, {'abs': abs_spec, 'main': main_spec})
        assert result.ok

    def test_invalid_call_detected(self):
        """Calling a function with wrong argument type should fail"""
        source = """
        fn need_pos(x) {
            return x;
        }
        fn main() {
            let a = 0;
            let b = need_pos(a);
            return b;
        }
        """
        need_pos_spec = RefinedFuncType(
            params=[('x', pos_type())],
            ret=pos_type()
        )
        main_spec = RefinedFuncType(
            params=[],
            ret=unrefined(INT)
        )
        result = check_refinements(source, {'need_pos': need_pos_spec, 'main': main_spec})
        assert not result.ok
        assert any('argument' in e.message for e in result.errors)


# ===================================================================
# Section 11: Subtype Lattice Properties
# ===================================================================

class TestSubtypeLattice:
    def test_transitivity(self):
        """pos <: nat <: int => pos <: int"""
        r1 = check_subtype(pos_type(), nat_type())
        r2 = check_subtype(nat_type(), unrefined(INT))
        r3 = check_subtype(pos_type(), unrefined(INT))
        assert r1.is_subtype and r2.is_subtype and r3.is_subtype

    def test_range_containment(self):
        """{1..5} <: {0..10}"""
        result = check_subtype(range_type(1, 5), range_type(0, 10))
        assert result.is_subtype

    def test_range_not_contained(self):
        """{0..10} is NOT <: {1..5}"""
        result = check_subtype(range_type(0, 10), range_type(1, 5))
        assert not result.is_subtype

    def test_singleton_in_range(self):
        """{v == 3} <: {1..10}"""
        result = check_subtype(eq_type(3), range_type(1, 10))
        assert result.is_subtype

    def test_singleton_outside_range(self):
        """{v == 15} is NOT <: {1..10}"""
        result = check_subtype(eq_type(15), range_type(1, 10))
        assert not result.is_subtype


# ===================================================================
# Section 12: Dependent Return Types
# ===================================================================

class TestDependentReturn:
    def test_return_depends_on_input(self):
        """fn inc(x) returns {v == x + 1} -- return depends on param"""
        source = "fn inc(x) { let r = (x + 1); return r; }"
        ret_type = RefinedType(INT, 'v', SBinOp('==', SVar('v'), SBinOp('+', SVar('x'), SInt(1))))
        result = check_function_refinements(
            source, 'inc',
            params=[('x', unrefined(INT))],
            ret=ret_type
        )
        assert result.ok

    def test_return_geq_input(self):
        """fn clamp_pos(x) returns {v >= x}"""
        source = """
        fn clamp_pos(x) {
            if (x >= 0) {
                return x;
            } else {
                return 0;
            }
        }
        """
        ret_type = RefinedType(INT, 'v', SBinOp('>=', SVar('v'), SVar('x')))
        result = check_function_refinements(
            source, 'clamp_pos',
            params=[('x', unrefined(INT))],
            ret=ret_type
        )
        assert result.ok


# ===================================================================
# Section 13: While Loops
# ===================================================================

class TestWhileLoops:
    def test_loop_weakens_refinement(self):
        """After a loop, modified variables lose precise refinements"""
        source = """
        fn count(n) {
            let i = 0;
            while (i < n) {
                i = (i + 1);
            }
            return i;
        }
        """
        # We can't prove i == n without invariants, but we can check
        # that the checker doesn't crash
        result = check_function_refinements(
            source, 'count',
            params=[('n', nat_type())],
            ret=unrefined(INT)  # Weak return type
        )
        assert result.ok

    def test_loop_body_has_path_condition(self):
        """Inside loop, loop condition is assumed"""
        source = """
        fn count(n) {
            let i = 0;
            while (i < n) {
                i = (i + 1);
            }
            return i;
        }
        """
        result = check_function_refinements(
            source, 'count',
            params=[('n', unrefined(INT))],
            ret=unrefined(INT)
        )
        assert result.ok


# ===================================================================
# Section 14: Edge Cases and Error Handling
# ===================================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = check_refinements("let x = 1;", {})
        assert result.ok

    def test_no_specs_no_errors(self):
        source = "fn f(x) { return x; }"
        result = check_refinements(source, {})
        assert result.ok

    def test_unrefined_is_always_subtype_of_unrefined(self):
        result = check_subtype(unrefined(INT), unrefined(INT))
        assert result.is_subtype

    def test_multiple_errors_reported(self):
        """Multiple subtype violations should all be reported"""
        source = """
        fn f(x) {
            return x;
        }
        fn g(y) {
            return y;
        }
        """
        # Both functions claim to return pos, but params are unrefined
        specs = {
            'f': RefinedFuncType(params=[('x', unrefined(INT))], ret=pos_type()),
            'g': RefinedFuncType(params=[('y', unrefined(INT))], ret=pos_type()),
        }
        result = check_refinements(source, specs)
        assert len(result.errors) == 2

    def test_obligation_counting(self):
        """Verified and total obligation counts are correct"""
        source = "fn id(x) { return x; }"
        result = check_function_refinements(
            source, 'id',
            params=[('x', nat_type())],
            ret=nat_type()
        )
        assert result.total_obligations >= 1
        assert result.verified_obligations == result.total_obligations

    def test_result_repr(self):
        result = CheckResult(errors=[], verified_obligations=3, total_obligations=3)
        s = repr(result)
        assert 'OK' in s
        assert '3/3' in s


# ===================================================================
# Section 15: Composition with Base Type Checker
# ===================================================================

class TestCompositionWithBaseTypes:
    def test_int_refinement_on_int_base(self):
        """Refinement must be consistent with base type"""
        # nat is a refinement of int
        result = check_subtype(nat_type(), unrefined(INT))
        assert result.is_subtype

    def test_bool_refinement(self):
        """Boolean refinement: {v: bool | v == 1} (true)"""
        true_type = RefinedType(BOOL, 'v', SBinOp('==', SVar('v'), SInt(1)))
        bool_type = unrefined(BOOL)
        result = check_subtype(true_type, bool_type)
        assert result.is_subtype


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
