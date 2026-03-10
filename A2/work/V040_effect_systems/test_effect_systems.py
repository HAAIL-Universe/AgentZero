"""Tests for V040: Effect Systems"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from effect_systems import (
    Effect, EffectKind, EffectSet, PURE, IO, DIV, NONDET, State, Exn,
    BaseType, T_INT, T_BOOL, T_UNIT, EffectfulFuncType,
    EffectInferrer, EffectChecker, EffectVerifier,
    EffectCheckStatus, EffectCheckResult, EffectVerificationResult,
    FnEffectSig, EffectVar, PolyEffectfulType,
    infer_effects, check_effects, verify_effects,
    compose_effects, handle_effect, handle_all_exn, effect_subtype,
)


# ============================================================
# Effect Representation Tests
# ============================================================

class TestEffect:
    def test_pure_effect(self):
        assert PURE.kind == EffectKind.PURE
        assert PURE.detail is None

    def test_state_effect(self):
        e = State("x")
        assert e.kind == EffectKind.STATE
        assert e.detail == "x"

    def test_exn_effect(self):
        e = Exn("DivByZero")
        assert e.kind == EffectKind.EXN
        assert e.detail == "DivByZero"

    def test_effect_equality(self):
        assert State("x") == State("x")
        assert State("x") != State("y")
        assert IO == IO
        assert IO != DIV

    def test_effect_repr(self):
        assert "state(x)" == repr(State("x"))
        assert "io" == repr(IO)
        assert "exn(Error)" == repr(Exn("Error"))


class TestEffectSet:
    def test_pure_set(self):
        es = EffectSet.pure()
        assert es.is_pure
        assert len(es.effects) == 0

    def test_of_filters_pure(self):
        es = EffectSet.of(PURE, IO)
        assert IO in es.effects
        assert PURE not in es.effects

    def test_union(self):
        a = EffectSet.of(IO)
        b = EffectSet.of(State("x"))
        c = a.union(b)
        assert IO in c.effects
        assert State("x") in c.effects

    def test_minus(self):
        es = EffectSet.of(IO, State("x"))
        masked = es.minus(IO)
        assert not masked.has(EffectKind.IO)
        assert masked.has(EffectKind.STATE)

    def test_minus_kind(self):
        es = EffectSet.of(Exn("A"), Exn("B"), IO)
        masked = es.minus_kind(EffectKind.EXN)
        assert not masked.has(EffectKind.EXN)
        assert masked.has(EffectKind.IO)

    def test_has(self):
        es = EffectSet.of(IO, State("x"))
        assert es.has(EffectKind.IO)
        assert es.has(EffectKind.STATE)
        assert not es.has(EffectKind.EXN)

    def test_get(self):
        es = EffectSet.of(State("x"), State("y"), IO)
        states = es.get(EffectKind.STATE)
        assert len(states) == 2

    def test_subtype(self):
        pure = EffectSet.pure()
        io = EffectSet.of(IO)
        io_state = EffectSet.of(IO, State("x"))
        assert pure <= io
        assert pure <= io_state
        assert io <= io_state
        assert not io_state <= io


# ============================================================
# Effect Subtyping & Composition Tests
# ============================================================

class TestEffectComposition:
    def test_compose_effects(self):
        a = EffectSet.of(IO)
        b = EffectSet.of(State("x"))
        c = compose_effects(a, b)
        assert c.has(EffectKind.IO)
        assert c.has(EffectKind.STATE)

    def test_handle_effect(self):
        body = EffectSet.of(IO, Exn("Error"))
        handled = handle_effect(body, Exn("Error"))
        assert not handled.has(EffectKind.EXN)
        assert handled.has(EffectKind.IO)

    def test_handle_all_exn(self):
        body = EffectSet.of(Exn("A"), Exn("B"), IO)
        handled = handle_all_exn(body)
        assert not handled.has(EffectKind.EXN)
        assert handled.has(EffectKind.IO)

    def test_effect_subtype_pure_is_bottom(self):
        assert effect_subtype(EffectSet.pure(), EffectSet.of(IO))
        assert effect_subtype(EffectSet.pure(), EffectSet.of(State("x"), Exn("E")))

    def test_effect_subtype_not_supertype(self):
        assert not effect_subtype(EffectSet.of(IO, State("x")), EffectSet.of(IO))


# ============================================================
# Effect Polymorphism Tests
# ============================================================

class TestEffectPolymorphism:
    def test_poly_instantiate(self):
        poly = PolyEffectfulType(
            effect_vars=("E",),
            params=(("f", T_INT),),
            ret=T_INT,
            effects=EffectVar("E"),
        )
        concrete = poly.instantiate({"E": EffectSet.of(IO)})
        assert concrete.effects.has(EffectKind.IO)

    def test_poly_instantiate_pure(self):
        poly = PolyEffectfulType(
            effect_vars=("E",),
            params=(("f", T_INT),),
            ret=T_INT,
            effects=EffectVar("E"),
        )
        concrete = poly.instantiate({"E": EffectSet.pure()})
        assert concrete.effects.is_pure


# ============================================================
# Effect Inference Tests
# ============================================================

class TestEffectInference:
    def test_pure_function(self):
        source = """
fn add(x, y) {
    return x + y;
}
"""
        sigs = infer_effects(source)
        assert "add" in sigs
        assert sigs["add"].effects.is_pure

    def test_state_assignment(self):
        source = """
fn inc(x) {
    x = x + 1;
    return x;
}
"""
        sigs = infer_effects(source)
        assert sigs["inc"].effects.has(EffectKind.STATE)

    def test_io_print(self):
        source = """
fn greet(x) {
    print(x);
}
"""
        sigs = infer_effects(source)
        assert sigs["greet"].effects.has(EffectKind.IO)

    def test_div_effect(self):
        source = """
fn loop_fn(n) {
    let i = 0;
    while (i < n) {
        i = i + 1;
    }
    return i;
}
"""
        sigs = infer_effects(source)
        assert sigs["loop_fn"].effects.has(EffectKind.DIV)

    def test_division_exn(self):
        source = """
fn divide(a, b) {
    return a / b;
}
"""
        sigs = infer_effects(source)
        assert sigs["divide"].effects.has(EffectKind.EXN)

    def test_multiple_effects(self):
        source = """
fn complex_fn(x) {
    x = x + 1;
    print(x);
    return x / x;
}
"""
        sigs = infer_effects(source)
        eff = sigs["complex_fn"].effects
        assert eff.has(EffectKind.STATE)
        assert eff.has(EffectKind.IO)
        assert eff.has(EffectKind.EXN)

    def test_conditional_effects(self):
        source = """
fn maybe_print(x) {
    if (x > 0) {
        print(x);
    }
    return x;
}
"""
        sigs = infer_effects(source)
        assert sigs["maybe_print"].effects.has(EffectKind.IO)

    def test_callee_effects_propagate(self):
        source = """
fn helper(x) {
    print(x);
    return x;
}
fn caller(y) {
    return helper(y);
}
"""
        sigs = infer_effects(source)
        assert sigs["caller"].effects.has(EffectKind.IO)

    def test_top_level_effects(self):
        source = """
let x = 42;
print(x);
"""
        sigs = infer_effects(source)
        assert "__main__" in sigs
        assert sigs["__main__"].effects.has(EffectKind.IO)

    def test_nested_function_no_effect_at_define(self):
        source = """
fn outer(x) {
    fn inner(y) {
        print(y);
    }
    return x;
}
"""
        sigs = infer_effects(source)
        assert sigs["outer"].effects.is_pure

    def test_division_by_literal_nonzero_safe(self):
        source = """
fn half(x) {
    return x / 2;
}
"""
        sigs = infer_effects(source)
        assert not sigs["half"].effects.has(EffectKind.EXN)

    def test_division_by_literal_zero_unsafe(self):
        source = """
fn bad(x) {
    return x / 0;
}
"""
        sigs = infer_effects(source)
        assert sigs["bad"].effects.has(EffectKind.EXN)

    def test_while_adds_div(self):
        source = """
fn spin(n) {
    while (n > 0) {
        n = n - 1;
    }
    return 0;
}
"""
        sigs = infer_effects(source)
        assert sigs["spin"].effects.has(EffectKind.DIV)
        assert sigs["spin"].effects.has(EffectKind.STATE)


# ============================================================
# Effect Checking Tests
# ============================================================

class TestEffectChecking:
    def test_correct_declaration(self):
        source = """
fn greet(x) {
    print(x);
}
"""
        result = check_effects(source, declared={"greet": EffectSet.of(IO)})
        assert result.ok

    def test_undeclared_effect_error(self):
        source = """
fn greet(x) {
    print(x);
}
"""
        result = check_effects(source, declared={"greet": EffectSet.pure()})
        assert not result.ok
        assert len(result.errors) == 1
        assert "undeclared" in result.errors[0].message.lower()

    def test_over_declaration_warning(self):
        source = """
fn add(x, y) {
    return x + y;
}
"""
        result = check_effects(source, declared={"add": EffectSet.of(IO)})
        assert result.ok
        assert len(result.warnings) == 1
        assert "unnecessary" in result.warnings[0].message.lower()

    def test_no_declaration_pure_ok(self):
        source = """
fn add(x, y) {
    return x + y;
}
"""
        result = check_effects(source)
        assert result.ok

    def test_no_declaration_effectful_warning(self):
        source = """
fn greet(x) {
    print(x);
}
"""
        result = check_effects(source)
        assert result.ok
        assert len(result.warnings) == 1
        assert "no effect declaration" in result.warnings[0].message.lower()

    def test_state_wildcard_covers_specific(self):
        source = """
fn swap(a, b) {
    let tmp = a;
    a = b;
    b = tmp;
    return 0;
}
"""
        result = check_effects(source, declared={"swap": EffectSet.of(State("*"))})
        assert result.ok

    def test_multiple_state_specific(self):
        source = """
fn swap(a, b) {
    let tmp = a;
    a = b;
    b = tmp;
    return 0;
}
"""
        result = check_effects(source, declared={
            "swap": EffectSet.of(State("a"), State("b"))
        })
        assert result.ok


# ============================================================
# Effect Verification Tests
# ============================================================

class TestEffectVerification:
    def test_verify_pure(self):
        source = """
fn add(x, y) {
    return x + y;
}
"""
        result = verify_effects(source, declared={"add": EffectSet.pure()})
        assert result.ok

    def test_verify_state_frame(self):
        source = """
fn inc_x(x) {
    x = x + 1;
    return x;
}
"""
        result = verify_effects(source, declared={"inc_x": EffectSet.of(State("x"))})
        assert result.ok
        frame_checks = [c for c in result.checks if "frame" in c.message.lower()]
        assert len(frame_checks) > 0

    def test_verify_exception_free(self):
        source = """
fn safe_add(x, y) {
    return x + y;
}
"""
        result = verify_effects(source, declared={"safe_add": EffectSet.pure()})
        assert result.ok
        exn_checks = [c for c in result.checks if "exception-free" in c.message.lower()]
        assert len(exn_checks) > 0

    def test_verify_catches_missing_effect(self):
        source = """
fn bad(x) {
    print(x);
}
"""
        result = verify_effects(source, declared={"bad": EffectSet.pure()})
        assert not result.ok


# ============================================================
# Effectful Function Type Tests
# ============================================================

class TestEffectfulFuncType:
    def test_pure_func_type(self):
        ft = EffectfulFuncType(
            params=(("x", T_INT), ("y", T_INT)),
            ret=T_INT,
            effects=EffectSet.pure(),
        )
        assert "pure" not in repr(ft) or "!" not in repr(ft)

    def test_effectful_func_type(self):
        ft = EffectfulFuncType(
            params=(("x", T_INT),),
            ret=T_UNIT,
            effects=EffectSet.of(IO),
        )
        assert "!" in repr(ft)
        assert "io" in repr(ft)


# ============================================================
# Integration: Effect System on Realistic Programs
# ============================================================

class TestIntegration:
    def test_recursive_function(self):
        source = """
fn factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}
"""
        sigs = infer_effects(source)
        assert sigs["factorial"].effects.is_pure

    def test_stateful_accumulator(self):
        source = """
fn sum_to(n) {
    let acc = 0;
    let i = 0;
    while (i < n) {
        acc = acc + i;
        i = i + 1;
    }
    return acc;
}
"""
        sigs = infer_effects(source)
        eff = sigs["sum_to"].effects
        assert eff.has(EffectKind.STATE)
        assert eff.has(EffectKind.DIV)

    def test_io_logging_function(self):
        source = """
fn compute_and_log(x) {
    let result = x * x;
    print(result);
    return result;
}
"""
        sigs = infer_effects(source)
        assert sigs["compute_and_log"].effects.has(EffectKind.IO)

    def test_multi_function_program(self):
        source = """
fn pure_helper(x) {
    return x + 1;
}
fn io_fn(x) {
    let y = pure_helper(x);
    print(y);
    return y;
}
fn stateful_fn(x) {
    x = x + 1;
    return x;
}
"""
        sigs = infer_effects(source)
        assert sigs["pure_helper"].effects.is_pure
        assert sigs["io_fn"].effects.has(EffectKind.IO)
        assert sigs["stateful_fn"].effects.has(EffectKind.STATE)

    def test_effect_handler_composition(self):
        body = EffectSet.of(IO, Exn("FileNotFound"), Exn("ParseError"), State("buf"))
        after_handle = handle_effect(body, Exn("FileNotFound"))
        assert after_handle.has(EffectKind.EXN)
        assert after_handle.has(EffectKind.IO)
        assert after_handle.has(EffectKind.STATE)

        after_handle_all = handle_all_exn(after_handle)
        assert not after_handle_all.has(EffectKind.EXN)
        assert after_handle_all.has(EffectKind.IO)

    def test_effect_lattice_subtyping(self):
        pure = EffectSet.pure()
        io = EffectSet.of(IO)
        io_exn = EffectSet.of(IO, Exn("E"))
        all_eff = EffectSet.of(IO, State("x"), Exn("E"), DIV)

        assert effect_subtype(pure, io)
        assert effect_subtype(pure, all_eff)
        assert effect_subtype(io, io_exn)
        assert effect_subtype(io_exn, all_eff)
        assert not effect_subtype(all_eff, io)

    def test_else_branch_effects(self):
        source = """
fn cond_io(x) {
    if (x > 0) {
        return x;
    } else {
        print(x);
        return 0;
    }
}
"""
        sigs = infer_effects(source)
        assert sigs["cond_io"].effects.has(EffectKind.IO)

    def test_chained_calls(self):
        source = """
fn step1(x) {
    x = x + 1;
    return x;
}
fn step2(x) {
    print(x);
    return x;
}
fn pipeline(x) {
    let a = step1(x);
    let b = step2(a);
    return b;
}
"""
        sigs = infer_effects(source)
        eff = sigs["pipeline"].effects
        assert eff.has(EffectKind.STATE)
        assert eff.has(EffectKind.IO)
