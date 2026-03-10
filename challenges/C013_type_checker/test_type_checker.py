"""
Tests for the Type Checker (C013)
AgentZero Session 014

Coverage:
  - Type representations and equality
  - Unification algorithm
  - Literal type inference
  - Variable type tracking
  - Arithmetic type checking
  - String operations
  - Comparison and equality
  - Logical operators
  - Unary operators
  - Control flow (if/while)
  - Function declarations and calls
  - Return type checking
  - Scoping
  - Type inference
  - Error detection (the main point)
  - Complex programs
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from type_checker import (
    INT, FLOAT, STRING, BOOL, VOID,
    TInt, TFloat, TString, TBool, TVoid, TFunc, TVar,
    TypeEnv, resolve, unify, UnificationError, occurs_in,
    types_compatible, TypeError_,
    TypeChecker, check_source, check_program, format_errors, parse,
)


# ============================================================
# Type Representations
# ============================================================

class TestTypes:
    def test_int_repr(self):
        assert repr(INT) == "int"

    def test_float_repr(self):
        assert repr(FLOAT) == "float"

    def test_string_repr(self):
        assert repr(STRING) == "string"

    def test_bool_repr(self):
        assert repr(BOOL) == "bool"

    def test_void_repr(self):
        assert repr(VOID) == "void"

    def test_func_repr(self):
        t = TFunc((INT, FLOAT), BOOL)
        assert repr(t) == "fn(int, float) -> bool"

    def test_func_no_params(self):
        t = TFunc((), INT)
        assert repr(t) == "fn() -> int"

    def test_tvar_unresolved(self):
        tv = TVar(1)
        assert "?T1" in repr(tv)

    def test_tvar_resolved(self):
        tv = TVar(1, bound=INT)
        assert repr(tv) == "int"

    def test_type_equality(self):
        assert INT == TInt()
        assert FLOAT == TFloat()
        assert STRING == TString()
        assert BOOL == TBool()
        assert VOID == TVoid()

    def test_type_inequality(self):
        assert INT != FLOAT
        assert INT != STRING
        assert BOOL != INT

    def test_func_equality(self):
        t1 = TFunc((INT,), BOOL)
        t2 = TFunc((INT,), BOOL)
        assert t1 == t2

    def test_func_inequality(self):
        t1 = TFunc((INT,), BOOL)
        t2 = TFunc((FLOAT,), BOOL)
        assert t1 != t2


# ============================================================
# Type Environment
# ============================================================

class TestTypeEnv:
    def test_define_and_lookup(self):
        env = TypeEnv()
        env.define("x", INT)
        assert env.lookup("x") == INT

    def test_undefined(self):
        env = TypeEnv()
        assert env.lookup("x") is None

    def test_child_scope(self):
        parent = TypeEnv()
        parent.define("x", INT)
        child = parent.child()
        assert child.lookup("x") == INT

    def test_child_shadows_parent(self):
        parent = TypeEnv()
        parent.define("x", INT)
        child = parent.child()
        child.define("x", FLOAT)
        assert child.lookup("x") == FLOAT
        assert parent.lookup("x") == INT

    def test_child_does_not_leak_to_parent(self):
        parent = TypeEnv()
        child = parent.child()
        child.define("y", BOOL)
        assert parent.lookup("y") is None


# ============================================================
# Unification
# ============================================================

class TestUnification:
    def test_same_types(self):
        assert unify(INT, INT) == INT
        assert unify(FLOAT, FLOAT) == FLOAT

    def test_int_promotes_to_float(self):
        assert unify(INT, FLOAT) == FLOAT
        assert unify(FLOAT, INT) == FLOAT

    def test_incompatible_types(self):
        with pytest.raises(UnificationError):
            unify(INT, STRING)

    def test_tvar_binds_to_concrete(self):
        tv = TVar(1)
        result = unify(tv, INT)
        assert result == INT
        assert tv.bound == INT

    def test_tvar_binds_right(self):
        tv = TVar(2)
        result = unify(FLOAT, tv)
        assert result == FLOAT
        assert tv.bound == FLOAT

    def test_two_tvars(self):
        tv1 = TVar(3)
        tv2 = TVar(4)
        unify(tv1, tv2)
        # One should be bound to the other
        assert resolve(tv1) is resolve(tv2)

    def test_func_unification(self):
        t1 = TFunc((INT,), BOOL)
        t2 = TFunc((INT,), BOOL)
        result = unify(t1, t2)
        assert isinstance(result, TFunc)

    def test_func_arity_mismatch(self):
        t1 = TFunc((INT,), BOOL)
        t2 = TFunc((INT, INT), BOOL)
        with pytest.raises(UnificationError):
            unify(t1, t2)

    def test_func_param_unification(self):
        tv = TVar(5)
        t1 = TFunc((tv,), BOOL)
        t2 = TFunc((INT,), BOOL)
        result = unify(t1, t2)
        assert resolve(tv) == INT

    def test_occurs_check(self):
        tv = TVar(6)
        with pytest.raises(UnificationError, match="Infinite"):
            unify(tv, TFunc((tv,), INT))

    def test_resolve_chain(self):
        tv1 = TVar(7)
        tv2 = TVar(8)
        tv1.bound = tv2
        tv2.bound = INT
        assert resolve(tv1) == INT


class TestTypesCompatible:
    def test_same_types(self):
        assert types_compatible(INT, INT)

    def test_int_to_float(self):
        assert types_compatible(INT, FLOAT)

    def test_float_to_int_incompatible(self):
        assert not types_compatible(FLOAT, INT)

    def test_string_to_int_incompatible(self):
        assert not types_compatible(STRING, INT)

    def test_tvar_always_compatible(self):
        assert types_compatible(TVar(99), INT)
        assert types_compatible(STRING, TVar(99))

    def test_func_compatible(self):
        t1 = TFunc((INT,), BOOL)
        t2 = TFunc((INT,), BOOL)
        assert types_compatible(t1, t2)


# ============================================================
# Literal Type Inference
# ============================================================

class TestLiterals:
    def test_int_literal(self):
        errors, _ = check_source("let x = 42;")
        assert len(errors) == 0

    def test_float_literal(self):
        errors, _ = check_source("let x = 3.14;")
        assert len(errors) == 0

    def test_string_literal(self):
        errors, _ = check_source('let x = "hello";')
        assert len(errors) == 0

    def test_bool_true(self):
        errors, _ = check_source("let x = true;")
        assert len(errors) == 0

    def test_bool_false(self):
        errors, _ = check_source("let x = false;")
        assert len(errors) == 0


# ============================================================
# Variable Type Tracking
# ============================================================

class TestVariables:
    def test_var_defined(self):
        errors, checker = check_source("let x = 5; let y = x;")
        assert len(errors) == 0

    def test_undefined_var(self):
        errors, _ = check_source("let x = y;")
        assert len(errors) == 1
        assert "Undefined variable 'y'" in errors[0].message

    def test_var_type_tracked(self):
        errors, checker = check_source("let x = 5;")
        assert len(errors) == 0
        t = checker.env.lookup("x")
        assert resolve(t) == INT

    def test_var_type_float(self):
        errors, checker = check_source("let x = 3.14;")
        assert len(errors) == 0
        t = checker.env.lookup("x")
        assert resolve(t) == FLOAT

    def test_var_type_string(self):
        errors, checker = check_source('let x = "hi";')
        assert len(errors) == 0
        t = checker.env.lookup("x")
        assert resolve(t) == STRING

    def test_var_type_bool(self):
        errors, checker = check_source("let x = true;")
        assert len(errors) == 0
        t = checker.env.lookup("x")
        assert resolve(t) == BOOL


# ============================================================
# Arithmetic
# ============================================================

class TestArithmetic:
    def test_int_plus_int(self):
        errors, checker = check_source("let x = 1 + 2;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == INT

    def test_float_plus_float(self):
        errors, checker = check_source("let x = 1.0 + 2.0;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == FLOAT

    def test_int_plus_float_promotes(self):
        errors, checker = check_source("let x = 1 + 2.0;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == FLOAT

    def test_string_plus_string(self):
        errors, checker = check_source('let x = "a" + "b";')
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == STRING

    def test_int_plus_string_error(self):
        errors, _ = check_source('let x = 1 + "hello";')
        assert len(errors) >= 1
        assert any("Cannot apply" in e.message for e in errors)

    def test_bool_plus_int_error(self):
        errors, _ = check_source("let x = true + 1;")
        assert len(errors) >= 1

    def test_int_minus_int(self):
        errors, checker = check_source("let x = 10 - 3;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == INT

    def test_int_multiply(self):
        errors, checker = check_source("let x = 3 * 4;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == INT

    def test_int_divide(self):
        errors, checker = check_source("let x = 10 / 3;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == INT

    def test_float_divide(self):
        errors, checker = check_source("let x = 10.0 / 3.0;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == FLOAT

    def test_modulo(self):
        errors, checker = check_source("let x = 10 % 3;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == INT

    def test_chained_arithmetic(self):
        errors, checker = check_source("let x = 1 + 2 * 3 - 4;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == INT


# ============================================================
# Comparison and Equality
# ============================================================

class TestComparison:
    def test_int_less_than(self):
        errors, checker = check_source("let x = 1 < 2;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL

    def test_float_comparison(self):
        errors, checker = check_source("let x = 1.0 > 2.0;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL

    def test_int_float_comparison(self):
        errors, checker = check_source("let x = 1 <= 2.0;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL

    def test_string_comparison(self):
        errors, checker = check_source('let x = "a" < "b";')
        assert len(errors) == 0

    def test_bool_comparison_error(self):
        errors, _ = check_source("let x = true < false;")
        assert len(errors) >= 1

    def test_equality_same_type(self):
        errors, checker = check_source("let x = 1 == 2;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL

    def test_equality_int_float(self):
        errors, _ = check_source("let x = 1 == 2.0;")
        assert len(errors) == 0

    def test_equality_different_types_error(self):
        errors, _ = check_source('let x = 1 == "hello";')
        assert len(errors) >= 1

    def test_not_equal(self):
        errors, checker = check_source("let x = 1 != 2;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL


# ============================================================
# Logical Operators
# ============================================================

class TestLogical:
    def test_and(self):
        errors, checker = check_source("let x = true and false;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL

    def test_or(self):
        errors, checker = check_source("let x = true or false;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL

    def test_and_with_non_bool_error(self):
        errors, _ = check_source("let x = 1 and true;")
        assert len(errors) >= 1

    def test_or_with_non_bool_error(self):
        errors, _ = check_source('let x = true or "yes";')
        assert len(errors) >= 1

    def test_chained_logical(self):
        errors, _ = check_source("let x = true and true or false;")
        assert len(errors) == 0


# ============================================================
# Unary Operators
# ============================================================

class TestUnary:
    def test_negate_int(self):
        errors, checker = check_source("let x = -5;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == INT

    def test_negate_float(self):
        errors, checker = check_source("let x = -3.14;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == FLOAT

    def test_negate_string_error(self):
        errors, _ = check_source('let x = -"hello";')
        assert len(errors) >= 1
        assert any("Cannot negate" in e.message for e in errors)

    def test_not_bool(self):
        errors, checker = check_source("let x = not true;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL

    def test_not_int_error(self):
        errors, _ = check_source("let x = not 5;")
        assert len(errors) >= 1
        assert any("'not' requires bool" in e.message for e in errors)


# ============================================================
# Control Flow
# ============================================================

class TestControlFlow:
    def test_if_bool_condition(self):
        errors, _ = check_source("if (true) { let x = 1; }")
        assert len(errors) == 0

    def test_if_int_condition_error(self):
        errors, _ = check_source("if (42) { let x = 1; }")
        assert len(errors) >= 1
        assert any("Condition must be bool" in e.message for e in errors)

    def test_if_else(self):
        errors, _ = check_source("if (true) { let x = 1; } else { let x = 2; }")
        assert len(errors) == 0

    def test_while_bool_condition(self):
        errors, _ = check_source("let x = true; while (x) { x = false; }")
        assert len(errors) == 0

    def test_while_int_condition_error(self):
        errors, _ = check_source("let x = 1; while (x) { x = 0; }")
        assert len(errors) >= 1
        assert any("While condition must be bool" in e.message for e in errors)

    def test_if_with_comparison(self):
        errors, _ = check_source("let x = 5; if (x > 3) { let y = 1; }")
        assert len(errors) == 0

    def test_while_with_comparison(self):
        errors, _ = check_source("let x = 10; while (x > 0) { x = x - 1; }")
        assert len(errors) == 0


# ============================================================
# Assignment
# ============================================================

class TestAssignment:
    def test_assign_same_type(self):
        errors, _ = check_source("let x = 5; x = 10;")
        assert len(errors) == 0

    def test_assign_int_to_float_promotes(self):
        errors, _ = check_source("let x = 1.0; x = 5;")
        assert len(errors) == 0

    def test_assign_wrong_type_error(self):
        errors, _ = check_source('let x = 5; x = "hello";')
        assert len(errors) >= 1
        assert any("Cannot assign" in e.message for e in errors)

    def test_assign_undefined_error(self):
        errors, _ = check_source("z = 5;")
        assert len(errors) >= 1
        assert any("undefined" in e.message.lower() for e in errors)


# ============================================================
# Functions
# ============================================================

class TestFunctions:
    def test_simple_function(self):
        src = """
        fn add(a, b) {
            return a + b;
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_function_call(self):
        src = """
        fn double(x) {
            return x + x;
        }
        let y = double(5);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_function_wrong_arg_count(self):
        src = """
        fn add(a, b) {
            return a + b;
        }
        let x = add(1, 2, 3);
        """
        errors, _ = check_source(src)
        assert len(errors) >= 1
        assert any("expects 2 args, got 3" in e.message for e in errors)

    def test_function_call_non_function_error(self):
        src = """
        let x = 5;
        let y = x(1);
        """
        errors, _ = check_source(src)
        assert len(errors) >= 1
        assert any("not a function" in e.message for e in errors)

    def test_function_return_type_inference(self):
        src = """
        fn five() {
            return 5;
        }
        let x = five();
        """
        errors, checker = check_source(src)
        assert len(errors) == 0

    def test_undefined_function(self):
        src = """
        let x = foo(1, 2);
        """
        errors, _ = check_source(src)
        assert len(errors) >= 1
        assert any("Undefined function 'foo'" in e.message for e in errors)

    def test_recursive_function(self):
        src = """
        fn fact(n) {
            if (n <= 1) {
                return 1;
            }
            return n * fact(n - 1);
        }
        let x = fact(5);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_return_outside_function(self):
        src = "return 5;"
        errors, _ = check_source(src)
        assert len(errors) >= 1
        assert any("outside of function" in e.message for e in errors)


# ============================================================
# Scoping
# ============================================================

class TestScoping:
    def test_block_scope(self):
        src = """
        let x = 5;
        {
            let y = 10;
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_inner_scope_sees_outer(self):
        src = """
        let x = 5;
        {
            let y = x + 1;
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0


# ============================================================
# Print Statement
# ============================================================

class TestPrint:
    def test_print_int(self):
        errors, _ = check_source("print(42);")
        assert len(errors) == 0

    def test_print_string(self):
        errors, _ = check_source('print("hello");')
        assert len(errors) == 0

    def test_print_bool(self):
        errors, _ = check_source("print(true);")
        assert len(errors) == 0

    def test_print_expr(self):
        errors, _ = check_source("print(1 + 2);")
        assert len(errors) == 0


# ============================================================
# Type Inference
# ============================================================

class TestInference:
    def test_infer_from_assignment(self):
        src = """
        fn id(x) {
            return x;
        }
        let y = id(42);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_infer_arithmetic_result(self):
        src = """
        let x = 1;
        let y = 2;
        let z = x + y;
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        assert resolve(checker.env.lookup("z")) == INT

    def test_infer_comparison_result(self):
        src = """
        let x = 1 < 2;
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL


# ============================================================
# Error Formatting
# ============================================================

class TestFormatting:
    def test_no_errors(self):
        result = format_errors([])
        assert "No type errors" in result

    def test_single_error(self):
        errors = [TypeError_("bad type", 5)]
        result = format_errors(errors)
        assert "1 type error" in result
        assert "Line 5" in result
        assert "bad type" in result

    def test_multiple_errors(self):
        errors = [TypeError_("err1", 1), TypeError_("err2", 2)]
        result = format_errors(errors)
        assert "2 type error" in result


# ============================================================
# Complex Programs
# ============================================================

class TestComplexPrograms:
    def test_fibonacci(self):
        src = """
        fn fib(n) {
            if (n <= 1) {
                return n;
            }
            return fib(n - 1) + fib(n - 2);
        }
        let result = fib(10);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_counter_loop(self):
        src = """
        let sum = 0;
        let i = 1;
        while (i <= 10) {
            sum = sum + i;
            i = i + 1;
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_nested_functions(self):
        src = """
        fn outer(x) {
            fn inner(y) {
                return y + 1;
            }
            return inner(x);
        }
        let z = outer(5);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_multiple_errors_in_program(self):
        src = """
        let x = 5;
        let y = true;
        let z = x + y;
        if (42) {
            let w = 1;
        }
        """
        errors, _ = check_source(src)
        assert len(errors) >= 2  # arithmetic error + condition error

    def test_mixed_valid_and_invalid(self):
        src = """
        let a = 1;
        let b = 2;
        let c = a + b;
        let d = "hello";
        let e = a + d;
        """
        errors, _ = check_source(src)
        assert len(errors) >= 1
        # Valid parts should still be checked
        assert any("Cannot apply" in e.message for e in errors)

    def test_print_in_function(self):
        src = """
        fn greet(name) {
            print(name);
            return 0;
        }
        greet("world");
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_if_else_chains(self):
        src = """
        let x = 5;
        if (x > 10) {
            print(1);
        } else if (x > 5) {
            print(2);
        } else {
            print(3);
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_function_as_value_error(self):
        src = """
        fn add(a, b) {
            return a + b;
        }
        let x = add + 1;
        """
        errors, _ = check_source(src)
        assert len(errors) >= 1

    def test_string_concat_and_compare(self):
        src = """
        let greeting = "hello" + " " + "world";
        let same = greeting == "hello world";
        """
        errors, checker = check_source(src)
        assert len(errors) == 0
        assert resolve(checker.env.lookup("greeting")) == STRING
        assert resolve(checker.env.lookup("same")) == BOOL

    def test_deeply_nested_scopes(self):
        src = """
        let a = 1;
        {
            let b = a + 1;
            {
                let c = b + 1;
                {
                    let d = c + 1;
                    print(d);
                }
            }
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        errors, _ = check_source("")
        assert len(errors) == 0

    def test_single_literal_statement(self):
        errors, _ = check_source("42;")
        assert len(errors) == 0

    def test_negate_negate(self):
        errors, checker = check_source("let x = - -5;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == INT

    def test_not_not(self):
        errors, checker = check_source("let x = not not true;")
        assert len(errors) == 0
        assert resolve(checker.env.lookup("x")) == BOOL

    def test_function_no_return(self):
        src = """
        fn noop() {
            let x = 1;
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_function_void_return(self):
        src = """
        fn noop() {
            return;
        }
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_multiple_functions(self):
        src = """
        fn add(a, b) {
            return a + b;
        }
        fn mul(a, b) {
            return a * b;
        }
        let x = add(1, 2);
        let y = mul(3, 4);
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_reassign_multiple_times(self):
        src = """
        let x = 1;
        x = 2;
        x = 3;
        """
        errors, _ = check_source(src)
        assert len(errors) == 0

    def test_assign_type_mismatch_after_valid(self):
        src = """
        let x = 1;
        x = 2;
        x = "oops";
        """
        errors, _ = check_source(src)
        assert len(errors) >= 1
        assert any("Cannot assign" in e.message for e in errors)
