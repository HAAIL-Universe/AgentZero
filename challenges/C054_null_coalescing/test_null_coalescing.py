"""
Tests for C054: Null Coalescing Operator (?? and ??=)
"""

import pytest
from null_coalescing import run, execute, lex, TokenType, ParseError, ModuleRegistry


def val(source):
    result, _ = run(source)
    return result

def out(source):
    _, output = run(source)
    return output


# ============================================================
# Basic ?? operator
# ============================================================

class TestBasicNullCoalescing:
    def test_null_returns_right(self):
        assert val("null ?? 42;") == 42

    def test_non_null_returns_left(self):
        assert val("10 ?? 42;") == 10

    def test_zero_is_not_null(self):
        assert val("0 ?? 42;") == 0

    def test_false_is_not_null(self):
        assert val("false ?? 42;") is False

    def test_empty_string_is_not_null(self):
        assert val('"" ?? "default";') == ""

    def test_empty_array_is_not_null(self):
        assert val("[] ?? [1, 2, 3];") == []

    def test_empty_hash_is_not_null(self):
        result = val("{} ?? {a: 1};")
        assert isinstance(result, dict) and len(result) == 0

    def test_string_value(self):
        assert val('"hello" ?? "default";') == "hello"

    def test_null_with_string_default(self):
        assert val('null ?? "default";') == "default"


# ============================================================
# Variables with ??
# ============================================================

class TestVariableCoalescing:
    def test_null_variable(self):
        assert val("let x = null; x ?? 99;") == 99

    def test_non_null_variable(self):
        assert val("let x = 5; x ?? 99;") == 5

    def test_both_variables(self):
        assert val("let x = null; let y = 42; x ?? y;") == 42

    def test_both_variables_non_null(self):
        assert val("let x = 10; let y = 42; x ?? y;") == 10


# ============================================================
# Chaining: a ?? b ?? c
# ============================================================

class TestChaining:
    def test_first_non_null(self):
        assert val("1 ?? 2 ?? 3;") == 1

    def test_second_non_null(self):
        assert val("null ?? 2 ?? 3;") == 2

    def test_third_non_null(self):
        assert val("null ?? null ?? 3;") == 3

    def test_all_null(self):
        assert val("null ?? null ?? null;") is None

    def test_four_levels(self):
        assert val("null ?? null ?? null ?? 99;") == 99

    def test_four_levels_second(self):
        assert val("null ?? 50 ?? null ?? 99;") == 50


# ============================================================
# ?? with expressions
# ============================================================

class TestExpressions:
    def test_with_arithmetic(self):
        assert val("null ?? (1 + 2);") == 3

    def test_with_function_call(self):
        assert val("let f = fn() { return 42; }; null ?? f();") == 42

    def test_with_array_index(self):
        assert val("let a = [10, 20]; null ?? a[1];") == 20

    def test_with_hash_access(self):
        assert val('let h = {x: 99}; null ?? h.x;') == 99

    def test_in_let_declaration(self):
        assert out("let x = null ?? 42; print x;") == ['42']

    def test_in_if_condition(self):
        assert out('if (null ?? true) { print "yes"; }') == ['yes']

    def test_in_function_argument(self):
        assert out("let f = fn(x) { print x; }; f(null ?? 7);") == ['7']

    def test_in_return_statement(self):
        assert val("let f = fn() { return null ?? 55; }; f();") == 55

    def test_in_array_literal(self):
        assert val("[null ?? 1, null ?? 2, 3];") == [1, 2, 3]


# ============================================================
# Precedence
# ============================================================

class TestPrecedence:
    def test_lower_than_comparison(self):
        # ?? is lower than ==, so: null ?? (1 == 1)
        assert val("null ?? 1 == 1;") is True

    def test_lower_than_and(self):
        # ?? is lower than and: null ?? (true and false)
        assert val("null ?? true and false;") is False

    def test_or_lower_than_coalesce(self):
        # or calls null_coalesce_expr: 5 ?? 10 || false -> (5 ?? 10) || false
        # Since ?? is called from or_expr, 5 ?? 10 -> 5, then 5 || false -> 5
        assert val("5 ?? 10 || false;") == 5

    def test_higher_than_pipe(self):
        assert val("let f = fn(x) { return x + 1; }; null ?? 5 |> f;") == 6

    def test_null_or_complex(self):
        # null ?? false || true -> (null ?? false) || true -> false || true -> true
        assert val("null ?? false || true;") is True

    def test_parens_override(self):
        assert val("(null ?? 5) + 10;") == 15


# ============================================================
# ??= operator
# ============================================================

class TestNullCoalescingAssignment:
    def test_assign_when_null(self):
        assert out("let x = null; x ??= 42; print x;") == ['42']

    def test_no_assign_when_non_null(self):
        assert out("let x = 10; x ??= 42; print x;") == ['10']

    def test_assign_when_zero(self):
        assert out("let x = 0; x ??= 42; print x;") == ['0']

    def test_assign_when_false(self):
        assert out("let x = false; x ??= true; print x;") == ['false']

    def test_assign_when_empty_string(self):
        assert out('let x = ""; x ??= "default"; print x;') == ['']

    def test_assign_chain(self):
        assert out("let x = null; let y = null; x ??= 1; y ??= 2; print x; print y;") == ['1', '2']

    def test_assign_preserves_value(self):
        assert out("let x = null; x ??= 42; x ??= 99; print x;") == ['42']

    def test_dot_assign_when_null(self):
        assert out('let h = {x: null}; h.x ??= 42; print h.x;') == ['42']

    def test_dot_assign_when_non_null(self):
        assert out('let h = {x: 10}; h.x ??= 42; print h.x;') == ['10']

    def test_index_assign_when_null(self):
        assert out('let a = [null, 2, 3]; a[0] ??= 42; print a[0];') == ['42']

    def test_index_assign_when_non_null(self):
        assert out('let a = [1, 2, 3]; a[0] ??= 42; print a[0];') == ['1']


# ============================================================
# ?? with optional chaining
# ============================================================

class TestWithOptionalChaining:
    def test_optional_chain_null_coalesce(self):
        assert val("let h = null; h?.x ?? 42;") == 42

    def test_optional_chain_non_null(self):
        assert val("let h = {x: 10}; h?.x ?? 42;") == 10

    def test_deep_optional_chain(self):
        assert val("let h = {a: {b: null}}; h?.a?.b ?? 99;") == 99

    def test_optional_chain_method(self):
        assert val("""
            let obj = {
                get: fn() { return null; }
            };
            obj?.get() ?? "fallback";
        """) == "fallback"

    def test_optional_chain_on_null_base(self):
        assert val("let x = null; x?.y?.z ?? 0;") == 0

    def test_optional_chain_with_index(self):
        assert val("let arr = null; arr?.[0] ?? -1;") == -1


# ============================================================
# ?? with classes
# ============================================================

class TestWithClasses:
    def test_class_property_null_coalesce(self):
        assert val("""
            class Foo {
                init() {
                    this.x = null;
                }
            }
            let f = Foo();
            f.x ?? 42;
        """) == 42

    def test_class_method_returns_null(self):
        assert val("""
            class Bar {
                get() {
                    return null;
                }
            }
            let b = Bar();
            b.get() ?? "default";
        """) == "default"

    def test_class_property_non_null(self):
        assert val("""
            class Baz {
                init() {
                    this.x = 10;
                }
            }
            let b = Baz();
            b.x ?? 42;
        """) == 10


# ============================================================
# ?? with closures and generators
# ============================================================

class TestWithClosuresAndGenerators:
    def test_closure_returns_null(self):
        assert val("let f = fn() { return null; }; f() ?? 42;") == 42

    def test_closure_captures_null(self):
        assert val("let x = null; let f = fn() { return x ?? 99; }; f();") == 99

    def test_generator_next_null(self):
        assert val("""
            let g = fn*() {
                yield null;
                yield 10;
            };
            let gen = g();
            next(gen) ?? 42;
        """) == 42

    def test_generator_next_non_null(self):
        assert val("""
            let g = fn*() {
                yield 5;
            };
            let gen = g();
            next(gen) ?? 42;
        """) == 5


# ============================================================
# ?? with error handling
# ============================================================

class TestWithErrorHandling:
    def test_in_try_block(self):
        assert val("""
            let result = null;
            try {
                result = null ?? 42;
            } catch(e) {
                result = -1;
            }
            result;
        """) == 42

    def test_null_access_throws_not_coalesces(self):
        """Accessing .foo on null throws, ?? doesn't prevent that"""
        assert val("""
            let result = 0;
            try {
                let x = null;
                result = x.foo ?? 42;
            } catch(e) {
                result = -1;
            }
            result;
        """) == -1


# ============================================================
# ?? with destructuring
# ============================================================

class TestWithDestructuring:
    def test_in_destructuring_value(self):
        assert out("let [a, b] = [null ?? 1, null ?? 2]; print a; print b;") == ['1', '2']

    def test_destructured_null_coalesce(self):
        assert out("let h = {x: null, y: 5}; let {x, y} = h; print (x ?? 0); print (y ?? 0);") == ['0', '5']


# ============================================================
# ?? with string interpolation
# ============================================================

class TestWithStringInterpolation:
    def test_in_fstring(self):
        assert val('let x = null; f"value: ${x ?? 0}";') == "value: 0"

    def test_non_null_in_fstring(self):
        assert val('let x = 5; f"value: ${x ?? 0}";') == "value: 5"


# ============================================================
# ?? with spread and pipe
# ============================================================

class TestWithSpreadAndPipe:
    def test_with_spread_in_array(self):
        assert val("let a = null ?? [1, 2]; [...a, 3];") == [1, 2, 3]

    def test_with_pipe(self):
        assert val("let f = fn(x) { return x * 2; }; null ?? 5 |> f;") == 10


# ============================================================
# ?? with for-in loops
# ============================================================

class TestWithForIn:
    def test_null_coalesce_in_body(self):
        assert out("""
            let arr = [null, 1, null, 3];
            for (x in arr) {
                print (x ?? 0);
            }
        """) == ['0', '1', '0', '3']

    def test_null_coalesce_in_iterable(self):
        assert out("""
            let arr = null ?? [1, 2, 3];
            for (x in arr) {
                print x;
            }
        """) == ['1', '2', '3']


# ============================================================
# ?? with modules
# ============================================================

class TestWithModules:
    def test_null_coalesce_with_module_value(self):
        registry = ModuleRegistry()
        registry.register("config", 'export let value = null;')
        result, _ = run('import { value } from "config"; value ?? "default";', registry=registry)
        assert result == "default"


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_nested_null_coalesce(self):
        assert val("(null ?? null) ?? 42;") == 42

    def test_null_coalesce_with_negative(self):
        assert val("null ?? -1;") == -1

    def test_null_coalesce_boolean_true(self):
        assert val("null ?? true;") is True

    def test_null_coalesce_boolean_false(self):
        assert val("null ?? false;") is False

    def test_left_side_evaluated(self):
        """Non-null left side is kept"""
        assert val("let f = fn() { return 10; }; f() ?? 42;") == 10

    def test_right_side_evaluated_when_null(self):
        """Null left triggers right side"""
        assert val("let f = fn() { return 42; }; null ?? f();") == 42

    def test_null_coalesce_with_null_literal(self):
        assert val("null ?? null;") is None

    def test_multiple_types(self):
        assert val('null ?? "string";') == "string"
        assert val("null ?? [1, 2];") == [1, 2]
        assert val("null ?? {a: 1};") == {"a": 1}
        assert val("null ?? true;") is True
        assert val("null ?? 3.14;") == 3.14


# ============================================================
# Lexer tests
# ============================================================

class TestLexer:
    def test_question_question_token(self):
        tokens = lex("a ?? b")
        types = [t.type for t in tokens]
        assert TokenType.QUESTION_QUESTION in types

    def test_question_question_assign_token(self):
        tokens = lex("a ??= b")
        types = [t.type for t in tokens]
        assert TokenType.QUESTION_QUESTION_ASSIGN in types

    def test_question_dot_still_works(self):
        tokens = lex("a?.b")
        types = [t.type for t in tokens]
        assert TokenType.QUESTION_DOT in types

    def test_all_three_in_one(self):
        tokens = lex("a ?? b?.c ??= d")
        types = [t.type for t in tokens]
        assert TokenType.QUESTION_QUESTION in types
        assert TokenType.QUESTION_DOT in types
        assert TokenType.QUESTION_QUESTION_ASSIGN in types


# ============================================================
# ??= error cases
# ============================================================

class TestErrorCases:
    def test_cannot_coalesce_assign_to_literal(self):
        with pytest.raises(Exception):
            run("42 ??= 10;")

    def test_cannot_coalesce_assign_to_optional_chain(self):
        with pytest.raises(Exception):
            run("let h = {}; h?.x ??= 10;")


# ============================================================
# ?? with while loop
# ============================================================

class TestWithWhileLoop:
    def test_null_coalesce_in_while(self):
        assert out("""
            let arr = [null, null, 3, null, 5];
            let i = 0;
            while (i < 5) {
                print (arr[i] ?? 0);
                i = i + 1;
            }
        """) == ['0', '0', '3', '0', '5']


# ============================================================
# ?? with if-expressions (C049)
# ============================================================

class TestWithIfExpressions:
    def test_null_coalesce_as_if_condition(self):
        assert out("""
            let x = null;
            let val = if (x ?? false) "yes" else "no";
            print val;
        """) == ['no']

    def test_if_expr_in_coalesce(self):
        assert val('null ?? if (true) 42 else 0;') == 42


# ============================================================
# Print tests
# ============================================================

class TestPrint:
    def test_print_null_coalesce(self):
        assert out("print (null ?? 42);") == ['42']

    def test_print_coalesce_non_null(self):
        assert out("print (10 ?? 42);") == ['10']
