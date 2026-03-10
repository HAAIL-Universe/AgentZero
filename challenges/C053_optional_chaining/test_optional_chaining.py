"""Tests for C053 Optional Chaining."""

import pytest
from optional_chaining import run, execute, LexError, ParseError, VMError


# ============================================================
# Helper
# ============================================================

def val(source):
    """Run source and return the result value."""
    result, output = run(source)
    return result

def out(source):
    """Run source and return printed output as a single string."""
    result, output = run(source)
    return "\n".join(output)


# ============================================================
# Lexer tests
# ============================================================

class TestLexer:
    def test_question_dot_token(self):
        from optional_chaining import lex, TokenType
        tokens = lex("a?.b")
        types = [t.type for t in tokens]
        assert TokenType.QUESTION_DOT in types

    def test_question_dot_before_bracket(self):
        from optional_chaining import lex, TokenType
        tokens = lex("a?.[0]")
        types = [t.type for t in tokens]
        assert TokenType.QUESTION_DOT in types

    def test_question_dot_before_paren(self):
        from optional_chaining import lex, TokenType
        tokens = lex("a?.()")
        types = [t.type for t in tokens]
        assert TokenType.QUESTION_DOT in types

    def test_null_keyword_lexes(self):
        from optional_chaining import lex, TokenType
        tokens = lex("null")
        assert tokens[0].type == TokenType.NULL


# ============================================================
# Null literal
# ============================================================

class TestNullLiteral:
    def test_null_value(self):
        assert val("null;") is None

    def test_null_in_let(self):
        assert val("let x = null; x;") is None

    def test_null_equality(self):
        assert val("null == null;") is True

    def test_null_inequality_with_zero(self):
        assert val("null == 0;") is False

    def test_null_inequality_with_false(self):
        assert val("null == false;") is False

    def test_null_is_falsy(self):
        assert val("if (null) { 1; } else { 0; }") == 0

    def test_null_print(self):
        assert out("print null;") == "null"

    def test_null_in_array(self):
        assert val("[1, null, 3];") == [1, None, 3]

    def test_null_in_hash(self):
        assert val("{a: null};") == {"a": None}


# ============================================================
# Basic optional dot access: obj?.prop
# ============================================================

class TestOptionalDot:
    def test_null_dot_returns_null(self):
        assert val("let x = null; x?.name;") is None

    def test_nonnull_dot_returns_value(self):
        assert val('let x = {name: "alice"}; x?.name;') == "alice"

    def test_nonnull_nested_hash(self):
        assert val('let x = {a: {b: 42}}; x?.a;') == {"b": 42}

    def test_chain_null_propagation(self):
        assert val("let x = null; x?.a?.b;") is None

    def test_chain_nonnull(self):
        assert val('let x = {a: {b: 10}}; x?.a?.b;') == 10

    def test_chain_intermediate_null(self):
        assert val('let x = {a: null}; x?.a?.b;') is None

    def test_optional_dot_on_integer_errors(self):
        with pytest.raises(VMError):
            val("let x = 42; x?.name;")

    def test_optional_dot_on_string_errors(self):
        with pytest.raises(VMError):
            val('let x = "hello"; x?.name;')

    def test_optional_dot_on_array_errors(self):
        with pytest.raises(VMError):
            val("let x = [1,2,3]; x?.name;")


# ============================================================
# Optional index access: obj?.[expr]
# ============================================================

class TestOptionalIndex:
    def test_null_index_returns_null(self):
        assert val("let x = null; x?.[0];") is None

    def test_nonnull_array_index(self):
        assert val("let x = [10, 20, 30]; x?.[1];") == 20

    def test_nonnull_hash_index(self):
        assert val('let x = {a: 1}; x?.["a"];') == 1

    def test_nonnull_string_index(self):
        assert val('let x = "hello"; x?.[0];') == "h"

    def test_chain_null_then_index(self):
        assert val("let x = null; x?.[0]?.[1];") is None

    def test_chain_nonnull_index(self):
        assert val("let x = [[1, 2], [3, 4]]; x?.[0]?.[1];") == 2


# ============================================================
# Optional call: obj?.(args)
# ============================================================

class TestOptionalCall:
    def test_null_call_returns_null(self):
        assert val("let x = null; x?.(1, 2);") is None

    def test_null_call_no_args(self):
        assert val("let x = null; x?.();") is None

    def test_nonnull_call(self):
        assert val("let add = fn(a, b) { return a + b; }; add?.(3, 4);") == 7

    def test_nonnull_call_no_args(self):
        assert val("let f = fn() { return 42; }; f?.();") == 42

    def test_chain_optional_call_on_null(self):
        assert val("let x = null; x?.()?.name;") is None


# ============================================================
# Mixed optional chaining
# ============================================================

class TestMixedChaining:
    def test_dot_then_index(self):
        assert val('let x = {items: [10, 20]}; x?.items?.[1];') == 20

    def test_dot_then_call(self):
        code = """
        let obj = {greet: fn(name) { return "hello " + name; }};
        obj?.greet?.("world");
        """
        assert val(code) == "hello world"

    def test_null_dot_then_call(self):
        assert val("let x = null; x?.greet?.();") is None

    def test_nonnull_dot_null_method(self):
        assert val('let x = {greet: null}; x?.greet?.();') is None

    def test_index_then_dot(self):
        code = """
        let arr = [{name: "alice"}, {name: "bob"}];
        arr?.[0]?.name;
        """
        assert val(code) == "alice"

    def test_deep_chain(self):
        code = """
        let x = {a: {b: {c: {d: 42}}}};
        x?.a?.b?.c?.d;
        """
        assert val(code) == 42

    def test_deep_chain_null_at_start(self):
        assert val("let x = null; x?.a?.b?.c?.d;") is None

    def test_deep_chain_null_in_middle(self):
        code = """
        let x = {a: {b: null}};
        x?.a?.b?.c?.d;
        """
        assert val(code) is None


# ============================================================
# Optional chaining with classes
# ============================================================

class TestOptionalChainClasses:
    def test_null_instance_dot(self):
        code = """
        class Dog { init(name) { this.name = name; } }
        let d = null;
        d?.name;
        """
        assert val(code) is None

    def test_nonnull_instance_dot(self):
        code = """
        class Dog { init(name) { this.name = name; } }
        let d = Dog("Rex");
        d?.name;
        """
        assert val(code) == "Rex"

    def test_null_method_call(self):
        code = """
        class Dog {
            init(name) { this.name = name; }
            bark() { return "woof"; }
        }
        let d = null;
        d?.bark?.();
        """
        assert val(code) is None

    def test_nonnull_method_call(self):
        code = """
        class Dog {
            init(name) { this.name = name; }
            bark() { return "woof"; }
        }
        let d = Dog("Rex");
        d?.bark();
        """
        assert val(code) == "woof"

    def test_optional_on_method_result(self):
        code = """
        class Box {
            init(v) { this.v = v; }
            get() { return this.v; }
        }
        let b = Box(null);
        b?.get()?.name;
        """
        assert val(code) is None

    def test_class_chain_nonnull(self):
        code = """
        class Node {
            init(val, nxt) {
                this.val = val;
                this.nxt = nxt;
            }
        }
        let list = Node(1, Node(2, Node(3, null)));
        list?.nxt?.nxt?.val;
        """
        assert val(code) == 3

    def test_class_chain_end_null(self):
        code = """
        class Node {
            init(val, nxt) {
                this.val = val;
                this.nxt = nxt;
            }
        }
        let list = Node(1, Node(2, null));
        list?.nxt?.nxt?.val;
        """
        assert val(code) is None

    def test_optional_with_super(self):
        code = """
        class Base {
            greet() { return "hello"; }
        }
        class Child < Base {
            greet() { return super.greet() + " world"; }
        }
        let c = Child();
        c?.greet();
        """
        assert val(code) == "hello world"

    def test_instanceof_with_optional(self):
        code = """
        class A {}
        let a = A();
        let b = null;
        [instanceof(a, A), b?.name];
        """
        assert val(code) == [True, None]


# ============================================================
# Assignment to optional chain is an error
# ============================================================

class TestOptionalAssignError:
    def test_cannot_assign_optional_dot(self):
        with pytest.raises(ParseError, match="Cannot assign to optional chain"):
            val('let x = {}; x?.name = 5;')

    def test_cannot_assign_optional_index(self):
        with pytest.raises(ParseError, match="Cannot assign to optional chain"):
            val('let x = []; x?.[0] = 5;')


# ============================================================
# Optional chaining with print
# ============================================================

class TestOptionalChainPrint:
    def test_print_null_chain(self):
        assert out("let x = null; print x?.name;") == "null"

    def test_print_nonnull_chain(self):
        assert out('let x = {name: "alice"}; print x?.name;') == "alice"

    def test_print_null_literal(self):
        assert out("print null;") == "null"


# ============================================================
# Integration with other features
# ============================================================

class TestIntegration:
    def test_optional_in_if(self):
        assert val("let x = null; if (x?.active) { 1; } else { 0; }") == 0

    def test_optional_in_if_nonnull(self):
        assert val("let x = {active: true}; if (x?.active) { 1; } else { 0; }") == 1

    def test_optional_with_pipe(self):
        code = """
        let double = fn(x) { return x * 2; };
        let obj = {val: 5};
        obj?.val |> double;
        """
        assert val(code) == 10

    def test_optional_with_spread(self):
        assert val('let x = {a: 1, b: 2}; let y = {...x, c: 3}; y?.a;') == 1

    def test_optional_in_string_interpolation(self):
        assert val('let x = null; f"value: ${x?.name}";') == "value: null"

    def test_optional_in_array_literal(self):
        assert val("let x = null; [x?.a, x?.b, 3];") == [None, None, 3]

    def test_optional_in_hash_literal(self):
        assert val("let x = null; {val: x?.name};") == {"val": None}

    def test_optional_in_function_arg(self):
        code = """
        let f = fn(x) { return x; };
        let obj = null;
        f(obj?.name);
        """
        assert val(code) is None

    def test_optional_with_closures(self):
        code = """
        let make = fn() {
            let data = null;
            return fn() { return data?.value; };
        };
        let get = make();
        get();
        """
        assert val(code) is None

    def test_optional_with_error_handling(self):
        code = """
        let x = null;
        try { x?.name; } catch(e) { "error"; }
        """
        assert val(code) is None

    def test_optional_call_with_spread_args(self):
        code = """
        let f = fn(a, b, c) { return a + b + c; };
        let args = [1, 2, 3];
        f?.(...args);
        """
        assert val(code) == 6

    def test_null_optional_call_with_spread(self):
        code = """
        let f = null;
        let args = [1, 2, 3];
        f?.(...args);
        """
        assert val(code) is None

    def test_optional_with_destructuring(self):
        code = """
        let x = {items: [1, 2, 3]};
        let arr = x?.items;
        let [a, b, c] = arr;
        a + b + c;
        """
        assert val(code) == 6

    def test_optional_with_for_in(self):
        code = """
        let x = {items: [10, 20, 30]};
        let sum = 0;
        for (item in x?.items) { sum = sum + item; }
        sum;
        """
        assert val(code) == 60

    def test_optional_with_generators(self):
        code = """
        let gen = fn*() { yield 1; yield 2; };
        let g = gen();
        let r = next(g);
        r;
        """
        assert val(code) == 1

    def test_null_generator_optional(self):
        code = """
        let g = null;
        g?.next;
        """
        assert val(code) is None

    def test_optional_with_while(self):
        code = """
        let data = {count: 3};
        let i = 0;
        while (i < data?.count) { i = i + 1; }
        i;
        """
        assert val(code) == 3


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_false_is_not_null(self):
        with pytest.raises(VMError):
            val("let x = false; x?.name;")

    def test_zero_is_not_null(self):
        with pytest.raises(VMError):
            val("let x = 0; x?.name;")

    def test_empty_string_is_not_null(self):
        with pytest.raises(VMError):
            val('let x = ""; x?.name;')

    def test_empty_array_index_errors(self):
        with pytest.raises(VMError):
            val("let x = []; x?.[0];")

    def test_empty_hash_key_errors(self):
        with pytest.raises(VMError):
            val('let x = {}; x?.["missing"];')

    def test_optional_preserves_null(self):
        assert val("let x = null; x?.prop;") is None

    def test_optional_dot_after_call(self):
        code = """
        let f = fn() { return null; };
        f()?.name;
        """
        assert val(code) is None

    def test_optional_dot_after_nonnull_call(self):
        code = """
        let f = fn() { return {name: "test"}; };
        f()?.name;
        """
        assert val(code) == "test"

    def test_nested_optional_calls(self):
        code = """
        let outer = fn() { return fn() { return 42; }; };
        outer?.()?.();
        """
        assert val(code) == 42

    def test_nested_optional_calls_null_inner(self):
        code = """
        let outer = fn() { return null; };
        outer?.()?.();
        """
        assert val(code) is None

    def test_optional_chain_in_let(self):
        assert val("let x = null; let y = x?.name; y;") is None

    def test_optional_chain_result_arithmetic_error(self):
        with pytest.raises((VMError, TypeError)):
            val("let x = null; x?.val + 1;")

    def test_optional_chaining_long_chain(self):
        code = """
        let x = {a: {b: {c: {d: {e: {f: 99}}}}}};
        x?.a?.b?.c?.d?.e?.f;
        """
        assert val(code) == 99

    def test_null_not_equal_to_false(self):
        assert val("null == false;") is False

    def test_null_not_equal_to_zero(self):
        assert val("null == 0;") is False

    def test_null_not_equal_to_empty_string(self):
        assert val('null == "";') is False

    def test_null_equal_to_null(self):
        assert val("null == null;") is True


# ============================================================
# Regression: all C052 features still work
# ============================================================

class TestC052Regression:
    def test_class_basic(self):
        code = """
        class Dog {
            init(name) { this.name = name; }
            bark() { return "woof " + this.name; }
        }
        let d = Dog("Rex");
        d.bark();
        """
        assert val(code) == "woof Rex"

    def test_class_inheritance(self):
        code = """
        class Animal {
            init(name) { this.name = name; }
            speak() { return this.name + " speaks"; }
        }
        class Dog < Animal {
            bark() { return this.name + " barks"; }
        }
        let d = Dog("Rex");
        d.bark();
        """
        assert val(code) == "Rex barks"

    def test_super_call(self):
        code = """
        class Base {
            greet() { return "hello"; }
        }
        class Child < Base {
            greet() { return super.greet() + " world"; }
        }
        Child().greet();
        """
        assert val(code) == "hello world"

    def test_instanceof(self):
        code = """
        class A {}
        class B < A {}
        let b = B();
        instanceof(b, A);
        """
        assert val(code) is True

    def test_pipe_operator(self):
        assert val("let f = fn(x) { return x * 2; }; 5 |> f;") == 10

    def test_spread_operator(self):
        assert val("let a = [1, 2]; let b = [...a, 3]; b;") == [1, 2, 3]

    def test_destructuring(self):
        assert val("let [a, b] = [10, 20]; a + b;") == 30

    def test_string_interpolation(self):
        assert val('let x = 42; f"val=${x}";') == "val=42"

    def test_closures(self):
        code = """
        let make = fn(x) { return fn() { return x; }; };
        let f = make(99);
        f();
        """
        assert val(code) == 99

    def test_error_handling(self):
        assert val('try { throw "oops"; } catch(e) { e; }') == "oops"

    def test_for_in(self):
        code = """
        let sum = 0;
        for (x in [1, 2, 3]) { sum = sum + x; }
        sum;
        """
        assert val(code) == 6

    def test_generators(self):
        code = """
        let gen = fn*() { yield 1; yield 2; yield 3; };
        let g = gen();
        let a = next(g);
        let b = next(g);
        a + b;
        """
        assert val(code) == 3

    def test_modules(self):
        from optional_chaining import ModuleRegistry
        registry = ModuleRegistry()
        registry.register("math", "export let pi = 3; export let double = fn(x) { return x * 2; };")
        result, output = run('import {pi, double} from "math"; double(pi);', registry=registry)
        assert result == 6

    def test_hash_maps(self):
        assert val('{a: 1, b: 2}.a;') == 1

    def test_arrays(self):
        assert val("[10, 20, 30][1];") == 20

    def test_while_loop(self):
        code = """
        let i = 0;
        while (i < 5) { i = i + 1; }
        i;
        """
        assert val(code) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
