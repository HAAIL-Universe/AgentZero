"""
Tests for C049: String Interpolation
Challenge C049 -- AgentZero Session 050
"""

import pytest
from string_interpolation import (
    lex, parse, run, execute, TokenType,
    LexError, ParseError, CompileError, VMError,
    InterpolatedString, StringLit,
)


# ============================================================
# Section 1: Lexer -- f-string tokenization
# ============================================================

class TestLexer:
    def test_simple_fstring_token(self):
        tokens = lex('f"hello ${name}"')
        assert tokens[0].type == TokenType.FSTRING
        parts = tokens[0].value
        assert len(parts) == 2
        assert parts[0] == ("text", "hello ")
        assert parts[1] == ("expr", "name")

    def test_fstring_text_only(self):
        tokens = lex('f"just text"')
        assert tokens[0].type == TokenType.FSTRING
        assert tokens[0].value == [("text", "just text")]

    def test_fstring_expr_only(self):
        tokens = lex('f"${x}"')
        assert tokens[0].type == TokenType.FSTRING
        assert tokens[0].value == [("expr", "x")]

    def test_fstring_multiple_exprs(self):
        tokens = lex('f"${a} and ${b}"')
        parts = tokens[0].value
        assert len(parts) == 3
        assert parts[0] == ("expr", "a")
        assert parts[1] == ("text", " and ")
        assert parts[2] == ("expr", "b")

    def test_fstring_escaped_dollar(self):
        tokens = lex('f"price: \\${99}"')
        parts = tokens[0].value
        assert len(parts) == 1
        assert parts[0] == ("text", "price: $99}")

    def test_fstring_escape_sequences(self):
        tokens = lex('f"line1\\nline2"')
        parts = tokens[0].value
        assert parts[0] == ("text", "line1\nline2")

    def test_fstring_with_expression(self):
        tokens = lex('f"${a + b}"')
        parts = tokens[0].value
        assert parts[0] == ("expr", "a + b")

    def test_fstring_nested_braces(self):
        tokens = lex('f"${fn({x: 1})}"')
        parts = tokens[0].value
        assert parts[0][0] == "expr"
        assert "{x: 1}" in parts[0][1]

    def test_fstring_empty_interpolation_error(self):
        with pytest.raises(LexError, match="Empty interpolation"):
            lex('f"${}"')

    def test_fstring_unterminated_error(self):
        with pytest.raises(LexError, match="Unterminated f-string"):
            lex('f"no closing quote')

    def test_fstring_unterminated_interpolation(self):
        with pytest.raises(LexError, match="Unterminated interpolation"):
            lex('f"${x"')

    def test_plain_f_identifier(self):
        """'f' not followed by quote is a normal identifier."""
        tokens = lex('f + 1')
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == 'f'

    def test_fstring_empty(self):
        tokens = lex('f""')
        assert tokens[0].type == TokenType.FSTRING
        assert tokens[0].value == []

    def test_fstring_nested_string_in_expr(self):
        tokens = lex('f"${concat("a", "b")}"')
        parts = tokens[0].value
        assert parts[0][0] == "expr"
        # The expression should contain the nested strings
        assert '"a"' in parts[0][1]


# ============================================================
# Section 2: Parser -- InterpolatedString AST
# ============================================================

class TestParser:
    def test_parse_fstring_to_ast(self):
        prog = parse('f"hello ${name}";')
        stmt = prog.stmts[0]
        assert isinstance(stmt, InterpolatedString)
        assert len(stmt.parts) == 2
        assert isinstance(stmt.parts[0], StringLit)
        assert stmt.parts[0].value == "hello "

    def test_parse_fstring_text_only_optimized(self):
        """F-string with only text becomes a plain StringLit."""
        prog = parse('f"just text";')
        stmt = prog.stmts[0]
        assert isinstance(stmt, StringLit)
        assert stmt.value == "just text"

    def test_parse_fstring_empty_optimized(self):
        prog = parse('f"";')
        stmt = prog.stmts[0]
        assert isinstance(stmt, StringLit)
        assert stmt.value == ""

    def test_parse_fstring_in_let(self):
        prog = parse('let msg = f"hi ${name}";')
        assert prog.stmts[0].value.__class__.__name__ == "InterpolatedString"

    def test_parse_fstring_in_print(self):
        prog = parse('print f"value: ${x}";')
        assert prog.stmts[0].__class__.__name__ == "PrintStmt"


# ============================================================
# Section 3: Basic interpolation
# ============================================================

class TestBasicInterpolation:
    def test_simple_variable(self):
        result, output = run('let name = "world"; print f"hello ${name}";')
        assert output == ["hello world"]

    def test_number_interpolation(self):
        result, output = run('let x = 42; print f"value: ${x}";')
        assert output == ["value: 42"]

    def test_float_interpolation(self):
        result, output = run('let x = 3.14; print f"pi ~ ${x}";')
        assert output == ["pi ~ 3.14"]

    def test_bool_interpolation(self):
        result, output = run('let x = true; print f"flag: ${x}";')
        assert output == ["flag: true"]

    def test_none_interpolation(self):
        result, output = run('''
            fn nothing() { return; }
            print f"got: ${nothing()}";
        ''')
        assert output == ["got: none"]

    def test_text_only_fstring(self):
        result, output = run('print f"just text";')
        assert output == ["just text"]

    def test_empty_fstring(self):
        result, output = run('print f"";')
        assert output == [""]

    def test_expression_only_fstring(self):
        result, output = run('let x = 5; print f"${x}";')
        assert output == ["5"]


# ============================================================
# Section 4: Expression interpolation
# ============================================================

class TestExpressionInterpolation:
    def test_arithmetic(self):
        result, output = run('print f"${2 + 3}";')
        assert output == ["5"]

    def test_complex_expression(self):
        result, output = run('let x = 10; let y = 20; print f"sum: ${x + y}";')
        assert output == ["sum: 30"]

    def test_comparison(self):
        result, output = run('let x = 5; print f"big? ${x > 3}";')
        assert output == ["big? true"]

    def test_function_call(self):
        result, output = run('''
            fn double(x) { return x * 2; }
            print f"${double(21)}";
        ''')
        assert output == ["42"]

    def test_builtin_call(self):
        result, output = run('let arr = [1, 2, 3]; print f"len: ${len(arr)}";')
        assert output == ["len: 3"]

    def test_index_access(self):
        result, output = run('let arr = [10, 20, 30]; print f"first: ${arr[0]}";')
        assert output == ["first: 10"]

    def test_dot_access(self):
        result, output = run('let obj = {name: "Alice"}; print f"name: ${obj.name}";')
        assert output == ["name: Alice"]

    def test_string_concat_in_expr(self):
        result, output = run('let a = "hello"; print f"${a + " world"}";')
        assert output == ["hello world"]

    def test_nested_function_call(self):
        result, output = run('''
            fn greet(name) { return "hi " + name; }
            print f"${greet("Bob")}";
        ''')
        assert output == ["hi Bob"]


# ============================================================
# Section 5: Multiple interpolations
# ============================================================

class TestMultipleInterpolations:
    def test_two_vars(self):
        result, output = run('''
            let first = "Jane";
            let last = "Doe";
            print f"${first} ${last}";
        ''')
        assert output == ["Jane Doe"]

    def test_three_interpolations(self):
        result, output = run('''
            let a = 1; let b = 2; let c = 3;
            print f"${a}, ${b}, ${c}";
        ''')
        assert output == ["1, 2, 3"]

    def test_mixed_text_and_exprs(self):
        result, output = run('''
            let name = "Alice";
            let age = 30;
            print f"Name: ${name}, Age: ${age} years";
        ''')
        assert output == ["Name: Alice, Age: 30 years"]

    def test_adjacent_interpolations(self):
        result, output = run('let a = "x"; let b = "y"; print f"${a}${b}";')
        assert output == ["xy"]


# ============================================================
# Section 6: Escaped dollar sign
# ============================================================

class TestEscaping:
    def test_escaped_dollar(self):
        result, output = run('print f"price: \\${99}";')
        assert output == ["price: $99}"]

    def test_escape_newline(self):
        result, output = run('print f"line1\\nline2";')
        assert output == ["line1\nline2"]

    def test_escape_tab(self):
        result, output = run('print f"col1\\tcol2";')
        assert output == ["col1\tcol2"]

    def test_escape_backslash(self):
        result, output = run('print f"path: C:\\\\dir";')
        assert output == ["path: C:\\dir"]

    def test_escape_quote(self):
        result, output = run('print f"she said \\"hi\\"";')
        assert output == ['she said "hi"']


# ============================================================
# Section 7: F-string as expression (not just print)
# ============================================================

class TestFstringAsExpression:
    def test_assign_to_variable(self):
        result, output = run('''
            let name = "world";
            let msg = f"hello ${name}";
            print msg;
        ''')
        assert output == ["hello world"]

    def test_return_from_function(self):
        result, output = run('''
            fn greet(name) { return f"hello ${name}"; }
            print greet("Alice");
        ''')
        assert output == ["hello Alice"]

    def test_pass_as_argument(self):
        result, output = run('''
            fn upper_first(s) { return s; }
            let name = "Bob";
            print upper_first(f"hi ${name}");
        ''')
        assert output == ["hi Bob"]

    def test_in_array_literal(self):
        result, output = run('''
            let x = 1;
            let arr = [f"a${x}", f"b${x + 1}"];
            print arr[0];
            print arr[1];
        ''')
        assert output == ["a1", "b2"]

    def test_in_hash_literal(self):
        result, output = run('''
            let n = "test";
            let h = {msg: f"hello ${n}"};
            print h.msg;
        ''')
        assert output == ["hello test"]

    def test_concatenate_fstrings(self):
        result, output = run('''
            let a = "x";
            let b = "y";
            print f"${a}" + f"${b}";
        ''')
        assert output == ["xy"]

    def test_fstring_comparison(self):
        result, output = run('''
            let name = "Alice";
            print f"hello ${name}" == "hello Alice";
        ''')
        assert output == ["true"]


# ============================================================
# Section 8: Coercion edge cases
# ============================================================

class TestCoercion:
    def test_array_in_interpolation(self):
        result, output = run('let arr = [1, 2, 3]; print f"arr: ${arr}";')
        assert output == ["arr: [1, 2, 3]"]

    def test_hash_in_interpolation(self):
        result, output = run('let h = {a: 1}; print f"h: ${h}";')
        # _format_value for hashes produces something like {a: 1}
        assert "a" in output[0]
        assert "1" in output[0]

    def test_negative_number(self):
        result, output = run('let x = 0 - 5; print f"neg: ${x}";')
        assert output == ["neg: -5"]

    def test_zero(self):
        result, output = run('print f"${0}";')
        assert output == ["0"]

    def test_empty_string_var(self):
        result, output = run('let s = ""; print f"[${s}]";')
        assert output == ["[]"]


# ============================================================
# Section 9: Closures and generators with f-strings
# ============================================================

class TestClosuresAndGenerators:
    def test_closure_captures_in_fstring(self):
        result, output = run('''
            fn make_greeter(greeting) {
                return fn(name) {
                    return f"${greeting}, ${name}!";
                };
            }
            let hi = make_greeter("Hi");
            print hi("Alice");
        ''')
        assert output == ["Hi, Alice!"]

    def test_fstring_in_map(self):
        result, output = run('''
            let nums = [1, 2, 3];
            let strs = map(nums, fn(n) { return f"item ${n}"; });
            print strs[0];
            print strs[1];
            print strs[2];
        ''')
        assert output == ["item 1", "item 2", "item 3"]

    def test_fstring_in_filter_callback(self):
        result, output = run('''
            let items = ["a", "bb", "ccc"];
            let long = filter(items, fn(s) { return len(s) > 1; });
            print f"found ${len(long)} long items";
        ''')
        assert output == ["found 2 long items"]

    def test_generator_value_in_fstring(self):
        result, output = run('''
            fn* counter() {
                yield 1;
                yield 2;
            }
            let g = counter();
            print f"first: ${next(g)}";
            print f"second: ${next(g)}";
        ''')
        assert output == ["first: 1", "second: 2"]


# ============================================================
# Section 10: Error handling with f-strings
# ============================================================

class TestErrorHandling:
    def test_fstring_in_try_catch(self):
        result, output = run('''
            try {
                throw f"error: ${42}";
            } catch (e) {
                print e;
            }
        ''')
        assert output == ["error: 42"]

    def test_fstring_with_error_message(self):
        result, output = run('''
            fn safe_div(a, b) {
                if (b == 0) {
                    throw f"cannot divide ${a} by zero";
                }
                return a / b;
            }
            try {
                safe_div(10, 0);
            } catch (e) {
                print e;
            }
        ''')
        assert output == ["cannot divide 10 by zero"]


# ============================================================
# Section 11: Module system with f-strings
# ============================================================

class TestModuleSystem:
    def test_fstring_in_module(self):
        from string_interpolation import ModuleRegistry
        registry = ModuleRegistry()
        registry.register("greet", '''
            export fn hello(name) {
                return f"hello ${name}";
            }
        ''')
        result, output = run('''
            import { hello } from "greet";
            print hello("world");
        ''', registry=registry)
        assert output == ["hello world"]


# ============================================================
# Section 12: Destructuring with f-strings
# ============================================================

class TestDestructuringWithFstrings:
    def test_destructured_var_in_fstring(self):
        result, output = run('''
            let [a, b] = [1, 2];
            print f"${a} and ${b}";
        ''')
        assert output == ["1 and 2"]

    def test_hash_destructure_in_fstring(self):
        result, output = run('''
            let {name, age} = {name: "Alice", age: 30};
            print f"${name} is ${age}";
        ''')
        assert output == ["Alice is 30"]

    def test_for_in_destructure_with_fstring(self):
        result, output = run('''
            let pairs = [[1, "a"], [2, "b"]];
            for ([n, s] in pairs) {
                print f"${n}: ${s}";
            }
        ''')
        assert output == ["1: a", "2: b"]


# ============================================================
# Section 13: Nested f-strings
# ============================================================

class TestNestedFstrings:
    def test_nested_fstring(self):
        result, output = run('''
            let x = 5;
            let label = "val";
            print f"outer: ${f"${label}=${x}"}";
        ''')
        assert output == ["outer: val=5"]

    def test_deeply_nested(self):
        result, output = run('''
            let a = 1;
            print f"${f"${f"${a}"}"}";
        ''')
        assert output == ["1"]


# ============================================================
# Section 14: Pattern matching with f-strings
# ============================================================

class TestPatternMatching:
    def test_fstring_with_ternary_like_pattern(self):
        result, output = run('''
            let x = 10;
            let label = if (x > 5) "big" else "small";
            print f"${x} is ${label}";
        ''')
        assert output == ["10 is big"]

    def test_fstring_in_while_loop(self):
        result, output = run('''
            let i = 0;
            while (i < 3) {
                print f"step ${i}";
                i = i + 1;
            }
        ''')
        assert output == ["step 0", "step 1", "step 2"]

    def test_fstring_in_for_in(self):
        result, output = run('''
            for (x in [10, 20, 30]) {
                print f"val: ${x}";
            }
        ''')
        assert output == ["val: 10", "val: 20", "val: 30"]


# ============================================================
# Section 15: Complex real-world-like usage
# ============================================================

class TestRealWorldUsage:
    def test_table_formatting(self):
        result, output = run('''
            let items = [{name: "apple", price: 1}, {name: "banana", price: 2}];
            for (item in items) {
                print f"${item.name}: $${item.price}";
            }
        ''')
        assert output == ["apple: $1", "banana: $2"]

    def test_debug_logging(self):
        result, output = run('''
            fn debug(label, val) {
                print f"[DEBUG] ${label} = ${val}";
            }
            debug("count", 42);
            debug("name", "test");
        ''')
        assert output == ["[DEBUG] count = 42", "[DEBUG] name = test"]

    def test_string_builder_pattern(self):
        result, output = run('''
            let parts = ["hello", "world", "!"];
            let result = "";
            for (p in parts) {
                result = f"${result}${p} ";
            }
            print result;
        ''')
        assert output == ["hello world ! "]

    def test_template_function(self):
        result, output = run('''
            fn card(name, title) {
                return f"[${title}] ${name}";
            }
            print card("Alice", "CEO");
            print card("Bob", "CTO");
        ''')
        assert output == ["[CEO] Alice", "[CTO] Bob"]

    def test_recursive_fstring(self):
        result, output = run('''
            fn repeat(s, n) {
                if (n <= 0) { return ""; }
                return f"${s}${repeat(s, n - 1)}";
            }
            print repeat("ab", 3);
        ''')
        assert output == ["ababab"]

    def test_fstring_with_len_and_slice(self):
        result, output = run('''
            let arr = [1, 2, 3, 4, 5];
            let first3 = slice(arr, 0, 3);
            print f"total: ${len(arr)}, showing: ${len(first3)}";
        ''')
        assert output == ["total: 5, showing: 3"]

    def test_error_report_template(self):
        result, output = run('''
            fn error_report(code, msg) {
                return f"Error ${code}: ${msg}";
            }
            try {
                throw error_report(404, "not found");
            } catch (e) {
                print f"Caught: ${e}";
            }
        ''')
        assert output == ["Caught: Error 404: not found"]

    def test_fstring_with_reduce(self):
        result, output = run('''
            let nums = [1, 2, 3, 4, 5];
            let sum = reduce(nums, fn(acc, n) { return acc + n; }, 0);
            let avg = sum / len(nums);
            print f"sum=${sum}, avg=${avg}";
        ''')
        assert output == ["sum=15, avg=3"]

    def test_multiline_fstring_usage(self):
        result, output = run('''
            let name = "Alice";
            let items = [1, 2, 3];
            let msg = f"User: ${name}, Items: ${len(items)}, First: ${items[0]}";
            print msg;
        ''')
        assert output == ["User: Alice, Items: 3, First: 1"]

    def test_fstring_type_coercion_all_types(self):
        result, output = run('''
            let i = 42;
            let f = 3.14;
            let b = true;
            let s = "text";
            let a = [1, 2];
            let h = {x: 1};
            print f"${i}";
            print f"${f}";
            print f"${b}";
            print f"${s}";
            print f"${a}";
            print f"${h}";
        ''')
        assert output[0] == "42"
        assert output[1] == "3.14"
        assert output[2] == "true"
        assert output[3] == "text"
        assert output[4] == "[1, 2]"
        assert "x" in output[5]


# ============================================================
# Section 16: Edge cases
# ============================================================

class TestEdgeCases:
    def test_fstring_with_only_text(self):
        result, output = run('print f"no interpolation here";')
        assert output == ["no interpolation here"]

    def test_fstring_dollar_not_followed_by_brace(self):
        """Dollar sign without { is treated as literal text."""
        tokens = lex('f"cost: $5"')
        parts = tokens[0].value
        # $ not followed by { should be literal
        assert parts[0] == ("text", "cost: $5")

    def test_fstring_result_is_string(self):
        result, output = run('''
            let x = 42;
            let s = f"${x}";
            print type(s);
        ''')
        assert output == ["string"]

    def test_plain_string_unchanged(self):
        """Regular strings should still work exactly as before."""
        result, output = run('print "hello ${world}";')
        assert output == ["hello ${world}"]

    def test_fstring_single_text_optimization(self):
        """Single text part is optimized to StringLit."""
        prog = parse('f"hello";')
        assert isinstance(prog.stmts[0], StringLit)

    def test_f_as_variable_name(self):
        """'f' should still work as a variable name."""
        result, output = run('let f = 42; print f;')
        assert output == ["42"]

    def test_fstring_with_semicolons_in_expr(self):
        """Expression in interpolation should not include semicolons."""
        # This should work: the expression is just "x"
        result, output = run('let x = 5; print f"${x}";')
        assert output == ["5"]

    def test_fstring_with_hash_literal_in_expr(self):
        """Hash literal inside interpolation (nested braces)."""
        result, output = run('''
            fn first_key(h) { return keys(h)[0]; }
            print f"${first_key({a: 1})}";
        ''')
        assert output == ["a"]


# ============================================================
# Section 17: Backward compatibility
# ============================================================

class TestBackwardCompatibility:
    def test_regular_strings_work(self):
        result, output = run('print "hello world";')
        assert output == ["hello world"]

    def test_string_escapes_work(self):
        result, output = run('print "a\\nb";')
        assert output == ["a\nb"]

    def test_string_concat_works(self):
        result, output = run('print "hello" + " " + "world";')
        assert output == ["hello world"]

    def test_destructuring_still_works(self):
        result, output = run('''
            let [a, b, ...rest] = [1, 2, 3, 4];
            print a;
            print b;
            print rest;
        ''')
        assert output == ["1", "2", "[3, 4]"]

    def test_generators_still_work(self):
        result, output = run('''
            fn* range_gen(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            let g = range_gen(3);
            print next(g);
            print next(g);
            print next(g);
        ''')
        assert output == ["0", "1", "2"]

    def test_modules_still_work(self):
        from string_interpolation import ModuleRegistry
        registry = ModuleRegistry()
        registry.register("math_mod", '''
            export fn add(a, b) { return a + b; }
        ''')
        result, output = run('''
            import { add } from "math_mod";
            print add(3, 4);
        ''', registry=registry)
        assert output == ["7"]

    def test_error_handling_still_works(self):
        result, output = run('''
            try {
                throw "oops";
            } catch (e) {
                print e;
            }
        ''')
        assert output == ["oops"]

    def test_for_in_still_works(self):
        result, output = run('''
            for (x in [10, 20, 30]) {
                print x;
            }
        ''')
        assert output == ["10", "20", "30"]

    def test_closures_still_work(self):
        result, output = run('''
            fn make_adder(n) {
                return fn(x) { return x + n; };
            }
            let add5 = make_adder(5);
            print add5(10);
        ''')
        assert output == ["15"]

    def test_hash_maps_still_work(self):
        result, output = run('''
            let h = {a: 1, b: 2};
            print h.a;
            print keys(h);
        ''')
        assert output[0] == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
