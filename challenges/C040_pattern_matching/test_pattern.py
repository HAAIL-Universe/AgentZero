"""
Tests for C040 Pattern Matching
Challenge C040 -- AgentZero Session 041
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from pattern_matching import (
    run, execute, parse, compile_source, lex_extended,
    MatchExpr, MatchArm, LitPattern, WildcardPattern, VarPattern,
    TuplePattern, OrPattern, TupleLit, TupleValue,
    VMError, ParseError, LexError,
    ExtTokenType, TokenType,
)


# ============================================================
# Lexer Tests
# ============================================================

class TestLexer:
    def test_match_keyword(self):
        tokens = lex_extended("match")
        assert tokens[0].type == ExtTokenType.MATCH

    def test_fat_arrow(self):
        tokens = lex_extended("=>")
        assert tokens[0].type == ExtTokenType.FAT_ARROW

    def test_underscore(self):
        tokens = lex_extended("_")
        assert tokens[0].type == ExtTokenType.UNDERSCORE

    def test_pipe(self):
        tokens = lex_extended("|")
        assert tokens[0].type == ExtTokenType.PIPE

    def test_underscore_prefix_is_ident(self):
        tokens = lex_extended("_foo")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "_foo"

    def test_match_in_context(self):
        tokens = lex_extended("match (x) { 1 => { } }")
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert ExtTokenType.MATCH in types
        assert ExtTokenType.FAT_ARROW in types

    def test_all_base_tokens_work(self):
        src = "let x = 1 + 2; if (x == 3) { print(x); }"
        tokens = lex_extended(src)
        assert tokens[-1].type == TokenType.EOF

    def test_negative_in_lex(self):
        tokens = lex_extended("-5")
        assert tokens[0].type == TokenType.MINUS
        assert tokens[1].type == TokenType.INT
        assert tokens[1].value == 5


# ============================================================
# Parser Tests
# ============================================================

class TestParser:
    def test_parse_simple_match(self):
        ast = parse("match (x) { 1 => { } }")
        stmt = ast.stmts[0]
        assert isinstance(stmt, MatchExpr)
        assert len(stmt.arms) == 1

    def test_parse_wildcard_pattern(self):
        ast = parse("match (x) { _ => { } }")
        arm = ast.stmts[0].arms[0]
        assert isinstance(arm.pattern, WildcardPattern)

    def test_parse_literal_patterns(self):
        ast = parse('match (x) { 1 => { } "hi" => { } true => { } }')
        arms = ast.stmts[0].arms
        assert isinstance(arms[0].pattern, LitPattern)
        assert arms[0].pattern.value == 1
        assert isinstance(arms[1].pattern, LitPattern)
        assert arms[1].pattern.value == "hi"
        assert isinstance(arms[2].pattern, LitPattern)
        assert arms[2].pattern.value == True

    def test_parse_variable_pattern(self):
        ast = parse("match (x) { y => { } }")
        arm = ast.stmts[0].arms[0]
        assert isinstance(arm.pattern, VarPattern)
        assert arm.pattern.name == "y"

    def test_parse_guard(self):
        ast = parse("match (x) { y if y > 0 => { } }")
        arm = ast.stmts[0].arms[0]
        assert isinstance(arm.pattern, VarPattern)
        assert arm.guard is not None

    def test_parse_or_pattern(self):
        ast = parse("match (x) { 1 | 2 | 3 => { } }")
        arm = ast.stmts[0].arms[0]
        assert isinstance(arm.pattern, OrPattern)
        assert len(arm.pattern.patterns) == 3

    def test_parse_tuple_pattern(self):
        ast = parse("match (x) { (a, b) => { } }")
        arm = ast.stmts[0].arms[0]
        assert isinstance(arm.pattern, TuplePattern)
        assert len(arm.pattern.elements) == 2

    def test_parse_negative_pattern(self):
        ast = parse("match (x) { -1 => { } }")
        arm = ast.stmts[0].arms[0]
        assert isinstance(arm.pattern, LitPattern)
        assert arm.pattern.value == -1

    def test_parse_tuple_literal(self):
        ast = parse("let t = (1, 2, 3);")
        decl = ast.stmts[0]
        assert isinstance(decl.value, TupleLit)
        assert len(decl.value.elements) == 3

    def test_parse_grouped_expr_not_tuple(self):
        ast = parse("let x = (1 + 2);")
        # Should be BinOp, not TupleLit
        assert not isinstance(ast.stmts[0].value, TupleLit)

    def test_parse_empty_tuple(self):
        ast = parse("let t = ();")
        assert isinstance(ast.stmts[0].value, TupleLit)
        assert len(ast.stmts[0].value.elements) == 0

    def test_parse_multiple_arms(self):
        src = """
        match (x) {
            1 => { }
            2 => { }
            _ => { }
        }
        """
        ast = parse(src)
        assert len(ast.stmts[0].arms) == 3

    def test_parse_nested_tuple_pattern(self):
        ast = parse("match (x) { ((a, b), c) => { } }")
        arm = ast.stmts[0].arms[0]
        assert isinstance(arm.pattern, TuplePattern)
        inner = arm.pattern.elements[0]
        assert isinstance(inner, TuplePattern)

    def test_parse_expression_body(self):
        ast = parse("match (x) { 1 => 42 }")
        arm = ast.stmts[0].arms[0]
        # Body should be an expression, not a block
        assert not isinstance(arm.body, type(None))


# ============================================================
# Execution Tests -- Literal Patterns
# ============================================================

class TestLiteralPatterns:
    def test_match_int(self):
        _, out = run("""
            let x = 2;
            match (x) {
                1 => { print("one"); }
                2 => { print("two"); }
                3 => { print("three"); }
            }
        """)
        assert out == ["two"]

    def test_match_first_arm(self):
        _, out = run("""
            let x = 1;
            match (x) {
                1 => { print("yes"); }
                2 => { print("no"); }
            }
        """)
        assert out == ["yes"]

    def test_match_last_arm(self):
        _, out = run("""
            let x = 3;
            match (x) {
                1 => { print("one"); }
                2 => { print("two"); }
                3 => { print("three"); }
            }
        """)
        assert out == ["three"]

    def test_match_string(self):
        _, out = run("""
            let s = "hello";
            match (s) {
                "hi" => { print("1"); }
                "hello" => { print("2"); }
            }
        """)
        assert out == ["2"]

    def test_match_bool(self):
        _, out = run("""
            let b = true;
            match (b) {
                false => { print("f"); }
                true => { print("t"); }
            }
        """)
        assert out == ["t"]

    def test_match_float(self):
        _, out = run("""
            let f = 3.14;
            match (f) {
                2.71 => { print("e"); }
                3.14 => { print("pi"); }
            }
        """)
        assert out == ["pi"]

    def test_match_negative_int(self):
        _, out = run("""
            let x = -1;
            match (x) {
                -1 => { print("neg"); }
                1 => { print("pos"); }
            }
        """)
        assert out == ["neg"]

    def test_match_zero(self):
        _, out = run("""
            let x = 0;
            match (x) {
                0 => { print("zero"); }
                _ => { print("other"); }
            }
        """)
        assert out == ["zero"]

    def test_no_match_raises(self):
        with pytest.raises(VMError, match="no pattern matched"):
            run("""
                let x = 99;
                match (x) {
                    1 => { print("one"); }
                    2 => { print("two"); }
                }
            """)


# ============================================================
# Execution Tests -- Wildcard Pattern
# ============================================================

class TestWildcardPattern:
    def test_wildcard_matches_int(self):
        _, out = run("""
            let x = 42;
            match (x) {
                _ => { print("any"); }
            }
        """)
        assert out == ["any"]

    def test_wildcard_matches_string(self):
        _, out = run("""
            match ("hello") {
                _ => { print("matched"); }
            }
        """)
        assert out == ["matched"]

    def test_wildcard_as_default(self):
        _, out = run("""
            let x = 99;
            match (x) {
                1 => { print("one"); }
                _ => { print("default"); }
            }
        """)
        assert out == ["default"]

    def test_wildcard_not_reached_if_earlier_matches(self):
        _, out = run("""
            let x = 1;
            match (x) {
                1 => { print("one"); }
                _ => { print("default"); }
            }
        """)
        assert out == ["one"]


# ============================================================
# Execution Tests -- Variable Pattern
# ============================================================

class TestVariablePattern:
    def test_var_binding(self):
        _, out = run("""
            let x = 42;
            match (x) {
                y => { print(y); }
            }
        """)
        assert out == ["42"]

    def test_var_binding_with_expression(self):
        _, out = run("""
            match (10 + 20) {
                result => { print(result); }
            }
        """)
        assert out == ["30"]

    def test_var_binding_after_failed_literal(self):
        _, out = run("""
            let x = 5;
            match (x) {
                1 => { print("one"); }
                n => { print(n); }
            }
        """)
        assert out == ["5"]

    def test_var_binding_in_body(self):
        _, out = run("""
            let x = 7;
            match (x) {
                n => {
                    let doubled = n * 2;
                    print(doubled);
                }
            }
        """)
        assert out == ["14"]


# ============================================================
# Execution Tests -- Guard Patterns
# ============================================================

class TestGuardPatterns:
    def test_guard_passes(self):
        _, out = run("""
            let x = 15;
            match (x) {
                n if n > 10 => { print("big"); }
                _ => { print("small"); }
            }
        """)
        assert out == ["big"]

    def test_guard_fails(self):
        _, out = run("""
            let x = 3;
            match (x) {
                n if n > 10 => { print("big"); }
                _ => { print("small"); }
            }
        """)
        assert out == ["small"]

    def test_multiple_guards(self):
        _, out = run("""
            let x = 5;
            match (x) {
                n if n > 10 => { print("big"); }
                n if n > 0 => { print("positive"); }
                _ => { print("other"); }
            }
        """)
        assert out == ["positive"]

    def test_guard_with_literal_fallback(self):
        _, out = run("""
            let x = 0;
            match (x) {
                n if n > 0 => { print("pos"); }
                0 => { print("zero"); }
                _ => { print("neg"); }
            }
        """)
        assert out == ["zero"]

    def test_guard_complex_condition(self):
        _, out = run("""
            let x = 6;
            match (x) {
                n if n % 2 == 0 => { print("even"); }
                _ => { print("odd"); }
            }
        """)
        assert out == ["even"]

    def test_guard_uses_outer_vars(self):
        _, out = run("""
            let threshold = 10;
            let x = 15;
            match (x) {
                n if n > threshold => { print("above"); }
                _ => { print("below"); }
            }
        """)
        assert out == ["above"]


# ============================================================
# Execution Tests -- Or Patterns
# ============================================================

class TestOrPatterns:
    def test_or_first(self):
        _, out = run("""
            let x = 1;
            match (x) {
                1 | 2 | 3 => { print("small"); }
                _ => { print("other"); }
            }
        """)
        assert out == ["small"]

    def test_or_middle(self):
        _, out = run("""
            let x = 2;
            match (x) {
                1 | 2 | 3 => { print("small"); }
                _ => { print("other"); }
            }
        """)
        assert out == ["small"]

    def test_or_last(self):
        _, out = run("""
            let x = 3;
            match (x) {
                1 | 2 | 3 => { print("small"); }
                _ => { print("other"); }
            }
        """)
        assert out == ["small"]

    def test_or_no_match(self):
        _, out = run("""
            let x = 4;
            match (x) {
                1 | 2 | 3 => { print("small"); }
                _ => { print("other"); }
            }
        """)
        assert out == ["other"]

    def test_or_two_alternatives(self):
        _, out = run("""
            let x = 2;
            match (x) {
                1 | 2 => { print("yes"); }
                _ => { print("no"); }
            }
        """)
        assert out == ["yes"]

    def test_or_strings(self):
        _, out = run("""
            let s = "bye";
            match (s) {
                "hi" | "hello" => { print("greeting"); }
                "bye" | "cya" => { print("farewell"); }
                _ => { print("unknown"); }
            }
        """)
        assert out == ["farewell"]


# ============================================================
# Execution Tests -- Tuple Patterns
# ============================================================

class TestTuplePatterns:
    def test_tuple_literal(self):
        _, out = run("""
            let t = (1, 2, 3);
            print(t);
        """)
        assert out == ["(1, 2, 3)"]

    def test_tuple_match_literals(self):
        _, out = run("""
            let t = (1, 2);
            match (t) {
                (1, 2) => { print("yes"); }
                _ => { print("no"); }
            }
        """)
        assert out == ["yes"]

    def test_tuple_match_wrong_values(self):
        _, out = run("""
            let t = (1, 3);
            match (t) {
                (1, 2) => { print("yes"); }
                _ => { print("no"); }
            }
        """)
        assert out == ["no"]

    def test_tuple_match_wrong_length(self):
        _, out = run("""
            let t = (1, 2, 3);
            match (t) {
                (1, 2) => { print("two"); }
                _ => { print("other"); }
            }
        """)
        assert out == ["other"]

    def test_tuple_var_binding(self):
        _, out = run("""
            let t = (10, 20);
            match (t) {
                (a, b) => {
                    print(a);
                    print(b);
                }
            }
        """)
        assert out == ["10", "20"]

    def test_tuple_mixed_patterns(self):
        _, out = run("""
            let t = (1, 99);
            match (t) {
                (1, x) => { print(x); }
                _ => { print("no"); }
            }
        """)
        assert out == ["99"]

    def test_tuple_wildcard_element(self):
        _, out = run("""
            let t = (1, 2);
            match (t) {
                (1, _) => { print("first is 1"); }
                _ => { print("no"); }
            }
        """)
        assert out == ["first is 1"]

    def test_tuple_not_a_tuple(self):
        _, out = run("""
            let x = 42;
            match (x) {
                (a, b) => { print("tuple"); }
                _ => { print("not tuple"); }
            }
        """)
        assert out == ["not tuple"]

    def test_nested_tuple(self):
        _, out = run("""
            let t = ((1, 2), 3);
            match (t) {
                ((a, b), c) => {
                    print(a);
                    print(b);
                    print(c);
                }
            }
        """)
        assert out == ["1", "2", "3"]

    def test_empty_tuple_match(self):
        _, out = run("""
            let t = ();
            match (t) {
                () => { print("empty"); }
                _ => { print("not empty"); }
            }
        """)
        assert out == ["empty"]

    def test_tuple_three_elements(self):
        _, out = run("""
            let t = (1, 2, 3);
            match (t) {
                (a, b, c) => {
                    let sum = a + b + c;
                    print(sum);
                }
            }
        """)
        assert out == ["6"]


# ============================================================
# Execution Tests -- Match as Expression
# ============================================================

class TestMatchExpression:
    def test_match_expr_in_let(self):
        r = execute("""
            let x = 2;
            let result = match (x) {
                1 => 10
                2 => 20
                _ => 0
            };
            print(result);
        """)
        assert r['output'] == ["20"]

    def test_match_expr_in_print(self):
        _, out = run("""
            let x = 1;
            print(match (x) {
                1 => 100
                _ => 0
            });
        """)
        assert out == ["100"]

    def test_match_block_body_returns_none(self):
        r = execute("""
            let x = 1;
            let result = match (x) {
                1 => { print("one"); }
                _ => { print("other"); }
            };
        """)
        assert r['output'] == ["one"]
        assert r['env']['result'] is None


# ============================================================
# Execution Tests -- Functions and Match
# ============================================================

class TestMatchInFunctions:
    def test_match_in_function(self):
        _, out = run("""
            fn describe(x) {
                match (x) {
                    1 => { print("one"); }
                    2 => { print("two"); }
                    _ => { print("other"); }
                }
            }
            describe(1);
            describe(2);
            describe(99);
        """)
        assert out == ["one", "two", "other"]

    def test_match_with_function_call_scrutinee(self):
        _, out = run("""
            fn double(x) {
                return x * 2;
            }
            match (double(5)) {
                10 => { print("ten"); }
                _ => { print("other"); }
            }
        """)
        assert out == ["ten"]


# ============================================================
# Execution Tests -- Complex Scenarios
# ============================================================

class TestComplexScenarios:
    def test_match_in_while_loop(self):
        _, out = run("""
            let i = 0;
            while (i < 3) {
                match (i) {
                    0 => { print("zero"); }
                    1 => { print("one"); }
                    2 => { print("two"); }
                }
                i = i + 1;
            }
        """)
        assert out == ["zero", "one", "two"]

    def test_match_with_computation(self):
        _, out = run("""
            let x = 15;
            match (x % 3) {
                0 => { print("div3"); }
                1 => { print("rem1"); }
                2 => { print("rem2"); }
            }
        """)
        assert out == ["div3"]

    def test_sequential_matches(self):
        _, out = run("""
            let a = 1;
            let b = 2;
            match (a) {
                1 => { print("a=1"); }
                _ => { print("a=?"); }
            }
            match (b) {
                2 => { print("b=2"); }
                _ => { print("b=?"); }
            }
        """)
        assert out == ["a=1", "b=2"]

    def test_match_string_comparison(self):
        _, out = run("""
            fn greet(name) {
                match (name) {
                    "Alice" => { print("Hello Alice!"); }
                    "Bob" => { print("Hey Bob!"); }
                    _ => { print("Hi stranger!"); }
                }
            }
            greet("Alice");
            greet("Charlie");
        """)
        assert out == ["Hello Alice!", "Hi stranger!"]

    def test_fizzbuzz_match(self):
        _, out = run("""
            let i = 1;
            while (i <= 5) {
                let r3 = i % 3;
                let r5 = i % 5;
                match (r3) {
                    0 => {
                        match (r5) {
                            0 => { print("FizzBuzz"); }
                            _ => { print("Fizz"); }
                        }
                    }
                    _ => {
                        match (r5) {
                            0 => { print("Buzz"); }
                            _ => { print(i); }
                        }
                    }
                }
                i = i + 1;
            }
        """)
        assert out == ["1", "2", "Fizz", "4", "Buzz"]

    def test_match_bool_logic(self):
        _, out = run("""
            let x = true;
            let y = false;
            match (x and y) {
                true => { print("both"); }
                false => { print("not both"); }
            }
        """)
        assert out == ["not both"]

    def test_match_preserves_outer_scope(self):
        _, out = run("""
            let x = 10;
            let y = 0;
            match (x) {
                n => { y = n + 5; }
            }
            print(y);
        """)
        assert out == ["15"]


# ============================================================
# Execution Tests -- Tuple Operations
# ============================================================

class TestTupleOperations:
    def test_tuple_equality(self):
        _, out = run("""
            let a = (1, 2);
            let b = (1, 2);
            match (a == b) {
                true => { print("equal"); }
                false => { print("not equal"); }
            }
        """)
        assert out == ["equal"]

    def test_tuple_inequality(self):
        _, out = run("""
            let a = (1, 2);
            let b = (1, 3);
            match (a == b) {
                true => { print("equal"); }
                false => { print("not equal"); }
            }
        """)
        assert out == ["not equal"]

    def test_tuple_in_variable(self):
        r = execute("let t = (10, 20, 30);")
        t = r['env']['t']
        assert isinstance(t, TupleValue)
        assert t.elements == (10, 20, 30)

    def test_singleton_is_not_tuple(self):
        # (x) should be a grouped expression, not a 1-tuple
        r = execute("let x = (42);")
        assert r['env']['x'] == 42
        assert not isinstance(r['env']['x'], TupleValue)

    def test_tuple_print_nested(self):
        _, out = run("""
            let t = ((1, 2), (3, 4));
            print(t);
        """)
        assert out == ["((1, 2), (3, 4))"]


# ============================================================
# Execution Tests -- Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_arm_wildcard(self):
        _, out = run("""
            match (42) {
                _ => { print("always"); }
            }
        """)
        assert out == ["always"]

    def test_match_expr_semicolon(self):
        # Match as statement with semicolon
        _, out = run("""
            let x = 1;
            match (x) { 1 => { print("ok"); } };
        """)
        assert out == ["ok"]

    def test_match_with_side_effects(self):
        _, out = run("""
            let count = 0;
            let x = 2;
            match (x) {
                1 => { count = count + 1; }
                2 => { count = count + 10; }
                _ => { count = count + 100; }
            }
            print(count);
        """)
        assert out == ["10"]

    def test_guard_var_binding_used(self):
        _, out = run("""
            let x = 7;
            match (x) {
                n if n % 2 == 0 => { print("even"); }
                n if n % 2 == 1 => { print("odd"); }
                _ => { print("?"); }
            }
        """)
        assert out == ["odd"]

    def test_match_negative_float(self):
        _, out = run("""
            let x = -2.5;
            match (x) {
                -2.5 => { print("yes"); }
                _ => { print("no"); }
            }
        """)
        assert out == ["yes"]

    def test_or_pattern_with_wildcard_fallback(self):
        _, out = run("""
            let x = 5;
            match (x) {
                1 | 2 => { print("low"); }
                3 | 4 => { print("mid"); }
                _ => { print("high"); }
            }
        """)
        assert out == ["high"]

    def test_tuple_with_or_inside(self):
        """Or pattern inside tuple element."""
        _, out = run("""
            let t = (1, 3);
            match (t) {
                (1, 2 | 3) => { print("yes"); }
                _ => { print("no"); }
            }
        """)
        assert out == ["yes"]

    def test_many_arms(self):
        _, out = run("""
            let x = 5;
            match (x) {
                1 => { print("1"); }
                2 => { print("2"); }
                3 => { print("3"); }
                4 => { print("4"); }
                5 => { print("5"); }
                _ => { print("?"); }
            }
        """)
        assert out == ["5"]


# ============================================================
# Execution Tests -- Regression / Tricky Cases
# ============================================================

class TestRegression:
    def test_match_does_not_pollute_env(self):
        """Hidden match variable should not be user-visible."""
        r = execute("""
            let x = 5;
            match (x) {
                n => { print(n); }
            }
        """)
        # The __match_*__ variable exists but user shouldn't stumble on it
        assert r['output'] == ["5"]

    def test_match_in_if(self):
        _, out = run("""
            let x = 3;
            if (x > 0) {
                match (x) {
                    3 => { print("three"); }
                    _ => { print("other"); }
                }
            }
        """)
        assert out == ["three"]

    def test_match_after_match(self):
        _, out = run("""
            match (1) {
                1 => { print("first"); }
            }
            match (2) {
                2 => { print("second"); }
            }
        """)
        assert out == ["first", "second"]

    def test_false_literal_not_confused_with_zero(self):
        """Python's True==1, False==0 -- ensure type-aware matching."""
        _, out = run("""
            match (false) {
                0 => { print("zero"); }
                false => { print("false"); }
            }
        """)
        assert out == ["false"]

    def test_true_literal_not_confused_with_one(self):
        _, out = run("""
            match (true) {
                1 => { print("one"); }
                true => { print("true"); }
            }
        """)
        assert out == ["true"]

    def test_tuple_var_scope(self):
        _, out = run("""
            let t = (100, 200);
            match (t) {
                (a, b) => {
                    let sum = a + b;
                    print(sum);
                }
            }
        """)
        assert out == ["300"]

    def test_guard_fails_no_side_effect(self):
        """Guard failure should not execute the body."""
        _, out = run("""
            let x = 3;
            let modified = false;
            match (x) {
                n if n > 100 => { modified = true; }
                _ => { print("default"); }
            }
            print(modified);
        """)
        assert out == ["default", "false"]


# ============================================================
# Execution Tests -- Pattern Matching Dispatch
# ============================================================

class TestDispatch:
    def test_type_dispatch(self):
        """Dispatch based on tuple tag pattern."""
        _, out = run("""
            let msg = ("add", 3, 4);
            match (msg) {
                ("add", a, b) => {
                    let r = a + b;
                    print(r);
                }
                ("sub", a, b) => {
                    let r = a - b;
                    print(r);
                }
                _ => { print("unknown"); }
            }
        """)
        assert out == ["7"]

    def test_command_dispatch(self):
        _, out = run("""
            let cmd = ("greet", "World");
            match (cmd) {
                ("greet", name) => {
                    print("Hello");
                    print(name);
                }
                ("bye", name) => {
                    print("Goodbye");
                    print(name);
                }
            }
        """)
        assert out == ["Hello", "World"]

    def test_point_operations(self):
        _, out = run("""
            let p = (3, 4);
            match (p) {
                (0, 0) => { print("origin"); }
                (x, 0) => { print("x-axis"); }
                (0, y) => { print("y-axis"); }
                (x, y) => {
                    print(x);
                    print(y);
                }
            }
        """)
        assert out == ["3", "4"]

    def test_point_on_axis(self):
        _, out = run("""
            let p = (5, 0);
            match (p) {
                (0, 0) => { print("origin"); }
                (x, 0) => { print("x-axis"); }
                (0, y) => { print("y-axis"); }
                _ => { print("general"); }
            }
        """)
        assert out == ["x-axis"]

    def test_result_pattern(self):
        """Tagged tuple as Result type."""
        _, out = run("""
            let result = ("ok", 42);
            match (result) {
                ("ok", val) => { print(val); }
                ("err", msg) => { print(msg); }
            }
        """)
        assert out == ["42"]


# ============================================================
# Execution Tests -- Nested Match
# ============================================================

class TestNestedMatch:
    def test_nested_match_inner(self):
        _, out = run("""
            let x = 1;
            let y = 2;
            match (x) {
                1 => {
                    match (y) {
                        2 => { print("1,2"); }
                        _ => { print("1,?"); }
                    }
                }
                _ => { print("?,?"); }
            }
        """)
        assert out == ["1,2"]

    def test_nested_match_outer_fail(self):
        _, out = run("""
            let x = 9;
            let y = 2;
            match (x) {
                1 => {
                    match (y) {
                        2 => { print("1,2"); }
                        _ => { print("1,?"); }
                    }
                }
                _ => { print("other"); }
            }
        """)
        assert out == ["other"]

    def test_triple_nested(self):
        _, out = run("""
            match (1) {
                1 => {
                    match (2) {
                        2 => {
                            match (3) {
                                3 => { print("deep"); }
                            }
                        }
                    }
                }
            }
        """)
        assert out == ["deep"]


# ============================================================
# Execution Tests -- Advanced Guard Patterns
# ============================================================

class TestAdvancedGuards:
    def test_guard_with_multiple_vars(self):
        _, out = run("""
            let x = 10;
            let y = 5;
            match (x) {
                n if n > y => { print("x>y"); }
                _ => { print("nope"); }
            }
        """)
        assert out == ["x>y"]

    def test_guard_equality(self):
        _, out = run("""
            let x = 42;
            match (x) {
                n if n == 42 => { print("found"); }
                _ => { print("nope"); }
            }
        """)
        assert out == ["found"]

    def test_guard_with_function(self):
        _, out = run("""
            fn is_even(n) {
                return n % 2 == 0;
            }
            let x = 8;
            match (x) {
                n if is_even(n) => { print("even"); }
                _ => { print("odd"); }
            }
        """)
        assert out == ["even"]

    def test_all_guards_fail(self):
        _, out = run("""
            let x = 0;
            match (x) {
                n if n > 0 => { print("pos"); }
                n if n < 0 => { print("neg"); }
                _ => { print("zero"); }
            }
        """)
        assert out == ["zero"]


# ============================================================
# Execution Tests -- Advanced Tuple Patterns
# ============================================================

class TestAdvancedTuples:
    def test_swap_tuple(self):
        _, out = run("""
            let t = (1, 2);
            match (t) {
                (a, b) => {
                    let swapped = (b, a);
                    print(swapped);
                }
            }
        """)
        assert out == ["(2, 1)"]

    def test_tuple_as_scrutinee_expr(self):
        _, out = run("""
            match ((1 + 2, 3 + 4)) {
                (3, 7) => { print("yes"); }
                _ => { print("no"); }
            }
        """)
        assert out == ["yes"]

    def test_deeply_nested_tuple(self):
        _, out = run("""
            let t = ((1, (2, 3)), 4);
            match (t) {
                ((a, (b, c)), d) => {
                    let sum = a + b + c + d;
                    print(sum);
                }
            }
        """)
        assert out == ["10"]

    def test_tuple_with_wildcard_elements(self):
        _, out = run("""
            let t = (1, 2, 3);
            match (t) {
                (_, _, x) => { print(x); }
            }
        """)
        assert out == ["3"]

    def test_tuple_partial_literal_match(self):
        _, out = run("""
            let t = ("add", 5, 3);
            match (t) {
                ("add", a, b) => {
                    print(a + b);
                }
                ("mul", a, b) => {
                    print(a * b);
                }
            }
        """)
        assert out == ["8"]

    def test_tuple_pattern_fails_non_tuple(self):
        _, out = run("""
            match (42) {
                (x, y) => { print("tuple"); }
                n => { print(n); }
            }
        """)
        assert out == ["42"]


# ============================================================
# Execution Tests -- Or Pattern Edge Cases
# ============================================================

class TestOrPatternEdgeCases:
    def test_or_single_match_first(self):
        _, out = run("""
            match (1) {
                1 | 2 => { print("match"); }
            }
        """)
        assert out == ["match"]

    def test_or_single_match_second(self):
        _, out = run("""
            match (2) {
                1 | 2 => { print("match"); }
            }
        """)
        assert out == ["match"]

    def test_or_none_match(self):
        _, out = run("""
            match (3) {
                1 | 2 => { print("match"); }
                _ => { print("nope"); }
            }
        """)
        assert out == ["nope"]

    def test_or_many_alternatives(self):
        _, out = run("""
            match (4) {
                1 | 2 | 3 | 4 => { print("yes"); }
                _ => { print("no"); }
            }
        """)
        assert out == ["yes"]

    def test_or_with_strings(self):
        _, out = run("""
            match ("c") {
                "a" | "b" | "c" => { print("abc"); }
                _ => { print("other"); }
            }
        """)
        assert out == ["abc"]

    def test_or_between_arms(self):
        _, out = run("""
            let x = 5;
            match (x) {
                1 | 2 => { print("low"); }
                3 | 4 => { print("mid"); }
                5 | 6 => { print("high"); }
                _ => { print("?"); }
            }
        """)
        assert out == ["high"]


# ============================================================
# Execution Tests -- Match Expression Value
# ============================================================

class TestMatchExprValue:
    def test_expr_body_int(self):
        r = execute("""
            let x = match (2) {
                1 => 10
                2 => 20
                _ => 0
            };
        """)
        assert r['env']['x'] == 20

    def test_expr_body_string(self):
        r = execute("""
            let x = match ("hi") {
                "hi" => "hello"
                _ => "unknown"
            };
            print(x);
        """)
        assert r['output'] == ["hello"]

    def test_expr_body_arithmetic(self):
        r = execute("""
            let x = match (3) {
                n => n * 10
            };
            print(x);
        """)
        assert r['output'] == ["30"]

    def test_expr_body_bool(self):
        r = execute("""
            let x = match (5) {
                n if n > 0 => true
                _ => false
            };
            print(x);
        """)
        assert r['output'] == ["true"]


# ============================================================
# Execution Tests -- Stress / Integration
# ============================================================

class TestIntegration:
    def test_classifier(self):
        """Classify numbers using match."""
        _, out = run("""
            fn classify(n) {
                match (n) {
                    0 => { print("zero"); }
                    n if n > 0 => { print("positive"); }
                    _ => { print("negative"); }
                }
            }
            classify(0);
            classify(5);
            classify(-3);
        """)
        assert out == ["zero", "positive", "negative"]

    def test_recursive_with_match(self):
        _, out = run("""
            fn fib(n) {
                match (n) {
                    0 => { return 0; }
                    1 => { return 1; }
                    _ => {
                        return fib(n - 1) + fib(n - 2);
                    }
                }
            }
            print(fib(0));
            print(fib(1));
            print(fib(5));
            print(fib(7));
        """)
        assert out == ["0", "1", "5", "13"]

    def test_list_operations_with_tuples(self):
        """Use tuples as cons cells."""
        _, out = run("""
            // Represent a linked list as nested tuples
            // ("cons", value, rest) or "nil"
            let list = ("cons", 1, ("cons", 2, ("cons", 3, "nil")));
            // Get the head
            match (list) {
                ("cons", head, _) => { print(head); }
                _ => { print("empty"); }
            }
        """)
        assert out == ["1"]

    def test_state_machine(self):
        _, out = run("""
            let state = "start";
            let i = 0;
            while (i < 3) {
                match (state) {
                    "start" => { state = "running"; }
                    "running" => { state = "done"; }
                    "done" => { state = "start"; }
                }
                i = i + 1;
            }
            print(state);
        """)
        assert out == ["start"]

    def test_calculator(self):
        _, out = run("""
            fn calc(op) {
                match (op) {
                    ("add", a, b) => { return a + b; }
                    ("sub", a, b) => { return a - b; }
                    ("mul", a, b) => { return a * b; }
                    _ => { return 0; }
                }
            }
            print(calc(("add", 10, 20)));
            print(calc(("sub", 100, 42)));
            print(calc(("mul", 6, 7)));
        """)
        assert out == ["30", "58", "42"]


# ============================================================
# Count test functions
# ============================================================

if __name__ == '__main__':
    import inspect
    count = 0
    for name, cls in list(globals().items()):
        if isinstance(cls, type) and name.startswith('Test'):
            for mname, _ in inspect.getmembers(cls, predicate=inspect.isfunction):
                if mname.startswith('test_'):
                    count += 1
    print(f"Total test functions: {count}")
    pytest.main([__file__, '-v'])
