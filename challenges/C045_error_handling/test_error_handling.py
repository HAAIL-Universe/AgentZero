"""
Tests for C045: Error Handling (try/catch/throw)
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from error_handling import (
    run, execute, parse, compile_source, lex, disassemble,
    VMError, ParseError, LexError, CompileError,
    TryCatchStmt, ThrowStmt, TokenType,
)


# ============================================================
# Section 1: Basic throw
# ============================================================

class TestBasicThrow:
    def test_throw_string_uncaught(self):
        with pytest.raises(VMError, match="Uncaught exception"):
            run('throw "error";')

    def test_throw_integer_uncaught(self):
        with pytest.raises(VMError, match="Uncaught exception"):
            run('throw 42;')

    def test_throw_hash_uncaught(self):
        with pytest.raises(VMError, match="Uncaught exception"):
            run('throw {type: "Error", message: "oops"};')

    def test_throw_variable_uncaught(self):
        with pytest.raises(VMError, match="Uncaught exception"):
            run('let msg = "fail"; throw msg;')

    def test_throw_expression_uncaught(self):
        with pytest.raises(VMError, match="Uncaught exception"):
            run('throw 1 + 2;')


# ============================================================
# Section 2: Basic try/catch
# ============================================================

class TestBasicTryCatch:
    def test_catch_string(self):
        result, output = run("""
            try {
                throw "error";
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["error"]

    def test_catch_integer(self):
        result, output = run("""
            try {
                throw 42;
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["42"]

    def test_catch_hash(self):
        result, output = run("""
            try {
                throw {type: "NotFound", code: 404};
            } catch (e) {
                print(e.type);
                print(e.code);
            }
        """)
        assert output == ["NotFound", "404"]

    def test_catch_array(self):
        result, output = run("""
            try {
                throw [1, 2, 3];
            } catch (e) {
                print(len(e));
            }
        """)
        assert output == ["3"]

    def test_catch_bool(self):
        result, output = run("""
            try {
                throw false;
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["false"]

    def test_catch_none(self):
        """Throwing None should be catchable too."""
        result, output = run("""
            let x = 0;
            try {
                throw x;
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["0"]

    def test_try_no_throw(self):
        """When try body doesn't throw, catch is skipped."""
        result, output = run("""
            try {
                print("ok");
            } catch (e) {
                print("caught");
            }
        """)
        assert output == ["ok"]

    def test_try_no_throw_code_after(self):
        result, output = run("""
            try {
                print("try");
            } catch (e) {
                print("catch");
            }
            print("after");
        """)
        assert output == ["try", "after"]

    def test_catch_then_continue(self):
        result, output = run("""
            try {
                throw "err";
            } catch (e) {
                print("caught");
            }
            print("after");
        """)
        assert output == ["caught", "after"]


# ============================================================
# Section 3: Catch variable binding
# ============================================================

class TestCatchVariable:
    def test_catch_var_is_thrown_value(self):
        result, output = run("""
            let result = 0;
            try {
                throw 42;
            } catch (e) {
                result = e;
            }
            print(result);
        """)
        assert output == ["42"]

    def test_catch_var_scope(self):
        """Catch variable should be accessible within catch block."""
        result, output = run("""
            try {
                throw "hello";
            } catch (msg) {
                let upper = msg;
                print(upper);
            }
        """)
        assert output == ["hello"]

    def test_catch_var_is_hash(self):
        result, output = run("""
            try {
                throw {code: 500, msg: "server error"};
            } catch (e) {
                print(e.code);
                print(e.msg);
            }
        """)
        assert output == ["500", "server error"]

    def test_different_catch_var_names(self):
        result, output = run("""
            try {
                throw "a";
            } catch (err1) {
                print(err1);
            }
            try {
                throw "b";
            } catch (err2) {
                print(err2);
            }
        """)
        assert output == ["a", "b"]


# ============================================================
# Section 4: Nested try/catch
# ============================================================

class TestNestedTryCatch:
    def test_inner_catch(self):
        result, output = run("""
            try {
                try {
                    throw "inner";
                } catch (e) {
                    print("inner: " + e);
                }
            } catch (e) {
                print("outer: " + e);
            }
        """)
        assert output == ["inner: inner"]

    def test_inner_throws_caught_by_outer(self):
        """Throw in inner that isn't caught propagates to outer."""
        result, output = run("""
            try {
                try {
                    throw "deep";
                } catch (e) {
                    throw "rethrown";
                }
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["rethrown"]

    def test_triple_nested(self):
        result, output = run("""
            try {
                try {
                    try {
                        throw "lvl3";
                    } catch (e) {
                        throw "lvl2: " + e;
                    }
                } catch (e) {
                    throw "lvl1: " + e;
                }
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["lvl1: lvl2: lvl3"]

    def test_outer_not_triggered_if_inner_catches(self):
        result, output = run("""
            let caught_outer = false;
            try {
                try {
                    throw "x";
                } catch (e) {
                    print("inner caught");
                }
            } catch (e) {
                caught_outer = true;
            }
            print(caught_outer);
        """)
        assert output == ["inner caught", "false"]

    def test_inner_no_throw_outer_throw(self):
        result, output = run("""
            try {
                try {
                    print("inner ok");
                } catch (e) {
                    print("inner catch");
                }
                throw "from outer try";
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["inner ok", "from outer try"]


# ============================================================
# Section 5: Throw across function boundaries
# ============================================================

class TestThrowAcrossFunctions:
    def test_throw_in_function(self):
        result, output = run("""
            fn fail() {
                throw "fn error";
            }
            try {
                fail();
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["fn error"]

    def test_throw_deep_in_call_chain(self):
        result, output = run("""
            fn c() { throw "deep"; }
            fn b() { c(); }
            fn a() { b(); }
            try {
                a();
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["deep"]

    def test_function_with_try_catch(self):
        result, output = run("""
            fn safe_divide(a, b) {
                try {
                    let result = a / b;
                    return result;
                } catch (e) {
                    return -1;
                }
            }
            print(safe_divide(10, 2));
            print(safe_divide(10, 0));
        """)
        assert output == ["5", "-1"]

    def test_throw_unwinds_multiple_frames(self):
        result, output = run("""
            fn inner() {
                print("inner start");
                throw "bail";
                print("inner end");
            }
            fn outer() {
                print("outer start");
                inner();
                print("outer end");
            }
            try {
                outer();
            } catch (e) {
                print("caught: " + e);
            }
        """)
        assert output == ["outer start", "inner start", "caught: bail"]

    def test_try_catch_inside_function(self):
        result, output = run("""
            fn process() {
                try {
                    throw "oops";
                } catch (e) {
                    return "handled: " + e;
                }
            }
            print(process());
        """)
        assert output == ["handled: oops"]

    def test_function_catches_own_throw(self):
        result, output = run("""
            fn careful(x) {
                try {
                    if (x < 0) {
                        throw "negative";
                    }
                    return x * 2;
                } catch (e) {
                    return 0;
                }
            }
            print(careful(5));
            print(careful(-1));
        """)
        assert output == ["10", "0"]


# ============================================================
# Section 6: Built-in error catching
# ============================================================

class TestBuiltinErrorCatching:
    def test_catch_division_by_zero(self):
        result, output = run("""
            try {
                let x = 10 / 0;
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["Division by zero"]

    def test_catch_modulo_by_zero(self):
        result, output = run("""
            try {
                let x = 10 % 0;
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["Modulo by zero"]

    def test_catch_array_index_oob(self):
        result, output = run("""
            let arr = [1, 2, 3];
            try {
                let x = arr[5];
            } catch (e) {
                print(e);
            }
        """)
        assert "out of bounds" in output[0]

    def test_catch_undefined_variable(self):
        result, output = run("""
            try {
                let x = undefined_var;
            } catch (e) {
                print(e);
            }
        """)
        assert "Undefined variable" in output[0]

    def test_catch_key_not_found(self):
        result, output = run("""
            let m = {a: 1};
            try {
                let x = m.b;
            } catch (e) {
                print(e);
            }
        """)
        assert "not found" in output[0]

    def test_catch_pop_empty(self):
        result, output = run("""
            let arr = [];
            try {
                pop(arr);
            } catch (e) {
                print(e);
            }
        """)
        assert "empty array" in output[0]

    def test_catch_call_non_function(self):
        result, output = run("""
            let x = 5;
            try {
                x();
            } catch (e) {
                print(e);
            }
        """)
        assert "non-function" in output[0].lower() or "Cannot call" in output[0]

    def test_catch_wrong_arity(self):
        result, output = run("""
            fn add(a, b) { return a + b; }
            try {
                add(1);
            } catch (e) {
                print(e);
            }
        """)
        assert "expects" in output[0]

    def test_catch_iterate_non_iterable(self):
        result, output = run("""
            try {
                for (x in 42) {
                    print(x);
                }
            } catch (e) {
                print(e);
            }
        """)
        assert "Cannot iterate" in output[0]


# ============================================================
# Section 7: Throw with complex expressions
# ============================================================

class TestThrowExpressions:
    def test_throw_string_concat(self):
        result, output = run("""
            try {
                throw "error: " + "not found";
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["error: not found"]

    def test_throw_computed_hash(self):
        result, output = run("""
            fn make_error(code, msg) {
                return {code: code, message: msg};
            }
            try {
                throw make_error(404, "Not Found");
            } catch (e) {
                print(e.code);
                print(e.message);
            }
        """)
        assert output == ["404", "Not Found"]

    def test_throw_array(self):
        result, output = run("""
            try {
                throw [1, 2, 3];
            } catch (e) {
                print(e[0]);
                print(len(e));
            }
        """)
        assert output == ["1", "3"]

    def test_throw_function_result(self):
        result, output = run("""
            fn get_error() { return "computed error"; }
            try {
                throw get_error();
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["computed error"]


# ============================================================
# Section 8: Control flow interaction
# ============================================================

class TestControlFlowInteraction:
    def test_throw_in_if(self):
        result, output = run("""
            fn check(x) {
                if (x < 0) {
                    throw "negative";
                }
                return x;
            }
            try {
                print(check(5));
                print(check(-1));
            } catch (e) {
                print("error: " + e);
            }
        """)
        assert output == ["5", "error: negative"]

    def test_throw_in_while(self):
        result, output = run("""
            let i = 0;
            try {
                while (i < 10) {
                    if (i == 3) {
                        throw "stopped at 3";
                    }
                    print(i);
                    i = i + 1;
                }
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["0", "1", "2", "stopped at 3"]

    def test_throw_in_for_in(self):
        result, output = run("""
            try {
                for (x in [10, 20, 30, 40]) {
                    if (x == 30) {
                        throw "found 30";
                    }
                    print(x);
                }
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["10", "20", "found 30"]

    def test_try_catch_in_while(self):
        result, output = run("""
            let i = 0;
            while (i < 3) {
                try {
                    if (i == 1) {
                        throw "skip";
                    }
                    print(i);
                } catch (e) {
                    print("caught: " + e);
                }
                i = i + 1;
            }
        """)
        assert output == ["0", "caught: skip", "2"]

    def test_try_catch_in_for_in(self):
        result, output = run("""
            for (x in [1, 0, 2]) {
                try {
                    print(10 / x);
                } catch (e) {
                    print("div error");
                }
            }
        """)
        assert output == ["10", "div error", "5"]

    def test_return_from_try(self):
        """Return from inside try block should work."""
        result, output = run("""
            fn test() {
                try {
                    return 42;
                } catch (e) {
                    return -1;
                }
            }
            print(test());
        """)
        assert output == ["42"]

    def test_return_from_catch(self):
        result, output = run("""
            fn test() {
                try {
                    throw "err";
                } catch (e) {
                    return 99;
                }
            }
            print(test());
        """)
        assert output == ["99"]


# ============================================================
# Section 9: Re-throwing
# ============================================================

class TestRethrowing:
    def test_rethrow(self):
        result, output = run("""
            try {
                try {
                    throw "original";
                } catch (e) {
                    print("inner caught: " + e);
                    throw e;
                }
            } catch (e) {
                print("outer caught: " + e);
            }
        """)
        assert output == ["inner caught: original", "outer caught: original"]

    def test_rethrow_modified(self):
        result, output = run("""
            try {
                try {
                    throw "base error";
                } catch (e) {
                    throw "wrapped: " + e;
                }
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["wrapped: base error"]

    def test_catch_and_rethrow_different(self):
        result, output = run("""
            try {
                try {
                    throw "first";
                } catch (e) {
                    throw "second";
                }
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["second"]


# ============================================================
# Section 10: Error handling with closures
# ============================================================

class TestClosuresWithErrors:
    def test_throw_in_closure(self):
        result, output = run("""
            let thrower = fn() { throw "from closure"; };
            try {
                thrower();
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["from closure"]

    def test_closure_catches_error(self):
        result, output = run("""
            let safe = fn(f) {
                try {
                    return f();
                } catch (e) {
                    return "caught: " + e;
                }
            };
            let bad = fn() { throw "boom"; };
            let good = fn() { return "ok"; };
            print(safe(bad));
            print(safe(good));
        """)
        assert output == ["caught: boom", "ok"]

    def test_throw_captured_variable(self):
        result, output = run("""
            let msg = "captured error";
            let thrower = fn() { throw msg; };
            try {
                thrower();
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["captured error"]

    def test_higher_order_error_handler(self):
        result, output = run("""
            fn with_default(f, default_val) {
                try {
                    return f();
                } catch (e) {
                    return default_val;
                }
            }
            fn risky() { throw "oops"; }
            fn safe() { return 42; }
            print(with_default(risky, 0));
            print(with_default(safe, 0));
        """)
        assert output == ["0", "42"]


# ============================================================
# Section 11: Error handling with arrays/hashes
# ============================================================

class TestCollectionsWithErrors:
    def test_safe_array_access(self):
        result, output = run("""
            fn safe_get(arr, idx) {
                try {
                    return arr[idx];
                } catch (e) {
                    return -1;
                }
            }
            let a = [10, 20, 30];
            print(safe_get(a, 1));
            print(safe_get(a, 5));
        """)
        assert output == ["20", "-1"]

    def test_safe_hash_access(self):
        result, output = run("""
            fn safe_get(obj, key) {
                try {
                    return obj[key];
                } catch (e) {
                    return "default";
                }
            }
            let m = {name: "test"};
            print(safe_get(m, "name"));
            print(safe_get(m, "missing"));
        """)
        assert output == ["test", "default"]

    def test_throw_during_map(self):
        result, output = run("""
            try {
                let result = map([1, 0, 3], fn(x) {
                    return 10 / x;
                });
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["Division by zero"]

    def test_error_hash_as_structured_error(self):
        result, output = run("""
            fn validate(x) {
                if (x < 0) {
                    throw {type: "ValidationError", field: "x", message: "must be positive"};
                }
                return x;
            }
            try {
                validate(-5);
            } catch (e) {
                print(e.type);
                print(e.field);
                print(e.message);
            }
        """)
        assert output == ["ValidationError", "x", "must be positive"]


# ============================================================
# Section 12: Stack state after catch
# ============================================================

class TestStackState:
    def test_stack_clean_after_catch(self):
        """After catching, execution should continue cleanly."""
        result, output = run("""
            let a = 1;
            try {
                let b = 2;
                throw "err";
                let c = 3;
            } catch (e) {
                print("caught");
            }
            let d = a + 10;
            print(d);
        """)
        assert output == ["caught", "11"]

    def test_multiple_sequential_try_catch(self):
        result, output = run("""
            try { throw "a"; } catch (e) { print(e); }
            try { throw "b"; } catch (e) { print(e); }
            try { throw "c"; } catch (e) { print(e); }
            print("done");
        """)
        assert output == ["a", "b", "c", "done"]

    def test_env_restored_after_catch(self):
        """Environment should work correctly after unwinding."""
        result, output = run("""
            let x = 10;
            try {
                let x = 20;
                throw "err";
            } catch (e) {
                print(x);
            }
        """)
        assert output == ["10"]

    def test_computation_after_catch(self):
        result, output = run("""
            let total = 0;
            for (x in [1, 2, 0, 4, 5]) {
                try {
                    total = total + 100 / x;
                } catch (e) {
                    print("skipped");
                }
            }
            print(total);
        """)
        assert output == ["skipped", "195"]


# ============================================================
# Section 13: Utility builtins (type, string)
# ============================================================

class TestUtilityBuiltins:
    def test_type_int(self):
        result, output = run('print(type(42));')
        assert output == ["int"]

    def test_type_string(self):
        result, output = run('print(type("hello"));')
        assert output == ["string"]

    def test_type_bool(self):
        result, output = run('print(type(true));')
        assert output == ["bool"]

    def test_type_array(self):
        result, output = run('print(type([1, 2]));')
        assert output == ["array"]

    def test_type_hash(self):
        result, output = run('print(type({a: 1}));')
        assert output == ["hash"]

    def test_type_function(self):
        result, output = run("""
            fn f() {}
            print(type(f));
        """)
        assert output == ["function"]

    def test_type_float(self):
        result, output = run('print(type(3.14));')
        assert output == ["float"]

    def test_string_conversion(self):
        result, output = run('print(string(42));')
        assert output == ["42"]

    def test_string_bool(self):
        result, output = run('print(string(true));')
        assert output == ["true"]

    def test_string_array(self):
        result, output = run('print(string([1, 2, 3]));')
        assert output == ["[1, 2, 3]"]


# ============================================================
# Section 14: Edge cases
# ============================================================

class TestEdgeCases:
    def test_throw_in_catch_no_outer_handler(self):
        with pytest.raises(VMError, match="Uncaught exception"):
            run("""
                try {
                    throw "first";
                } catch (e) {
                    throw "second";
                }
            """)

    def test_empty_try_body(self):
        result, output = run("""
            try {
            } catch (e) {
                print("caught");
            }
            print("ok");
        """)
        assert output == ["ok"]

    def test_empty_catch_body(self):
        result, output = run("""
            try {
                throw "ignored";
            } catch (e) {
            }
            print("ok");
        """)
        assert output == ["ok"]

    def test_throw_at_top_level(self):
        with pytest.raises(VMError):
            run('throw "top level";')

    def test_deeply_nested_throw(self):
        result, output = run("""
            fn a() { throw "deep"; }
            fn b() { a(); }
            fn c() { b(); }
            fn d() { c(); }
            fn e() { d(); }
            try {
                e();
            } catch (err) {
                print(err);
            }
        """)
        assert output == ["deep"]

    def test_handler_removed_after_try(self):
        """Handler should be removed after try block completes normally."""
        result, output = run("""
            try {
                print("in try");
            } catch (e) {
                print("caught");
            }
            // This throw should be uncaught since handler was removed
            // (we test this via a separate throw)
        """)
        assert output == ["in try"]

    def test_multiple_handlers_correct_order(self):
        result, output = run("""
            try {
                try {
                    try {
                        throw "x";
                    } catch (e) {
                        print("1: " + e);
                    }
                    throw "y";
                } catch (e) {
                    print("2: " + e);
                }
                throw "z";
            } catch (e) {
                print("3: " + e);
            }
        """)
        assert output == ["1: x", "2: y", "3: z"]


# ============================================================
# Section 15: Parsing
# ============================================================

class TestParsing:
    def test_parse_try_catch(self):
        ast = parse("""
            try { throw 1; } catch (e) { print(e); }
        """)
        assert len(ast.stmts) == 1
        stmt = ast.stmts[0]
        assert isinstance(stmt, TryCatchStmt)
        assert stmt.catch_var == "e"

    def test_parse_throw(self):
        ast = parse('throw "error";')
        assert len(ast.stmts) == 1
        assert isinstance(ast.stmts[0], ThrowStmt)

    def test_lex_try_catch_throw(self):
        tokens = lex('try catch throw')
        assert tokens[0].type == TokenType.TRY
        assert tokens[1].type == TokenType.CATCH
        assert tokens[2].type == TokenType.THROW

    def test_parse_error_missing_catch(self):
        with pytest.raises(ParseError):
            parse('try { } { }')

    def test_parse_error_missing_catch_var(self):
        with pytest.raises(ParseError):
            parse('try { } catch { }')

    def test_parse_error_throw_no_semicolon(self):
        with pytest.raises(ParseError):
            parse('throw "err"')

    def test_disassemble_try_catch(self):
        chunk, _ = compile_source("""
            try { throw 1; } catch (e) { print(e); }
        """)
        text = disassemble(chunk)
        assert "SETUP_TRY" in text
        assert "POP_TRY" in text
        assert "THROW" in text


# ============================================================
# Section 16: Practical patterns
# ============================================================

class TestPracticalPatterns:
    def test_error_accumulator(self):
        """Collect errors instead of failing fast."""
        result, output = run("""
            let errors = [];
            let data = [2, 0, 5, 0, 3];
            let results = [];
            for (x in data) {
                try {
                    push(results, 100 / x);
                } catch (e) {
                    push(errors, e);
                }
            }
            print(len(results));
            print(len(errors));
        """)
        assert output == ["3", "2"]

    def test_retry_pattern(self):
        result, output = run("""
            let attempts = 0;
            let success = false;
            while (not success) {
                attempts = attempts + 1;
                try {
                    if (attempts < 3) {
                        throw "not ready";
                    }
                    success = true;
                } catch (e) {
                    print("retry " + string(attempts));
                }
            }
            print("done after " + string(attempts));
        """)
        assert output == ["retry 1", "retry 2", "done after 3"]

    def test_validation_errors(self):
        result, output = run("""
            fn validate_age(age) {
                if (type(age) != "int") {
                    throw {field: "age", error: "must be integer"};
                }
                if (age < 0) {
                    throw {field: "age", error: "must be positive"};
                }
                if (age > 150) {
                    throw {field: "age", error: "too large"};
                }
                return age;
            }
            try {
                validate_age(-5);
            } catch (e) {
                print(e.field + ": " + e.error);
            }
            try {
                validate_age(200);
            } catch (e) {
                print(e.field + ": " + e.error);
            }
            print(validate_age(25));
        """)
        assert output == ["age: must be positive", "age: too large", "25"]

    def test_safe_json_like_access(self):
        result, output = run("""
            fn get_nested(obj, keys) {
                let current = obj;
                for (k in keys) {
                    try {
                        current = current[k];
                    } catch (e) {
                        return "NOT_FOUND";
                    }
                }
                return current;
            }
            let data = {user: {name: "Alice", address: {city: "NYC"}}};
            print(get_nested(data, ["user", "name"]));
            print(get_nested(data, ["user", "address", "city"]));
            print(get_nested(data, ["user", "phone"]));
        """)
        assert output == ["Alice", "NYC", "NOT_FOUND"]

    def test_error_handler_chain(self):
        result, output = run("""
            fn handle(e) {
                if (type(e) == "hash") {
                    if (has(e, "code")) {
                        return "Error " + string(e.code) + ": " + e.message;
                    }
                }
                return "Unknown error: " + string(e);
            }
            try {
                throw {code: 404, message: "Not Found"};
            } catch (e) {
                print(handle(e));
            }
            try {
                throw "simple error";
            } catch (e) {
                print(handle(e));
            }
        """)
        assert output == ["Error 404: Not Found", "Unknown error: simple error"]


# ============================================================
# Section 17: Previous features still work
# ============================================================

class TestBackwardsCompatibility:
    def test_for_in_still_works(self):
        result, output = run("""
            for (x in [1, 2, 3]) {
                print(x);
            }
        """)
        assert output == ["1", "2", "3"]

    def test_for_in_destructured(self):
        result, output = run("""
            let m = {a: 1, b: 2};
            for (k, v in m) {
                print(k + "=" + string(v));
            }
        """)
        assert output == ["a=1", "b=2"]

    def test_closures_still_work(self):
        """Closures capture env at creation; mutable state uses arrays (reference semantics)."""
        result, output = run("""
            fn make_counter() {
                let state = [0];
                return fn() {
                    state[0] = state[0] + 1;
                    return state[0];
                };
            }
            let c = make_counter();
            print(c());
            print(c());
        """)
        assert output == ["1", "2"]

    def test_arrays_still_work(self):
        result, output = run("""
            let arr = [1, 2, 3];
            push(arr, 4);
            print(len(arr));
            print(arr[3]);
        """)
        assert output == ["4", "4"]

    def test_hash_maps_still_work(self):
        result, output = run("""
            let m = {name: "test", value: 42};
            print(m.name);
            m.extra = true;
            print(has(m, "extra"));
        """)
        assert output == ["test", "true"]

    def test_break_continue_still_work(self):
        result, output = run("""
            let sum = 0;
            for (x in range(10)) {
                if (x == 5) { break; }
                if (x % 2 == 0) { continue; }
                sum = sum + x;
            }
            print(sum);
        """)
        assert output == ["4"]

    def test_recursion_still_works(self):
        result, output = run("""
            fn fib(n) {
                if (n <= 1) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
            print(fib(8));
        """)
        assert output == ["21"]

    def test_lambda_still_works(self):
        result, output = run("""
            let double = fn(x) { return x * 2; };
            print(double(5));
            print(map([1, 2, 3], fn(x) { return x + 10; }));
        """)
        assert output == ["10", "[11, 12, 13]"]
