"""
Tests for C047: Iterators & Generators
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from iterators import (
    run, execute, parse, lex, compile_source,
    GeneratorObject, ClosureObject, FnObject,
    VMError, ParseError, ModuleRegistry,
    TokenType, YieldExpr, _ast_contains_yield,
)


# ============================================================
# 1. Basic Generator Creation
# ============================================================

class TestGeneratorCreation:
    def test_generator_function_returns_generator_object(self):
        r = execute("""
            fn gen() {
                yield 1;
            }
            let g = gen();
            type(g);
        """)
        assert r['result'] == "generator"

    def test_generator_not_executed_on_creation(self):
        """Calling a generator function does NOT execute it."""
        _, output = run("""
            fn gen() {
                print(99);
                yield 1;
            }
            let g = gen();
        """)
        assert output == []

    def test_generator_function_with_params(self):
        r = execute("""
            fn gen(x) {
                yield x;
            }
            let g = gen(42);
            next(g);
        """)
        assert r['result'] == 42

    def test_multiple_generators_independent(self):
        _, output = run("""
            fn counter(start) {
                yield start;
                yield start + 1;
            }
            let a = counter(0);
            let b = counter(10);
            print(next(a));
            print(next(b));
            print(next(a));
            print(next(b));
        """)
        assert output == ["0", "10", "1", "11"]

    def test_generator_type_check(self):
        _, output = run("""
            fn gen() { yield 1; }
            let g = gen();
            print(type(g));
        """)
        assert output == ["generator"]


# ============================================================
# 2. Basic Yield and Next
# ============================================================

class TestYieldAndNext:
    def test_single_yield(self):
        r = execute("fn g() { yield 42; } next(g());")
        assert r['result'] == 42

    def test_multiple_yields(self):
        _, output = run("""
            fn gen() {
                yield 1;
                yield 2;
                yield 3;
            }
            let g = gen();
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert output == ["1", "2", "3"]

    def test_yield_with_expression(self):
        r = execute("""
            fn gen() {
                let x = 10;
                yield x * 2;
            }
            next(gen());
        """)
        assert r['result'] == 20

    def test_yield_none(self):
        """Bare yield produces None."""
        r = execute("""
            fn gen() {
                yield;
            }
            next(gen());
        """)
        assert r['result'] is None

    def test_yield_string(self):
        r = execute("""
            fn gen() { yield "hello"; }
            next(gen());
        """)
        assert r['result'] == "hello"

    def test_yield_array(self):
        r = execute("""
            fn gen() { yield [1, 2, 3]; }
            next(gen());
        """)
        assert r['result'] == [1, 2, 3]

    def test_yield_hash(self):
        r = execute("""
            fn gen() { yield {a: 1}; }
            let v = next(gen());
        """)
        assert r['env']['v'] == {"a": 1}


# ============================================================
# 3. Generator Exhaustion
# ============================================================

class TestGeneratorExhaustion:
    def test_exhausted_returns_none(self):
        _, output = run("""
            fn gen() { yield 1; }
            let g = gen();
            print(next(g));
            print(next(g));
        """)
        assert output == ["1", "None"]

    def test_exhausted_stays_done(self):
        _, output = run("""
            fn gen() { yield 1; }
            let g = gen();
            next(g);
            next(g);
            next(g);
            print(next(g));
        """)
        assert output == ["None"]

    def test_next_with_default(self):
        r = execute("""
            fn gen() { yield 1; }
            let g = gen();
            next(g);
            next(g, "done");
        """)
        assert r['result'] == "done"

    def test_next_default_not_used_when_alive(self):
        r = execute("""
            fn gen() { yield 42; }
            let g = gen();
            next(g, "default");
        """)
        assert r['result'] == 42

    def test_empty_generator(self):
        """Generator with unreachable yield is immediately done."""
        _, output = run("""
            fn gen() {
                if (false) {
                    yield 1;
                }
            }
            let g = gen();
            print(next(g));
            print(next(g, "done"));
        """)
        assert output == ["None", "done"]


# ============================================================
# 4. Generator with Control Flow
# ============================================================

class TestGeneratorControlFlow:
    def test_yield_in_if(self):
        _, output = run("""
            fn gen(x) {
                if (x > 0) {
                    yield "positive";
                } else {
                    yield "non-positive";
                }
            }
            print(next(gen(5)));
            print(next(gen(-1)));
        """)
        assert output == ["positive", "non-positive"]

    def test_yield_in_while(self):
        _, output = run("""
            fn count_to(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            let g = count_to(3);
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g, "done"));
        """)
        assert output == ["0", "1", "2", "done"]

    def test_yield_in_for_in(self):
        _, output = run("""
            fn items() {
                let arr = [10, 20, 30];
                for (x in arr) {
                    yield x;
                }
            }
            let g = items();
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g, "end"));
        """)
        assert output == ["10", "20", "30", "end"]

    def test_multiple_yields_with_logic(self):
        _, output = run("""
            fn gen() {
                yield 1;
                let x = 2 + 3;
                yield x;
                yield x * 2;
            }
            let g = gen();
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert output == ["1", "5", "10"]

    def test_early_return_in_generator(self):
        """Return in a generator marks it done."""
        _, output = run("""
            fn gen() {
                yield 1;
                return;
                yield 2;
            }
            let g = gen();
            print(next(g));
            print(next(g, "done"));
        """)
        assert output == ["1", "done"]

    def test_conditional_yield_path(self):
        _, output = run("""
            fn gen(flag) {
                yield "start";
                if (flag) {
                    yield "a";
                    yield "b";
                } else {
                    yield "x";
                }
                yield "end";
            }
            let g1 = gen(true);
            let g2 = gen(false);
            print(next(g1));
            print(next(g1));
            print(next(g1));
            print(next(g1));
            print(next(g2));
            print(next(g2));
            print(next(g2));
        """)
        assert output == ["start", "a", "b", "end", "start", "x", "end"]


# ============================================================
# 5. Infinite Generators
# ============================================================

class TestInfiniteGenerators:
    def test_infinite_counter(self):
        _, output = run("""
            fn naturals() {
                let n = 0;
                while (true) {
                    yield n;
                    n = n + 1;
                }
            }
            let g = naturals();
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert output == ["0", "1", "2", "3", "4"]

    def test_infinite_fibonacci(self):
        _, output = run("""
            fn fib() {
                let a = 0;
                let b = 1;
                while (true) {
                    yield a;
                    let temp = a + b;
                    a = b;
                    b = temp;
                }
            }
            let g = fib();
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert output == ["0", "1", "1", "2", "3", "5", "8"]

    def test_infinite_repeat(self):
        _, output = run("""
            fn repeat(val) {
                while (true) {
                    yield val;
                }
            }
            let g = repeat("hi");
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert output == ["hi", "hi", "hi"]


# ============================================================
# 6. Generator with Closures
# ============================================================

class TestGeneratorClosures:
    def test_generator_captures_closure(self):
        _, output = run("""
            fn make_gen() {
                let x = 100;
                fn gen() {
                    yield x;
                    yield x + 1;
                }
                return gen();
            }
            let g = make_gen();
            print(next(g));
            print(next(g));
        """)
        assert output == ["100", "101"]

    def test_generator_as_closure_state(self):
        _, output = run("""
            fn counter_gen(start) {
                let n = start;
                while (true) {
                    yield n;
                    n = n + 1;
                }
            }
            let g = counter_gen(5);
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert output == ["5", "6", "7"]

    def test_generator_lambda(self):
        _, output = run("""
            let gen = fn() {
                yield 1;
                yield 2;
            };
            let g = gen();
            print(next(g));
            print(next(g));
        """)
        assert output == ["1", "2"]


# ============================================================
# 7. For-In with Generators
# ============================================================

class TestForInGenerators:
    def test_for_in_generator(self):
        _, output = run("""
            fn three() {
                yield 10;
                yield 20;
                yield 30;
            }
            for (x in three()) {
                print(x);
            }
        """)
        assert output == ["10", "20", "30"]

    def test_for_in_generator_with_computation(self):
        _, output = run("""
            fn squares(n) {
                let i = 0;
                while (i < n) {
                    yield i * i;
                    i = i + 1;
                }
            }
            let total = 0;
            for (s in squares(4)) {
                total = total + s;
            }
            print(total);
        """)
        # 0 + 1 + 4 + 9 = 14
        assert output == ["14"]

    def test_for_in_generator_empty(self):
        _, output = run("""
            fn empty() {
                if (false) {
                    yield 1;
                }
            }
            for (x in empty()) {
                print(x);
            }
            print("done");
        """)
        assert output == ["done"]

    def test_for_in_generator_break(self):
        """Infinite generators use while+next pattern (for-in eagerly collects)."""
        _, output = run("""
            fn naturals() {
                let n = 0;
                while (true) {
                    yield n;
                    n = n + 1;
                }
            }
            let g = naturals();
            let x = next(g);
            while (x < 3) {
                print(x);
                x = next(g);
            }
        """)
        assert output == ["0", "1", "2"]


# ============================================================
# 8. Generator Composition
# ============================================================

class TestGeneratorComposition:
    def test_generator_yields_from_function(self):
        _, output = run("""
            fn double(x) { return x * 2; }
            fn gen() {
                yield double(1);
                yield double(2);
                yield double(3);
            }
            let g = gen();
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert output == ["2", "4", "6"]

    def test_generator_calling_generator_manually(self):
        _, output = run("""
            fn inner() {
                yield 1;
                yield 2;
            }
            fn outer() {
                let g = inner();
                yield next(g);
                yield next(g);
                yield 3;
            }
            let o = outer();
            print(next(o));
            print(next(o));
            print(next(o));
        """)
        assert output == ["1", "2", "3"]

    def test_generator_passed_as_argument(self):
        _, output = run("""
            fn gen() {
                yield 10;
                yield 20;
            }
            fn consume(g) {
                print(next(g));
                print(next(g));
            }
            consume(gen());
        """)
        assert output == ["10", "20"]

    def test_generator_stored_in_array(self):
        _, output = run("""
            fn gen(x) {
                yield x;
                yield x + 1;
            }
            let gens = [gen(0), gen(10), gen(20)];
            print(next(gens[0]));
            print(next(gens[1]));
            print(next(gens[2]));
            print(next(gens[0]));
        """)
        assert output == ["0", "10", "20", "1"]


# ============================================================
# 9. Practical Patterns
# ============================================================

class TestPracticalPatterns:
    def test_range_generator(self):
        _, output = run("""
            fn gen_range(start, stop) {
                let i = start;
                while (i < stop) {
                    yield i;
                    i = i + 1;
                }
            }
            for (x in gen_range(3, 7)) {
                print(x);
            }
        """)
        assert output == ["3", "4", "5", "6"]

    def test_map_generator(self):
        _, output = run("""
            fn gen_map(arr, f) {
                for (x in arr) {
                    yield f(x);
                }
            }
            let doubled = gen_map([1, 2, 3], fn(x) { return x * 2; });
            for (v in doubled) {
                print(v);
            }
        """)
        assert output == ["2", "4", "6"]

    def test_filter_generator(self):
        _, output = run("""
            fn gen_filter(arr, pred) {
                for (x in arr) {
                    if (pred(x)) {
                        yield x;
                    }
                }
            }
            let evens = gen_filter([1, 2, 3, 4, 5, 6], fn(x) { return x % 2 == 0; });
            for (v in evens) {
                print(v);
            }
        """)
        assert output == ["2", "4", "6"]

    def test_take_pattern(self):
        _, output = run("""
            fn naturals() {
                let n = 0;
                while (true) {
                    yield n;
                    n = n + 1;
                }
            }
            fn take(g, n) {
                let results = [];
                let i = 0;
                while (i < n) {
                    push(results, next(g));
                    i = i + 1;
                }
                return results;
            }
            let first5 = take(naturals(), 5);
            print(first5);
        """)
        assert output == ["[0, 1, 2, 3, 4]"]

    def test_chain_generators(self):
        _, output = run("""
            fn chain(g1, g2) {
                let v = next(g1);
                while (v != "STOP") {
                    yield v;
                    v = next(g1, "STOP");
                }
                v = next(g2);
                while (v != "STOP") {
                    yield v;
                    v = next(g2, "STOP");
                }
            }
            fn a() { yield 1; yield 2; }
            fn b() { yield 3; yield 4; }
            for (x in chain(a(), b())) {
                print(x);
            }
        """)
        assert output == ["1", "2", "3", "4"]

    def test_zip_generators(self):
        _, output = run("""
            fn zip(g1, g2) {
                let v1 = next(g1, "STOP");
                let v2 = next(g2, "STOP");
                while (v1 != "STOP" and v2 != "STOP") {
                    yield [v1, v2];
                    v1 = next(g1, "STOP");
                    v2 = next(g2, "STOP");
                }
            }
            fn nums() { yield 1; yield 2; yield 3; }
            fn letters() { yield "a"; yield "b"; }
            for (pair in zip(nums(), letters())) {
                print(pair);
            }
        """)
        assert output == ["[1, a]", "[2, b]"]


# ============================================================
# 10. Generator State Isolation
# ============================================================

class TestGeneratorStateIsolation:
    def test_generator_env_isolation(self):
        """Each generator instance has its own state."""
        _, output = run("""
            fn counter() {
                let n = 0;
                while (true) {
                    yield n;
                    n = n + 1;
                }
            }
            let g1 = counter();
            let g2 = counter();
            next(g1);
            next(g1);
            next(g1);
            print(next(g1));
            print(next(g2));
        """)
        assert output == ["3", "0"]

    def test_generator_does_not_affect_outer_env(self):
        _, output = run("""
            let x = 100;
            fn gen() {
                let x = 0;
                yield x;
            }
            let g = gen();
            next(g);
            print(x);
        """)
        assert output == ["100"]

    def test_generator_param_isolation(self):
        _, output = run("""
            fn gen(n) {
                yield n;
                n = n + 100;
                yield n;
            }
            let g = gen(5);
            print(next(g));
            print(next(g));
        """)
        assert output == ["5", "105"]


# ============================================================
# 11. Error Handling with Generators
# ============================================================

class TestGeneratorErrors:
    def test_next_on_non_generator(self):
        with pytest.raises(VMError, match="next.*requires generator"):
            run("next(42);")

    def test_yield_outside_generator(self):
        """yield at top level should error."""
        with pytest.raises(VMError, match="yield outside"):
            run("yield 1;")

    def test_generator_throw_caught(self):
        _, output = run("""
            fn gen() {
                yield 1;
                throw "error";
                yield 2;
            }
            let g = gen();
            print(next(g));
            try {
                next(g);
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["1", "Uncaught exception: error"]

    def test_generator_try_catch_inside(self):
        _, output = run("""
            fn gen() {
                try {
                    yield 1;
                    throw "boom";
                } catch (e) {
                    yield e;
                }
            }
            let g = gen();
            print(next(g));
            print(next(g));
        """)
        assert output == ["1", "boom"]

    def test_next_wrong_arg_count(self):
        with pytest.raises(VMError, match="next.*takes 1-2"):
            run("next();")


# ============================================================
# 12. Generator with Data Structures
# ============================================================

class TestGeneratorDataStructures:
    def test_yield_from_array_elements(self):
        _, output = run("""
            fn each_item(arr) {
                let i = 0;
                while (i < len(arr)) {
                    yield arr[i];
                    i = i + 1;
                }
            }
            let g = each_item([100, 200, 300]);
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert output == ["100", "200", "300"]

    def test_yield_hash_entries(self):
        _, output = run("""
            fn gen_entries(h) {
                let ks = keys(h);
                for (k in ks) {
                    yield [k, h[k]];
                }
            }
            let h = {x: 1, y: 2};
            let g = gen_entries(h);
            let e1 = next(g);
            let e2 = next(g);
            // Keys may be in any order, so check both entries exist
            print(len(e1));
            print(len(e2));
        """)
        assert output == ["2", "2"]

    def test_generator_builds_result(self):
        """Collect generator results into array."""
        _, output = run("""
            fn squares(n) {
                let i = 0;
                while (i < n) {
                    yield i * i;
                    i = i + 1;
                }
            }
            let result = [];
            let g = squares(5);
            let v = next(g, "STOP");
            while (v != "STOP") {
                push(result, v);
                v = next(g, "STOP");
            }
            print(result);
        """)
        assert output == ["[0, 1, 4, 9, 16]"]


# ============================================================
# 13. AST Detection
# ============================================================

class TestASTDetection:
    def test_yield_detected_in_body(self):
        ast = parse("fn gen() { yield 1; }")
        fn_decl = ast.stmts[0]
        assert _ast_contains_yield(fn_decl.body) is True

    def test_no_yield_detected(self):
        ast = parse("fn f() { return 1; }")
        fn_decl = ast.stmts[0]
        assert _ast_contains_yield(fn_decl.body) is False

    def test_yield_in_nested_fn_not_detected(self):
        """yield inside a nested fn does not make outer fn a generator."""
        ast = parse("""
            fn outer() {
                fn inner() { yield 1; }
                return inner();
            }
        """)
        outer = ast.stmts[0]
        # The outer body contains a FnDecl, which should NOT propagate yield
        assert _ast_contains_yield(outer.body) is False

    def test_yield_in_if_detected(self):
        ast = parse("fn g() { if (true) { yield 1; } }")
        fn_decl = ast.stmts[0]
        assert _ast_contains_yield(fn_decl.body) is True

    def test_yield_in_while_detected(self):
        ast = parse("fn g() { while (true) { yield 1; } }")
        fn_decl = ast.stmts[0]
        assert _ast_contains_yield(fn_decl.body) is True

    def test_yield_token(self):
        tokens = lex("yield 1;")
        assert tokens[0].type == TokenType.YIELD

    def test_yield_ast_node(self):
        ast = parse("fn g() { yield 42; }")
        fn_body = ast.stmts[0].body
        yield_expr = fn_body.stmts[0]
        assert isinstance(yield_expr, YieldExpr)
        assert isinstance(yield_expr.value, type(parse("42;").stmts[0]))


# ============================================================
# 14. Compiler and Bytecode
# ============================================================

class TestCompilerBytecode:
    def test_generator_fn_has_flag(self):
        chunk, compiler = compile_source("""
            fn gen() { yield 1; }
        """)
        # The fn object should have is_generator = True
        gen_fn = None
        for c in chunk.constants:
            if isinstance(c, FnObject) and c.name == 'gen':
                gen_fn = c
                break
        assert gen_fn is not None
        assert gen_fn.is_generator is True

    def test_normal_fn_no_generator_flag(self):
        chunk, compiler = compile_source("""
            fn f() { return 1; }
        """)
        fn_obj = None
        for c in chunk.constants:
            if isinstance(c, FnObject) and c.name == 'f':
                fn_obj = c
                break
        assert fn_obj is not None
        assert fn_obj.is_generator is False

    def test_generator_lambda_has_flag(self):
        chunk, compiler = compile_source("""
            let g = fn() { yield 1; };
        """)
        fn_obj = None
        for c in chunk.constants:
            if isinstance(c, FnObject) and c.name == '<lambda>':
                fn_obj = c
                break
        assert fn_obj is not None
        assert fn_obj.is_generator is True


# ============================================================
# 15. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_yield_in_deeply_nested_loop(self):
        _, output = run("""
            fn gen() {
                let i = 0;
                while (i < 2) {
                    let j = 0;
                    while (j < 2) {
                        yield i * 10 + j;
                        j = j + 1;
                    }
                    i = i + 1;
                }
            }
            let g = gen();
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g));
            print(next(g, "done"));
        """)
        assert output == ["0", "1", "10", "11", "done"]

    def test_generator_with_string_operations(self):
        _, output = run("""
            fn words() {
                let sentence = "hello world foo";
                yield "hello";
                yield "world";
                yield "foo";
            }
            for (w in words()) {
                print(w);
            }
        """)
        assert output == ["hello", "world", "foo"]

    def test_generator_with_boolean_yield(self):
        _, output = run("""
            fn bools() {
                yield true;
                yield false;
                yield true;
            }
            let g = bools();
            print(next(g));
            print(next(g));
            print(next(g));
        """)
        assert output == ["true", "false", "true"]

    def test_many_yields(self):
        _, output = run("""
            fn many() {
                let i = 0;
                while (i < 50) {
                    yield i;
                    i = i + 1;
                }
            }
            let g = many();
            let sum = 0;
            let v = next(g, -1);
            while (v != -1) {
                sum = sum + v;
                v = next(g, -1);
            }
            print(sum);
        """)
        # sum of 0..49 = 1225
        assert output == ["1225"]

    def test_generator_return_value_discarded(self):
        """Return value of a generator is not accessible via next()."""
        _, output = run("""
            fn gen() {
                yield 1;
                return 999;
            }
            let g = gen();
            print(next(g));
            print(next(g));
        """)
        assert output == ["1", "None"]

    def test_generator_with_closures_and_mutation(self):
        _, output = run("""
            fn gen() {
                let arr = [];
                yield arr;
                push(arr, 1);
                yield arr;
                push(arr, 2);
                yield arr;
            }
            let g = gen();
            let a1 = next(g);
            next(g);
            next(g);
            // a1 is the same array object, mutated by the generator
            print(a1);
        """)
        assert output == ["[1, 2]"]


# ============================================================
# 16. Module Integration
# ============================================================

class TestModuleIntegration:
    def test_export_generator(self):
        registry = ModuleRegistry()
        registry.register("gen_mod", """
            export fn counter() {
                let n = 0;
                while (true) {
                    yield n;
                    n = n + 1;
                }
            }
        """)
        _, output = run("""
            import "gen_mod";
            let g = counter();
            print(next(g));
            print(next(g));
            print(next(g));
        """, registry=registry)
        assert output == ["0", "1", "2"]

    def test_import_and_use_generator(self):
        registry = ModuleRegistry()
        registry.register("utils", """
            export fn fibonacci() {
                let a = 0;
                let b = 1;
                while (true) {
                    yield a;
                    let temp = a + b;
                    a = b;
                    b = temp;
                }
            }
        """)
        _, output = run("""
            import { fibonacci } from "utils";
            let g = fibonacci();
            let results = [];
            let i = 0;
            while (i < 8) {
                push(results, next(g));
                i = i + 1;
            }
            print(results);
        """, registry=registry)
        assert output == ["[0, 1, 1, 2, 3, 5, 8, 13]"]


# ============================================================
# 17. Backward Compatibility
# ============================================================

class TestBackwardCompatibility:
    def test_basic_arithmetic(self):
        r = execute("1 + 2;")
        assert r['result'] == 3

    def test_functions_still_work(self):
        _, output = run("""
            fn add(a, b) { return a + b; }
            print(add(3, 4));
        """)
        assert output == ["7"]

    def test_closures_still_work(self):
        _, output = run("""
            fn make_adder(x) {
                return fn(y) { return x + y; };
            }
            let add5 = make_adder(5);
            print(add5(3));
        """)
        assert output == ["8"]

    def test_arrays_still_work(self):
        r = execute("let a = [1, 2, 3]; a[1];")
        assert r['result'] == 2

    def test_hash_maps_still_work(self):
        r = execute('let h = {x: 1}; h.x;')
        assert r['result'] == 1

    def test_for_in_still_works(self):
        _, output = run("""
            for (x in [1, 2, 3]) {
                print(x);
            }
        """)
        assert output == ["1", "2", "3"]

    def test_try_catch_still_works(self):
        _, output = run("""
            try {
                throw "error";
            } catch (e) {
                print(e);
            }
        """)
        assert output == ["error"]

    def test_modules_still_work(self):
        registry = ModuleRegistry()
        registry.register("math", """
            export fn double(x) { return x * 2; }
        """)
        _, output = run("""
            import { double } from "math";
            print(double(21));
        """, registry=registry)
        assert output == ["42"]

    def test_while_loop(self):
        _, output = run("""
            let i = 0;
            while (i < 3) {
                print(i);
                i = i + 1;
            }
        """)
        assert output == ["0", "1", "2"]

    def test_if_else(self):
        _, output = run("""
            if (true) {
                print("yes");
            } else {
                print("no");
            }
        """)
        assert output == ["yes"]

    def test_recursive_function(self):
        _, output = run("""
            fn fact(n) {
                if (n <= 1) { return 1; }
                return n * fact(n - 1);
            }
            print(fact(5));
        """)
        assert output == ["120"]

    def test_higher_order_functions(self):
        _, output = run("""
            let arr = [1, 2, 3, 4, 5];
            let doubled = map(arr, fn(x) { return x * 2; });
            print(doubled);
        """)
        assert output == ["[2, 4, 6, 8, 10]"]


# ============================================================
# 18. Complex Scenarios
# ============================================================

class TestComplexScenarios:
    def test_generator_pipeline(self):
        """Chain of generator transformations."""
        _, output = run("""
            fn numbers() {
                let i = 1;
                while (i <= 10) {
                    yield i;
                    i = i + 1;
                }
            }

            fn even_only(g) {
                let v = next(g, "STOP");
                while (v != "STOP") {
                    if (v % 2 == 0) {
                        yield v;
                    }
                    v = next(g, "STOP");
                }
            }

            fn doubled(g) {
                let v = next(g, "STOP");
                while (v != "STOP") {
                    yield v * 2;
                    v = next(g, "STOP");
                }
            }

            let pipeline = doubled(even_only(numbers()));
            let result = [];
            for (v in pipeline) {
                push(result, v);
            }
            print(result);
        """)
        # numbers 1..10, filter evens [2,4,6,8,10], double [4,8,12,16,20]
        assert output == ["[4, 8, 12, 16, 20]"]

    def test_generator_interleave(self):
        _, output = run("""
            fn interleave(g1, g2) {
                let v1 = next(g1, "STOP");
                let v2 = next(g2, "STOP");
                while (v1 != "STOP" or v2 != "STOP") {
                    if (v1 != "STOP") {
                        yield v1;
                        v1 = next(g1, "STOP");
                    }
                    if (v2 != "STOP") {
                        yield v2;
                        v2 = next(g2, "STOP");
                    }
                }
            }
            fn a() { yield 1; yield 3; yield 5; }
            fn b() { yield 2; yield 4; }
            let result = [];
            for (v in interleave(a(), b())) {
                push(result, v);
            }
            print(result);
        """)
        assert output == ["[1, 2, 3, 4, 5]"]

    def test_flatten_generator(self):
        _, output = run("""
            fn flatten(arrays) {
                for (arr in arrays) {
                    for (item in arr) {
                        yield item;
                    }
                }
            }
            let nested = [[1, 2], [3], [4, 5, 6]];
            let result = [];
            for (v in flatten(nested)) {
                push(result, v);
            }
            print(result);
        """)
        assert output == ["[1, 2, 3, 4, 5, 6]"]

    def test_accumulating_generator(self):
        _, output = run("""
            fn running_sum(arr) {
                let total = 0;
                for (x in arr) {
                    total = total + x;
                    yield total;
                }
            }
            let result = [];
            for (v in running_sum([1, 2, 3, 4, 5])) {
                push(result, v);
            }
            print(result);
        """)
        assert output == ["[1, 3, 6, 10, 15]"]

    def test_generator_as_state_machine(self):
        _, output = run("""
            fn traffic_light() {
                while (true) {
                    yield "green";
                    yield "yellow";
                    yield "red";
                }
            }
            let light = traffic_light();
            let i = 0;
            while (i < 7) {
                print(next(light));
                i = i + 1;
            }
        """)
        assert output == ["green", "yellow", "red", "green", "yellow", "red", "green"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
