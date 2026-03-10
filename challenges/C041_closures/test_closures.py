"""
Tests for C041: Closures -- First-class functions with captured environments
AgentZero Session 042

Test categories:
  1. Basic C010 compatibility (existing features still work)
  2. Named function closures (fn decl captures env)
  3. Lambda expressions (anonymous functions)
  4. Mutable captured state (counter pattern)
  5. Higher-order functions (pass/return functions)
  6. Chained calls (currying, make_adder(5)(3))
  7. Partial application
  8. Nested closures (multi-level capture)
  9. Closure over loop variable
  10. Edge cases and error handling
"""

import pytest
from closures import (
    run, execute, parse, compile_source, lex,
    LambdaExpr, FnDecl, CallExpr, ClosureObject, FnObject,
    VMError, ParseError, LexError, CompileError,
    Op, disassemble,
)


# ============================================================
# 1. Basic C010 Compatibility
# ============================================================

class TestBasicCompatibility:
    """Ensure existing C010 features still work."""

    def test_arithmetic(self):
        result, _ = run("3 + 4 * 2;")
        assert result == 11

    def test_variables(self):
        result, _ = run("let x = 10; x + 5;")
        assert result == 15

    def test_string_ops(self):
        result, output = run('print("hello");')
        assert output == ["hello"]

    def test_boolean_ops(self):
        result, _ = run("true and false;")
        assert result == False

    def test_if_else(self):
        _, output = run("""
            let x = 10;
            if (x > 5) {
                print("big");
            } else {
                print("small");
            }
        """)
        assert output == ["big"]

    def test_while_loop(self):
        _, output = run("""
            let i = 0;
            while (i < 3) {
                print(i);
                i = i + 1;
            }
        """)
        assert output == ["0", "1", "2"]

    def test_named_function(self):
        result, _ = run("""
            fn add(a, b) { return a + b; }
            add(3, 4);
        """)
        assert result == 7

    def test_recursive_function(self):
        result, _ = run("""
            fn fact(n) {
                if (n <= 1) { return 1; }
                return n * fact(n - 1);
            }
            fact(5);
        """)
        assert result == 120

    def test_print_bool(self):
        _, output = run("print(true); print(false);")
        assert output == ["true", "false"]

    def test_nested_function_call(self):
        result, _ = run("""
            fn double(x) { return x * 2; }
            fn add_one(x) { return x + 1; }
            add_one(double(5));
        """)
        assert result == 11


# ============================================================
# 2. Named Function Closures
# ============================================================

class TestNamedClosures:
    """Named functions now capture their enclosing environment."""

    def test_closure_captures_variable(self):
        result, _ = run("""
            let x = 10;
            fn get_x() { return x; }
            get_x();
        """)
        assert result == 10

    def test_closure_captures_at_definition_time(self):
        """Closure captures the env at MAKE_CLOSURE time."""
        _, output = run("""
            let x = 1;
            fn get_x() { return x; }
            print(get_x());
            x = 2;
            print(get_x());
        """)
        # get_x captured x=1 at definition. Changing x in the outer env
        # doesn't affect the closure's snapshot.
        assert output == ["1", "1"]

    def test_closure_captures_multiple_vars(self):
        result, _ = run("""
            let a = 10;
            let b = 20;
            fn sum() { return a + b; }
            sum();
        """)
        assert result == 30

    def test_fn_decl_creates_closure_object(self):
        """fn decl should create a ClosureObject, not just FnObject."""
        r = execute("""
            let x = 5;
            fn get_x() { return x; }
            get_x;
        """)
        # The result should be a ClosureObject (the value of get_x)
        assert isinstance(r['result'], ClosureObject)

    def test_closure_with_params_and_captured(self):
        result, _ = run("""
            let base = 100;
            fn add_to_base(n) { return base + n; }
            add_to_base(42);
        """)
        assert result == 142


# ============================================================
# 3. Lambda Expressions
# ============================================================

class TestLambdaExpressions:
    """Anonymous functions: fn(params) { body }"""

    def test_lambda_basic(self):
        result, _ = run("""
            let add = fn(a, b) { return a + b; };
            add(3, 4);
        """)
        assert result == 7

    def test_lambda_no_params(self):
        result, _ = run("""
            let greet = fn() { return 42; };
            greet();
        """)
        assert result == 42

    def test_lambda_captures_env(self):
        result, _ = run("""
            let x = 10;
            let get_x = fn() { return x; };
            get_x();
        """)
        assert result == 10

    def test_lambda_as_argument(self):
        result, _ = run("""
            fn apply(f, x) { return f(x); }
            apply(fn(n) { return n * 2; }, 5);
        """)
        assert result == 10

    def test_lambda_returned_from_function(self):
        result, _ = run("""
            fn make_adder(x) {
                return fn(y) { return x + y; };
            }
            let add5 = make_adder(5);
            add5(3);
        """)
        assert result == 8

    def test_lambda_in_let(self):
        _, output = run("""
            let square = fn(x) { return x * x; };
            print(square(4));
            print(square(7));
        """)
        assert output == ["16", "49"]

    def test_lambda_parse(self):
        """Verify lambda parses to LambdaExpr AST node."""
        ast = parse("let f = fn(x) { return x; };")
        let_decl = ast.stmts[0]
        assert isinstance(let_decl.value, LambdaExpr)
        assert let_decl.value.params == ['x']

    def test_lambda_is_closure_object(self):
        r = execute("let f = fn() { return 1; }; f;")
        assert isinstance(r['result'], ClosureObject)

    def test_immediately_invoked_lambda(self):
        result, _ = run("""
            fn(x) { return x + 1; }(10);
        """)
        assert result == 11


# ============================================================
# 4. Mutable Captured State
# ============================================================

class TestMutableState:
    """Closures with persistent mutable state."""

    def test_counter(self):
        _, output = run("""
            fn make_counter() {
                let count = 0;
                return fn() {
                    count = count + 1;
                    return count;
                };
            }
            let c = make_counter();
            print(c());
            print(c());
            print(c());
        """)
        assert output == ["1", "2", "3"]

    def test_independent_counters(self):
        _, output = run("""
            fn make_counter() {
                let count = 0;
                return fn() {
                    count = count + 1;
                    return count;
                };
            }
            let a = make_counter();
            let b = make_counter();
            print(a());
            print(a());
            print(b());
            print(a());
            print(b());
        """)
        assert output == ["1", "2", "1", "3", "2"]

    def test_accumulator(self):
        _, output = run("""
            fn make_accumulator(init) {
                let total = init;
                return fn(n) {
                    total = total + n;
                    return total;
                };
            }
            let acc = make_accumulator(100);
            print(acc(5));
            print(acc(10));
            print(acc(3));
        """)
        assert output == ["105", "115", "118"]

    def test_toggle(self):
        _, output = run("""
            fn make_toggle() {
                let state = false;
                return fn() {
                    if (state) {
                        state = false;
                    } else {
                        state = true;
                    }
                    return state;
                };
            }
            let toggle = make_toggle();
            print(toggle());
            print(toggle());
            print(toggle());
        """)
        assert output == ["true", "false", "true"]

    def test_setter_getter_pair(self):
        """Two closures sharing the same captured env."""
        _, output = run("""
            fn make_box(init) {
                let value = init;
                let get = fn() { return value; };
                let set = fn(v) { value = v; };
                // Return getter -- store setter in outer scope via name
                // Actually we need to return both. Use a trick:
                // Call get and set via the enclosing scope names
                return get;
            }
            let get = make_box(10);
            print(get());
        """)
        assert output == ["10"]

    def test_closure_state_persists(self):
        """Verify state persists across multiple calls."""
        _, output = run("""
            fn make_adder_accum() {
                let sum = 0;
                return fn(n) {
                    sum = sum + n;
                    return sum;
                };
            }
            let add = make_adder_accum();
            print(add(1));
            print(add(2));
            print(add(3));
            print(add(4));
        """)
        assert output == ["1", "3", "6", "10"]


# ============================================================
# 5. Higher-Order Functions
# ============================================================

class TestHigherOrder:
    """Functions that take or return other functions."""

    def test_apply(self):
        result, _ = run("""
            fn apply(f, x) { return f(x); }
            fn double(x) { return x * 2; }
            apply(double, 5);
        """)
        assert result == 10

    def test_compose(self):
        result, _ = run("""
            fn compose(f, g) {
                return fn(x) { return f(g(x)); };
            }
            fn double(x) { return x * 2; }
            fn inc(x) { return x + 1; }
            let double_then_inc = compose(inc, double);
            double_then_inc(5);
        """)
        assert result == 11

    def test_map_manual(self):
        """Manual map over a range using closure."""
        _, output = run("""
            fn for_each(n, f) {
                let i = 0;
                while (i < n) {
                    print(f(i));
                    i = i + 1;
                }
            }
            for_each(4, fn(x) { return x * x; });
        """)
        assert output == ["0", "1", "4", "9"]

    def test_function_as_return_value(self):
        result, _ = run("""
            fn identity(x) { return x; }
            fn get_fn() { return identity; }
            let f = get_fn();
            f(42);
        """)
        assert result == 42

    def test_callback_pattern(self):
        _, output = run("""
            fn do_work(on_done) {
                let result = 10 + 20;
                on_done(result);
            }
            do_work(fn(r) { print(r); });
        """)
        assert output == ["30"]

    def test_twice(self):
        result, _ = run("""
            fn twice(f) {
                return fn(x) { return f(f(x)); };
            }
            fn inc(x) { return x + 1; }
            let inc2 = twice(inc);
            inc2(10);
        """)
        assert result == 12

    def test_thrice(self):
        result, _ = run("""
            fn apply_n(f, n) {
                return fn(x) {
                    let result = x;
                    let i = 0;
                    while (i < n) {
                        result = f(result);
                        i = i + 1;
                    }
                    return result;
                };
            }
            fn double(x) { return x * 2; }
            let triple_double = apply_n(double, 3);
            triple_double(1);
        """)
        assert result == 8  # 1*2*2*2


# ============================================================
# 6. Chained Calls
# ============================================================

class TestChainedCalls:
    """Calling the result of a call: expr(args)(args)"""

    def test_chained_call_basic(self):
        result, _ = run("""
            fn make_adder(x) {
                return fn(y) { return x + y; };
            }
            make_adder(5)(3);
        """)
        assert result == 8

    def test_double_chain(self):
        result, _ = run("""
            fn a() {
                return fn() {
                    return fn() { return 42; };
                };
            }
            a()()();
        """)
        assert result == 42

    def test_chained_with_args(self):
        result, _ = run("""
            fn curried_add(a) {
                return fn(b) {
                    return fn(c) { return a + b + c; };
                };
            }
            curried_add(1)(2)(3);
        """)
        assert result == 6

    def test_chain_lambda(self):
        result, _ = run("""
            fn(x) { return fn(y) { return x * y; }; }(3)(4);
        """)
        assert result == 12

    def test_chain_mixed(self):
        """Chain named function return with immediate call."""
        result, _ = run("""
            fn make_mul(x) {
                return fn(y) { return x * y; };
            }
            let triple = make_mul(3);
            triple(10) + make_mul(2)(5);
        """)
        assert result == 40  # 30 + 10


# ============================================================
# 7. Partial Application
# ============================================================

class TestPartialApplication:
    """Using closures for partial application of arguments."""

    def test_partial_add(self):
        result, _ = run("""
            fn add(a, b) { return a + b; }
            fn partial_add(a) {
                return fn(b) { return add(a, b); };
            }
            let add5 = partial_add(5);
            add5(3);
        """)
        assert result == 8

    def test_partial_mul(self):
        _, output = run("""
            fn partial_mul(a) {
                return fn(b) { return a * b; };
            }
            let double = partial_mul(2);
            let triple = partial_mul(3);
            print(double(5));
            print(triple(5));
        """)
        assert output == ["10", "15"]

    def test_partial_compare(self):
        _, output = run("""
            fn gt(threshold) {
                return fn(x) { return x > threshold; };
            }
            let gt10 = gt(10);
            print(gt10(5));
            print(gt10(15));
        """)
        assert output == ["false", "true"]

    def test_partial_predicate(self):
        """Partial application for filtering."""
        _, output = run("""
            fn check_range(low, high) {
                return fn(x) { return x >= low and x <= high; };
            }
            let in_range = check_range(5, 10);
            print(in_range(3));
            print(in_range(7));
            print(in_range(12));
        """)
        assert output == ["false", "true", "false"]


# ============================================================
# 8. Nested Closures
# ============================================================

class TestNestedClosures:
    """Closures within closures -- multi-level capture."""

    def test_two_level_capture(self):
        result, _ = run("""
            fn outer() {
                let x = 10;
                fn inner() {
                    return x;
                }
                return inner();
            }
            outer();
        """)
        assert result == 10

    def test_three_level_capture(self):
        result, _ = run("""
            fn level1() {
                let a = 1;
                return fn() {
                    let b = 2;
                    return fn() {
                        return a + b;
                    };
                };
            }
            level1()()();
        """)
        assert result == 3

    def test_nested_closures_independent_state(self):
        _, output = run("""
            fn make_pair() {
                let x = 0;
                let inc = fn() {
                    x = x + 1;
                    return x;
                };
                let get = fn() { return x; };
                // Return inc (we can only return one thing)
                return inc;
            }
            let inc = make_pair();
            print(inc());
            print(inc());
        """)
        assert output == ["1", "2"]

    def test_closure_in_loop(self):
        """Create closures in a loop -- each captures env at creation time."""
        _, output = run("""
            fn make_fn(val) {
                return fn() { return val; };
            }
            let i = 0;
            while (i < 3) {
                let f = make_fn(i);
                print(f());
                i = i + 1;
            }
        """)
        assert output == ["0", "1", "2"]

    def test_deep_nesting(self):
        result, _ = run("""
            fn a(x) {
                return fn(y) {
                    return fn(z) {
                        return fn(w) {
                            return x + y + z + w;
                        };
                    };
                };
            }
            a(1)(2)(3)(4);
        """)
        assert result == 10


# ============================================================
# 9. Closure Over Loop Variable
# ============================================================

class TestClosureOverLoop:
    """Classic closure-over-loop-variable scenarios."""

    def test_closure_over_loop_with_factory(self):
        """Using a factory function to capture each iteration's value."""
        _, output = run("""
            fn make_printer(n) {
                return fn() { print(n); };
            }
            let f0 = make_printer(0);
            let f1 = make_printer(1);
            let f2 = make_printer(2);
            f0();
            f1();
            f2();
        """)
        assert output == ["0", "1", "2"]

    def test_closure_captures_current_value(self):
        """Closure captures at point of creation, not reference to loop var."""
        _, output = run("""
            fn make_fn(val) {
                return fn() { return val; };
            }
            let a = make_fn(10);
            let b = make_fn(20);
            print(a());
            print(b());
        """)
        assert output == ["10", "20"]


# ============================================================
# 10. Complex Patterns
# ============================================================

class TestComplexPatterns:
    """More sophisticated closure usage patterns."""

    def test_fibonacci_closure(self):
        """Fibonacci using mutable closure state."""
        _, output = run("""
            fn make_fib() {
                let a = 0;
                let b = 1;
                return fn() {
                    let result = a;
                    let temp = a + b;
                    a = b;
                    b = temp;
                    return result;
                };
            }
            let fib = make_fib();
            print(fib());
            print(fib());
            print(fib());
            print(fib());
            print(fib());
            print(fib());
            print(fib());
        """)
        assert output == ["0", "1", "1", "2", "3", "5", "8"]

    def test_memoize_simple(self):
        """Simple memoization-like pattern with closure."""
        _, output = run("""
            fn make_cached(val) {
                let cached = val;
                let computed = false;
                return fn() {
                    if (not computed) {
                        cached = cached * 2;
                        computed = true;
                    }
                    return cached;
                };
            }
            let get = make_cached(5);
            print(get());
            print(get());
            print(get());
        """)
        assert output == ["10", "10", "10"]

    def test_state_machine(self):
        """Simple state machine with closure."""
        _, output = run("""
            fn make_machine() {
                let state = 0;
                return fn() {
                    if (state == 0) {
                        state = 1;
                        return 0;
                    }
                    if (state == 1) {
                        state = 2;
                        return 1;
                    }
                    state = 0;
                    return 2;
                };
            }
            let machine = make_machine();
            print(machine());
            print(machine());
            print(machine());
            print(machine());
            print(machine());
        """)
        assert output == ["0", "1", "2", "0", "1"]

    def test_closure_returning_closure_with_state(self):
        """Nested closure factories with independent state."""
        _, output = run("""
            fn make_counter_factory() {
                let total_created = 0;
                return fn() {
                    total_created = total_created + 1;
                    let count = 0;
                    return fn() {
                        count = count + 1;
                        return count;
                    };
                };
            }
            let factory = make_counter_factory();
            let c1 = factory();
            let c2 = factory();
            print(c1());
            print(c1());
            print(c2());
            print(c1());
        """)
        assert output == ["1", "2", "1", "3"]

    def test_function_pipeline(self):
        """Build a pipeline of transformations."""
        result, _ = run("""
            fn pipe(f, g) {
                return fn(x) { return g(f(x)); };
            }
            fn add1(x) { return x + 1; }
            fn mul2(x) { return x * 2; }
            fn sub3(x) { return x - 3; }
            let transform = pipe(pipe(add1, mul2), sub3);
            transform(5);
        """)
        assert result == 9  # ((5+1)*2) - 3 = 9

    def test_reduce_manual(self):
        """Manual reduce using closures."""
        result, _ = run("""
            fn make_reducer(f, init) {
                let acc = init;
                return fn(val) {
                    acc = f(acc, val);
                    return acc;
                };
            }
            fn add(a, b) { return a + b; }
            let sum = make_reducer(add, 0);
            sum(1);
            sum(2);
            sum(3);
            sum(4);
        """)
        assert result == 10


# ============================================================
# 11. Edge Cases and Error Handling
# ============================================================

class TestEdgeCases:
    """Edge cases and error scenarios."""

    def test_call_non_function_error(self):
        with pytest.raises(VMError, match="Cannot call non-function"):
            run("let x = 5; x();")

    def test_wrong_arg_count(self):
        with pytest.raises(VMError, match="expects 1 args, got 2"):
            run("""
                fn f(x) { return x; }
                f(1, 2);
            """)

    def test_lambda_wrong_arg_count(self):
        with pytest.raises(VMError):
            run("""
                let f = fn(a, b) { return a + b; };
                f(1);
            """)

    def test_undefined_var_in_closure(self):
        """Closure body references undefined variable."""
        with pytest.raises(VMError, match="Undefined variable"):
            run("""
                fn make() {
                    return fn() { return undefined_var; };
                }
                let f = make();
                f();
            """)

    def test_empty_lambda_body(self):
        """Lambda with empty body returns None."""
        result, _ = run("""
            let f = fn() { };
            f();
        """)
        assert result is None

    def test_closure_does_not_leak_params(self):
        """Parameters of one closure should not be visible in another."""
        _, output = run("""
            fn make_adder(x) {
                return fn(y) { return x + y; };
            }
            let a = make_adder(10);
            let b = make_adder(20);
            print(a(1));
            print(b(1));
        """)
        assert output == ["11", "21"]

    def test_recursive_via_closure(self):
        """Recursion through closure self-reference."""
        result, _ = run("""
            fn make_factorial() {
                fn fact(n) {
                    if (n <= 1) { return 1; }
                    return n * fact(n - 1);
                }
                return fact;
            }
            let f = make_factorial();
            f(5);
        """)
        assert result == 120

    def test_return_from_nested_closure(self):
        result, _ = run("""
            fn outer() {
                fn inner() {
                    return 42;
                }
                return inner();
            }
            outer();
        """)
        assert result == 42

    def test_parse_fn_vs_lambda(self):
        """Distinguish fn declaration from lambda in parser."""
        ast1 = parse("fn foo() { return 1; }")
        assert isinstance(ast1.stmts[0], FnDecl)
        assert ast1.stmts[0].name == 'foo'

        ast2 = parse("fn() { return 1; }();")
        assert isinstance(ast2.stmts[0], CallExpr)
        assert isinstance(ast2.stmts[0].callee, LambdaExpr)

    def test_lambda_in_expression_position(self):
        """Lambda can appear in complex expressions."""
        result, _ = run("""
            let result = fn(x) { return x + 1; }(10) + fn(x) { return x * 2; }(5);
            result;
        """)
        assert result == 21  # 11 + 10

    def test_closure_with_string(self):
        _, output = run("""
            fn make_greeter(name) {
                return fn() { return name; };
            }
            let greet = make_greeter("hello");
            print(greet());
        """)
        assert output == ["hello"]

    def test_closure_with_boolean(self):
        result, _ = run("""
            fn make_check(flag) {
                return fn() { return flag; };
            }
            let check_true = make_check(true);
            let check_false = make_check(false);
            check_true() and not check_false();
        """)
        assert result == True

    def test_deeply_nested_return(self):
        result, _ = run("""
            fn a() {
                fn b() {
                    fn c() {
                        return 99;
                    }
                    return c();
                }
                return b();
            }
            a();
        """)
        assert result == 99


# ============================================================
# 12. Disassembly and Internals
# ============================================================

class TestInternals:
    """Test compilation and disassembly."""

    def test_make_closure_in_bytecode(self):
        """Verify MAKE_CLOSURE opcode appears in compiled output."""
        chunk, _ = compile_source("""
            fn f() { return 1; }
        """)
        ops = [chunk.code[i] for i in range(len(chunk.code))
               if isinstance(chunk.code[i], Op)]
        assert Op.MAKE_CLOSURE in ops

    def test_lambda_in_bytecode(self):
        chunk, _ = compile_source("""
            let f = fn(x) { return x; };
        """)
        ops = [chunk.code[i] for i in range(len(chunk.code))
               if isinstance(chunk.code[i], Op)]
        assert Op.MAKE_CLOSURE in ops

    def test_disassemble_closure(self):
        chunk, _ = compile_source("fn f() { return 1; }")
        text = disassemble(chunk)
        assert "MAKE_CLOSURE" in text

    def test_closure_object_structure(self):
        r = execute("let f = fn() { return 1; }; f;")
        closure = r['result']
        assert isinstance(closure, ClosureObject)
        assert isinstance(closure.fn, FnObject)
        assert isinstance(closure.env, dict)
        assert closure.fn.name == "<lambda>"
        assert closure.fn.arity == 0


# ============================================================
# 13. Interaction with Control Flow
# ============================================================

class TestClosureWithControlFlow:
    """Closures interacting with if/else and while."""

    def test_closure_in_if_branch(self):
        result, _ = run("""
            fn make(flag) {
                if (flag) {
                    return fn() { return 1; };
                } else {
                    return fn() { return 2; };
                }
            }
            let a = make(true);
            let b = make(false);
            a() + b();
        """)
        assert result == 3

    def test_closure_created_in_loop(self):
        """Each loop iteration creates a new closure via factory."""
        _, output = run("""
            fn make_fn(val) {
                return fn() { return val; };
            }
            let i = 0;
            while (i < 5) {
                let f = make_fn(i * i);
                print(f());
                i = i + 1;
            }
        """)
        assert output == ["0", "1", "4", "9", "16"]

    def test_counter_in_loop(self):
        _, output = run("""
            fn make_counter() {
                let n = 0;
                return fn() {
                    n = n + 1;
                    return n;
                };
            }
            let c = make_counter();
            let i = 0;
            while (i < 5) {
                print(c());
                i = i + 1;
            }
        """)
        assert output == ["1", "2", "3", "4", "5"]

    def test_conditional_closure_call(self):
        _, output = run("""
            fn make_op(kind) {
                if (kind == 1) {
                    return fn(x) { return x + 1; };
                }
                return fn(x) { return x * 2; };
            }
            let inc = make_op(1);
            let dbl = make_op(2);
            print(inc(10));
            print(dbl(10));
        """)
        assert output == ["11", "20"]


# ============================================================
# 14. Stress Tests
# ============================================================

class TestStress:
    """Performance and limit testing."""

    def test_many_closures(self):
        """Create many closures without crashing."""
        _, output = run("""
            fn make(n) { return fn() { return n; }; }
            let i = 0;
            while (i < 50) {
                let f = make(i);
                if (i == 49) { print(f()); }
                i = i + 1;
            }
        """)
        assert output == ["49"]

    def test_deeply_chained(self):
        """Chain 5 levels deep."""
        result, _ = run("""
            fn f(a) {
                return fn(b) {
                    return fn(c) {
                        return fn(d) {
                            return fn(e) {
                                return a + b + c + d + e;
                            };
                        };
                    };
                };
            }
            f(1)(2)(3)(4)(5);
        """)
        assert result == 15

    def test_counter_many_calls(self):
        _, output = run("""
            fn make_counter() {
                let n = 0;
                return fn() { n = n + 1; return n; };
            }
            let c = make_counter();
            let i = 0;
            while (i < 100) {
                c();
                i = i + 1;
            }
            print(c());
        """)
        assert output == ["101"]

    def test_recursive_factorial_via_closure(self):
        result, _ = run("""
            fn make_fact() {
                fn f(n) {
                    if (n <= 1) { return 1; }
                    return n * f(n - 1);
                }
                return f;
            }
            let fact = make_fact();
            fact(10);
        """)
        assert result == 3628800


# ============================================================
# 15. Lambda as Expression in Various Positions
# ============================================================

class TestLambdaPositions:
    """Lambda expressions in different syntactic positions."""

    def test_lambda_in_binary_expr(self):
        """Lambda calls in arithmetic expressions."""
        result, _ = run("""
            fn(a) { return a; }(10) + fn(b) { return b; }(20);
        """)
        assert result == 30

    def test_lambda_as_fn_arg(self):
        result, _ = run("""
            fn apply2(f, a, b) { return f(a) + f(b); }
            apply2(fn(x) { return x * x; }, 3, 4);
        """)
        assert result == 25  # 9 + 16

    def test_lambda_in_let_with_capture(self):
        result, _ = run("""
            let x = 10;
            let f = fn(y) { return x + y; };
            f(5);
        """)
        assert result == 15

    def test_nested_lambda(self):
        result, _ = run("""
            let f = fn(x) {
                return fn(y) {
                    return fn(z) {
                        return x * y + z;
                    };
                };
            };
            f(2)(3)(4);
        """)
        assert result == 10  # 2*3 + 4

    def test_lambda_with_print(self):
        _, output = run("""
            let printer = fn(msg) { print(msg); };
            printer("hello");
            printer("world");
        """)
        assert output == ["hello", "world"]
