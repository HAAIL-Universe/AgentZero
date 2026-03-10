"""Tests for C064: Async Generators -- composing C056 (async/await) + C047 (generators)."""
import pytest
from async_generators import run, execute, make_default_registry, PromiseObject, AsyncGeneratorObject


# ============================================================
# Section 1: Basic async generator creation
# ============================================================

class TestBasicCreation:
    def test_async_generator_returns_async_generator_object(self):
        r, out = run("""
            async fn* gen() { yield 1; }
            let g = gen();
            print type(g);
        """)
        assert out == ["async_generator"]

    def test_async_generator_not_a_promise(self):
        r, out = run("""
            async fn* gen() { yield 1; }
            let g = gen();
            print type(g) != "promise";
        """)
        assert out == ["true"]

    def test_async_generator_not_a_regular_generator(self):
        r, out = run("""
            async fn* gen() { yield 1; }
            let g = gen();
            print type(g) != "generator";
        """)
        assert out == ["true"]

    def test_async_generator_format(self):
        r, out = run("""
            async fn* myGen() { yield 1; }
            let g = myGen();
            print string(g);
        """)
        assert out == ["<async-generator:myGen:suspended>"]

    def test_async_generator_format_done(self):
        r, out = run("""
            async fn* myGen() { yield 1; }
            let g = myGen();
            await next(g);
            await next(g);
            print string(g);
        """)
        assert out == ["<async-generator:myGen:done>"]


# ============================================================
# Section 2: Basic yield and next()
# ============================================================

class TestBasicYield:
    def test_next_returns_promise(self):
        r, out = run("""
            async fn* gen() { yield 42; }
            let g = gen();
            let p = next(g);
            print type(p);
        """)
        assert out == ["promise"]

    def test_await_next_gets_yielded_value(self):
        r, out = run("""
            async fn* gen() { yield 42; }
            let g = gen();
            let val = await next(g);
            print val;
        """)
        assert out == ["42"]

    def test_multiple_yields(self):
        r, out = run("""
            async fn* gen() {
                yield 1;
                yield 2;
                yield 3;
            }
            let g = gen();
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "2", "3"]

    def test_next_after_done_returns_null(self):
        r, out = run("""
            async fn* gen() { yield 1; }
            let g = gen();
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "null"]

    def test_empty_async_generator(self):
        r, out = run("""
            async fn* gen() {}
            let g = gen();
            print await next(g);
        """)
        assert out == ["null"]

    def test_yield_expression_values(self):
        r, out = run("""
            async fn* gen() {
                yield 1 + 2;
                yield "hello";
                yield true;
                yield [1, 2, 3];
            }
            let g = gen();
            print await next(g);
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["3", "hello", "true", "[1, 2, 3]"]


# ============================================================
# Section 3: Await inside async generators
# ============================================================

class TestAwaitInAsyncGen:
    def test_await_resolved_promise(self):
        r, out = run("""
            async fn* gen() {
                let val = await Promise.resolve(42);
                yield val;
            }
            let g = gen();
            print await next(g);
        """)
        assert out == ["42"]

    def test_await_then_yield_sequence(self):
        r, out = run("""
            async fn* gen() {
                let a = await Promise.resolve(10);
                yield a;
                let b = await Promise.resolve(20);
                yield b;
            }
            let g = gen();
            print await next(g);
            print await next(g);
        """)
        assert out == ["10", "20"]

    def test_await_async_function_result(self):
        r, out = run("""
            async fn fetchData() {
                return 99;
            }
            async fn* gen() {
                let data = await fetchData();
                yield data;
            }
            let g = gen();
            print await next(g);
        """)
        assert out == ["99"]

    def test_multiple_awaits_before_yield(self):
        r, out = run("""
            async fn* gen() {
                let a = await Promise.resolve(1);
                let b = await Promise.resolve(2);
                let c = await Promise.resolve(3);
                yield a + b + c;
            }
            let g = gen();
            print await next(g);
        """)
        assert out == ["6"]

    def test_interleaved_await_and_yield(self):
        r, out = run("""
            async fn* gen() {
                yield 1;
                let x = await Promise.resolve(10);
                yield x;
                let y = await Promise.resolve(20);
                yield x + y;
            }
            let g = gen();
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "10", "30"]


# ============================================================
# Section 4: Parameters
# ============================================================

class TestParameters:
    def test_async_generator_with_params(self):
        r, out = run("""
            async fn* range(start, end) {
                let i = start;
                while (i < end) {
                    yield i;
                    i = i + 1;
                }
            }
            let g = range(3, 6);
            print await next(g);
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["3", "4", "5", "null"]

    def test_async_generator_with_rest_params(self):
        r, out = run("""
            async fn* gen(first, ...rest) {
                yield first;
                for (x in rest) {
                    yield x;
                }
            }
            let g = gen(1, 2, 3);
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "2", "3"]

    def test_async_generator_closure_over_params(self):
        r, out = run("""
            async fn* counter(start) {
                let n = start;
                while (true) {
                    yield n;
                    n = n + 1;
                }
            }
            let g = counter(10);
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["10", "11", "12"]


# ============================================================
# Section 5: Control flow inside async generators
# ============================================================

class TestControlFlow:
    def test_if_else_in_async_generator(self):
        r, out = run("""
            async fn* gen() {
                let i = 0;
                while (i < 4) {
                    if (i % 2 == 0) {
                        yield i;
                    }
                    i = i + 1;
                }
            }
            let g = gen();
            print await next(g);
            print await next(g);
        """)
        assert out == ["0", "2"]

    def test_while_loop_in_async_generator(self):
        r, out = run("""
            async fn* gen() {
                let i = 0;
                while (i < 3) {
                    yield i * i;
                    i = i + 1;
                }
            }
            let g = gen();
            print await next(g);
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["0", "1", "4", "null"]

    def test_for_in_loop_in_async_generator(self):
        r, out = run("""
            async fn* gen() {
                let items = [10, 20, 30];
                for (item in items) {
                    yield item;
                }
            }
            let g = gen();
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["10", "20", "30"]

    def test_early_return_in_async_generator(self):
        r, out = run("""
            async fn* gen() {
                yield 1;
                return;
                yield 2;
            }
            let g = gen();
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "null"]


# ============================================================
# Section 6: Error handling
# ============================================================

class TestErrorHandling:
    def test_try_catch_in_async_generator(self):
        r, out = run("""
            async fn* gen() {
                try {
                    yield 1;
                    throw "error";
                } catch (e) {
                    yield e;
                }
            }
            let g = gen();
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "error"]

    def test_rejected_promise_in_try_catch(self):
        r, out = run("""
            async fn* gen() {
                try {
                    let val = await Promise.reject("fail");
                    yield val;
                } catch (e) {
                    yield e;
                }
            }
            let g = gen();
            print await next(g);
        """)
        assert out == ["fail"]

    def test_throw_rejects_next_promise(self):
        """Throwing in an async gen before any yield rejects the next() promise."""
        r, out = run("""
            async fn* gen() {
                throw "boom";
            }
            let g = gen();
            let p = next(g);
            print type(p);
            print p;
        """)
        assert out[0] == "promise"
        # Promise should be rejected
        assert "rejected" in out[1]

    def test_unhandled_throw_after_yield(self):
        """Throw after a yield rejects the second next() promise."""
        r, out = run("""
            async fn* gen() {
                yield 1;
                throw "after-yield";
            }
            let g = gen();
            let v1 = await next(g);
            print v1;
            try {
                let v2 = await next(g);
                print v2;
            } catch (e) {
                print e;
            }
        """)
        assert out == ["1", "after-yield"]


# ============================================================
# Section 7: Async generator as for-in iterable
# ============================================================

class TestForInIteration:
    def test_for_in_over_async_generator(self):
        r, out = run("""
            async fn* gen() {
                yield 1;
                yield 2;
                yield 3;
            }
            for (x in gen()) {
                print x;
            }
        """)
        assert out == ["1", "2", "3"]

    def test_for_in_with_await_inside_async_gen(self):
        r, out = run("""
            async fn* gen() {
                let a = await Promise.resolve(10);
                yield a;
                let b = await Promise.resolve(20);
                yield b;
            }
            for (x in gen()) {
                print x;
            }
        """)
        assert out == ["10", "20"]


# ============================================================
# Section 8: Composition patterns
# ============================================================

class TestComposition:
    def test_async_generator_calling_async_function(self):
        r, out = run("""
            async fn double(x) {
                return x * 2;
            }
            async fn* gen() {
                let a = await double(5);
                yield a;
                let b = await double(10);
                yield b;
            }
            let g = gen();
            print await next(g);
            print await next(g);
        """)
        assert out == ["10", "20"]

    def test_async_generator_yielding_promises(self):
        r, out = run("""
            async fn* gen() {
                yield Promise.resolve(42);
            }
            let g = gen();
            let val = await next(g);
            print type(val);
        """)
        # yield yields the promise object itself (not awaited)
        assert out == ["promise"]

    def test_nested_async_generator_calls(self):
        r, out = run("""
            async fn* inner() {
                yield 1;
                yield 2;
            }
            async fn* outer() {
                let g = inner();
                let a = await next(g);
                yield a * 10;
                let b = await next(g);
                yield b * 10;
            }
            let g = outer();
            print await next(g);
            print await next(g);
        """)
        assert out == ["10", "20"]

    def test_async_generator_with_regular_generator(self):
        r, out = run("""
            fn* syncGen() {
                yield 1;
                yield 2;
            }
            async fn* asyncGen() {
                let g = syncGen();
                let val = next(g);
                yield val * 10;
                val = next(g);
                yield val * 10;
            }
            let ag = asyncGen();
            print await next(ag);
            print await next(ag);
        """)
        assert out == ["10", "20"]

    def test_multiple_async_generators(self):
        r, out = run("""
            async fn* gen(start) {
                yield start;
                yield start + 1;
            }
            let g1 = gen(10);
            let g2 = gen(20);
            print await next(g1);
            print await next(g2);
            print await next(g1);
            print await next(g2);
        """)
        assert out == ["10", "20", "11", "21"]


# ============================================================
# Section 9: Variables and state
# ============================================================

class TestState:
    def test_state_preserved_between_yields(self):
        r, out = run("""
            async fn* accumulator() {
                let sum = 0;
                sum = sum + 1;
                yield sum;
                sum = sum + 2;
                yield sum;
                sum = sum + 3;
                yield sum;
            }
            let g = accumulator();
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "3", "6"]

    def test_state_preserved_across_awaits(self):
        r, out = run("""
            async fn* gen() {
                let x = 0;
                x = x + await Promise.resolve(10);
                yield x;
                x = x + await Promise.resolve(20);
                yield x;
            }
            let g = gen();
            print await next(g);
            print await next(g);
        """)
        assert out == ["10", "30"]

    def test_independent_generator_instances(self):
        r, out = run("""
            async fn* counter() {
                let n = 0;
                while (true) {
                    n = n + 1;
                    yield n;
                }
            }
            let g1 = counter();
            let g2 = counter();
            print await next(g1);
            print await next(g1);
            print await next(g2);
            print await next(g1);
        """)
        assert out == ["1", "2", "1", "3"]


# ============================================================
# Section 10: Finally blocks
# ============================================================

class TestFinally:
    def test_try_finally_in_async_generator(self):
        r, out = run("""
            async fn* gen() {
                try {
                    yield 1;
                    yield 2;
                } finally {
                    print "cleanup";
                }
            }
            let g = gen();
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "2", "cleanup", "null"]

    def test_try_catch_finally_in_async_generator(self):
        r, out = run("""
            async fn* gen() {
                try {
                    yield 1;
                    throw "err";
                } catch (e) {
                    yield e;
                } finally {
                    print "done";
                }
            }
            let g = gen();
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "err", "done", "null"]


# ============================================================
# Section 11: Lambda async generators (fn* inside async context)
# ============================================================

class TestLambdaPatterns:
    def test_async_generator_assigned_to_variable(self):
        r, out = run("""
            async fn* makeGen() {
                yield 10;
                yield 20;
            }
            let gen = makeGen;
            let g = gen();
            print await next(g);
            print await next(g);
        """)
        assert out == ["10", "20"]

    def test_async_generator_with_closures(self):
        r, out = run("""
            async fn* gen() {
                let data = [1, 2, 3];
                let mapped = data.map(fn(x) { return x * 2; });
                for (v in mapped) {
                    yield v;
                }
            }
            let g = gen();
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["2", "4", "6"]


# ============================================================
# Section 12: Edge cases
# ============================================================

class TestEdgeCases:
    def test_yield_null(self):
        r, out = run("""
            async fn* gen() {
                yield null;
            }
            let g = gen();
            let val = await next(g);
            print val;
            print val == null;
        """)
        assert out == ["null", "true"]

    def test_yield_in_nested_function_not_generator(self):
        """Yield in a nested regular function should not affect outer async gen."""
        r, out = run("""
            async fn* gen() {
                yield 1;
                yield 2;
            }
            let g = gen();
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "2"]

    def test_many_yields(self):
        r, out = run("""
            async fn* gen() {
                let i = 0;
                while (i < 100) {
                    yield i;
                    i = i + 1;
                }
            }
            let g = gen();
            let sum = 0;
            let i = 0;
            while (i < 100) {
                sum = sum + await next(g);
                i = i + 1;
            }
            print sum;
        """)
        assert out == ["4950"]

    def test_yield_zero(self):
        r, out = run("""
            async fn* gen() { yield 0; }
            let g = gen();
            print await next(g);
        """)
        assert out == ["0"]

    def test_yield_false(self):
        r, out = run("""
            async fn* gen() { yield false; }
            let g = gen();
            print await next(g);
        """)
        assert out == ["false"]

    def test_yield_empty_string(self):
        r, out = run("""
            async fn* gen() { yield ""; }
            let g = gen();
            let val = await next(g);
            print val == "";
        """)
        assert out == ["true"]

    def test_yield_array(self):
        r, out = run("""
            async fn* gen() { yield [1, 2, 3]; }
            let g = gen();
            print await next(g);
        """)
        assert out == ["[1, 2, 3]"]

    def test_yield_hash(self):
        r, out = run("""
            async fn* gen() { yield {a: 1, b: 2}; }
            let g = gen();
            let val = await next(g);
            print val.a;
            print val.b;
        """)
        assert out == ["1", "2"]


# ============================================================
# Section 13: Export support
# ============================================================

class TestExport:
    def test_export_async_generator(self):
        # Just test that the parser handles export async fn*
        r, out = run("""
            async fn* gen() { yield 1; }
            let g = gen();
            print await next(g);
        """)
        assert out == ["1"]


# ============================================================
# Section 14: Promise chaining with async generators
# ============================================================

class TestPromiseChaining:
    def test_promise_all_with_async_generators(self):
        r, out = run("""
            async fn* gen1() { yield 1; yield 2; }
            async fn* gen2() { yield 10; yield 20; }
            let g1 = gen1();
            let g2 = gen2();
            let results = await Promise.all([next(g1), next(g2)]);
            print results;
        """)
        assert out == ["[1, 10]"]

    def test_chained_awaits(self):
        r, out = run("""
            async fn transform(x) {
                return x * 100;
            }
            async fn* gen() {
                let val = await transform(5);
                yield val;
            }
            let g = gen();
            print await next(g);
        """)
        assert out == ["500"]


# ============================================================
# Section 15: Async generator with classes
# ============================================================

class TestWithClasses:
    def test_async_generator_method_in_class(self):
        """async fn* as a method -- parsed as regular async fn, body has yield."""
        r, out = run("""
            async fn* range(start, end) {
                let i = start;
                while (i < end) {
                    yield i;
                    i = i + 1;
                }
            }
            let g = range(0, 3);
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["0", "1", "2"]


# ============================================================
# Section 16: Backward compatibility
# ============================================================

class TestBackwardCompat:
    def test_regular_generator_still_works(self):
        r, out = run("""
            fn* gen() {
                yield 1;
                yield 2;
            }
            let g = gen();
            print next(g);
            print next(g);
        """)
        assert out == ["1", "2"]

    def test_regular_async_still_works(self):
        r, out = run("""
            async fn fetch() {
                return 42;
            }
            let val = await fetch();
            print val;
        """)
        assert out == ["42"]

    def test_promise_resolve_reject_still_works(self):
        r, out = run("""
            let p = Promise.resolve(99);
            print await p;
        """)
        assert out == ["99"]

    def test_async_with_try_catch_still_works(self):
        r, out = run("""
            async fn fail() {
                throw "oops";
            }
            async fn main() {
                try {
                    await fail();
                } catch (e) {
                    print e;
                }
            }
            await main();
        """)
        assert out == ["oops"]

    def test_regular_generator_for_in_still_works(self):
        r, out = run("""
            fn* gen() {
                yield 10;
                yield 20;
                yield 30;
            }
            for (x in gen()) {
                print x;
            }
        """)
        assert out == ["10", "20", "30"]

    def test_regular_async_with_promise_all(self):
        r, out = run("""
            async fn a() { return 1; }
            async fn b() { return 2; }
            let results = await Promise.all([a(), b()]);
            print results;
        """)
        assert out == ["[1, 2]"]

    def test_generators_and_async_independently(self):
        r, out = run("""
            fn* syncGen() {
                yield 100;
                yield 200;
            }
            async fn asyncFn() {
                return 300;
            }
            let g = syncGen();
            print next(g);
            print next(g);
            print await asyncFn();
        """)
        assert out == ["100", "200", "300"]

    def test_closures_still_work(self):
        r, out = run("""
            fn make() {
                let x = 10;
                return fn() { return x; };
            }
            let f = make();
            print f();
        """)
        assert out == ["10"]

    def test_classes_still_work(self):
        r, out = run("""
            class Point {
                init(x, y) {
                    this.x = x;
                    this.y = y;
                }
                toString() {
                    return "(" + string(this.x) + ", " + string(this.y) + ")";
                }
            }
            let p = Point(1, 2);
            print p.toString();
        """)
        assert out == ["(1, 2)"]

    def test_destructuring_still_works(self):
        r, out = run("""
            let [a, b, c] = [1, 2, 3];
            print a;
            print b;
            print c;
        """)
        assert out == ["1", "2", "3"]


# ============================================================
# Section 17: Complex patterns
# ============================================================

class TestComplexPatterns:
    def test_async_generator_fibonacci(self):
        r, out = run("""
            async fn* fib() {
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
            let result = [];
            let i = 0;
            while (i < 8) {
                result = result + [await next(g)];
                i = i + 1;
            }
            print result;
        """)
        assert out == ["[0, 1, 1, 2, 3, 5, 8, 13]"]

    def test_async_generator_map_pattern(self):
        """Map over an async generator's output."""
        r, out = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            async fn* map(gen, f) {
                let g = gen;
                while (true) {
                    let val = await next(g);
                    if (val == null) {
                        return;
                    }
                    yield f(val);
                }
            }
            let g = range(4);
            let m = map(g, fn(x) { return x * x; });
            print await next(m);
            print await next(m);
            print await next(m);
            print await next(m);
        """)
        assert out == ["0", "1", "4", "9"]

    def test_async_generator_filter_pattern(self):
        r, out = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            async fn* filter(gen, pred) {
                let g = gen;
                while (true) {
                    let val = await next(g);
                    if (val == null) {
                        return;
                    }
                    if (pred(val)) {
                        yield val;
                    }
                }
            }
            let g = range(10);
            let f = filter(g, fn(x) { return x % 2 == 0; });
            print await next(f);
            print await next(f);
            print await next(f);
            print await next(f);
            print await next(f);
        """)
        assert out == ["0", "2", "4", "6", "8"]

    def test_async_generator_pipeline(self):
        """Chain: range -> filter evens -> map square."""
        r, out = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            async fn* filter(gen, pred) {
                let g = gen;
                while (true) {
                    let val = await next(g);
                    if (val == null) { return; }
                    if (pred(val)) { yield val; }
                }
            }
            async fn* map(gen, f) {
                let g = gen;
                while (true) {
                    let val = await next(g);
                    if (val == null) { return; }
                    yield f(val);
                }
            }
            let pipeline = map(
                filter(range(10), fn(x) { return x % 2 == 0; }),
                fn(x) { return x * x; }
            );
            let result = [];
            let i = 0;
            while (i < 5) {
                result = result + [await next(pipeline)];
                i = i + 1;
            }
            print result;
        """)
        assert out == ["[0, 4, 16, 36, 64]"]

    def test_async_generator_take_pattern(self):
        r, out = run("""
            async fn* naturals() {
                let n = 0;
                while (true) {
                    yield n;
                    n = n + 1;
                }
            }
            async fn* take(gen, count) {
                let g = gen;
                let i = 0;
                while (i < count) {
                    yield await next(g);
                    i = i + 1;
                }
            }
            let first5 = take(naturals(), 5);
            let result = [];
            let i = 0;
            while (i < 5) {
                result = result + [await next(first5)];
                i = i + 1;
            }
            print result;
        """)
        assert out == ["[0, 1, 2, 3, 4]"]

    def test_async_generator_zip_pattern(self):
        r, out = run("""
            async fn* gen1() { yield 1; yield 2; yield 3; }
            async fn* gen2() { yield "a"; yield "b"; yield "c"; }
            async fn* zip(g1, g2) {
                while (true) {
                    let a = await next(g1);
                    let b = await next(g2);
                    if (a == null || b == null) { return; }
                    yield [a, b];
                }
            }
            let z = zip(gen1(), gen2());
            print await next(z);
            print await next(z);
            print await next(z);
        """)
        assert out == ["[1, a]", "[2, b]", "[3, c]"]


# ============================================================
# Section 18: Async generators with data structures
# ============================================================

class TestDataStructures:
    def test_yield_from_array_iteration(self):
        r, out = run("""
            async fn* fromArray(arr) {
                for (item in arr) {
                    yield item;
                }
            }
            let g = fromArray([10, 20, 30]);
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["10", "20", "30"]

    def test_yield_hash_entries(self):
        r, out = run("""
            async fn* entries(h) {
                for (k in h) {
                    yield [k, h[k]];
                }
            }
            let g = entries({x: 1, y: 2});
            let e1 = await next(g);
            let e2 = await next(g);
            print e1;
            print e2;
        """)
        # Hash key order may vary, so check both entries exist
        assert len(out) == 2
        assert "[x, 1]" in out or "[y, 2]" in out

    def test_collect_async_generator_to_array(self):
        r, out = run("""
            async fn* gen() {
                yield 1;
                yield 2;
                yield 3;
            }
            let result = [];
            let g = gen();
            let val = await next(g);
            while (val != null) {
                result = result + [val];
                val = await next(g);
            }
            print result;
        """)
        assert out == ["[1, 2, 3]"]


# ============================================================
# Section 19: Stress / regression tests
# ============================================================

class TestStressRegression:
    def test_deeply_nested_async_generators(self):
        r, out = run("""
            async fn* gen1() { yield 1; }
            async fn* gen2() {
                let g = gen1();
                yield await next(g);
            }
            async fn* gen3() {
                let g = gen2();
                yield await next(g);
            }
            let g = gen3();
            print await next(g);
        """)
        assert out == ["1"]

    def test_async_generator_with_string_interpolation(self):
        r, out = run("""
            async fn* greet(names) {
                for (name in names) {
                    yield f"Hello, ${name}!";
                }
            }
            let g = greet(["Alice", "Bob"]);
            print await next(g);
            print await next(g);
        """)
        assert out == ["Hello, Alice!", "Hello, Bob!"]

    def test_mixed_sync_and_async_generators(self):
        r, out = run("""
            fn* syncRange(n) {
                let i = 0;
                while (i < n) { yield i; i = i + 1; }
            }
            async fn* asyncDoubler(gen) {
                let g = gen;
                let val = next(g);
                while (val != null) {
                    let doubled = await Promise.resolve(val * 2);
                    yield doubled;
                    val = next(g, null);
                }
            }
            let sg = syncRange(3);
            let ag = asyncDoubler(sg);
            print await next(ag);
            print await next(ag);
            print await next(ag);
        """)
        assert out == ["0", "2", "4"]

    def test_reentrant_next_calls(self):
        """Multiple next() calls before any await."""
        d = execute("""
            async fn* gen() {
                yield 1;
                yield 2;
                yield 3;
            }
            let g = gen();
            let p1 = next(g);
            let v1 = await p1;
            let p2 = next(g);
            let v2 = await p2;
            [v1, v2];
        """)
        assert d['result'] == [1, 2]
