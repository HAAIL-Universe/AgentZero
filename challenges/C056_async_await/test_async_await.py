"""
Tests for C056: Async/Await
"""
import pytest
from async_await import run, execute, parse, PromiseObject, VMError, ParseError, CompileError


# ============================================================
# Basic async function declaration and calling
# ============================================================

class TestAsyncBasics:
    def test_async_fn_returns_promise(self):
        result, output = run("""
            async fn greet() {
                return 42;
            }
            let p = greet();
            print type(p);
        """)
        assert output == ["promise"]

    def test_async_fn_resolved_value(self):
        result, output = run("""
            async fn greet() {
                return 42;
            }
            let p = greet();
            print await p;
        """)
        assert output == ["42"]

    def test_async_fn_no_return(self):
        result, output = run("""
            async fn doNothing() {
            }
            let p = doNothing();
            print await p;
        """)
        assert output == ["null"]

    def test_async_fn_with_params(self):
        result, output = run("""
            async fn add(a, b) {
                return a + b;
            }
            print await add(3, 4);
        """)
        assert output == ["7"]

    def test_async_fn_with_local_vars(self):
        result, output = run("""
            async fn compute() {
                let x = 10;
                let y = 20;
                return x + y;
            }
            print await compute();
        """)
        assert output == ["30"]

    def test_multiple_async_calls(self):
        result, output = run("""
            async fn double(x) {
                return x * 2;
            }
            print await double(5);
            print await double(10);
            print await double(15);
        """)
        assert output == ["10", "20", "30"]

    def test_async_fn_calling_sync_fn(self):
        result, output = run("""
            fn square(x) { return x * x; }
            async fn compute(x) {
                return square(x) + 1;
            }
            print await compute(5);
        """)
        assert output == ["26"]

    def test_async_fn_with_conditionals(self):
        result, output = run("""
            async fn abs(x) {
                if (x < 0) {
                    return -x;
                }
                return x;
            }
            print await abs(-5);
            print await abs(3);
        """)
        assert output == ["5", "3"]

    def test_async_fn_with_loop(self):
        result, output = run("""
            async fn sum(n) {
                let total = 0;
                let i = 1;
                while (i <= n) {
                    total = total + i;
                    i = i + 1;
                }
                return total;
            }
            print await sum(5);
        """)
        assert output == ["15"]


# ============================================================
# Await behavior
# ============================================================

class TestAwait:
    def test_await_non_promise(self):
        """Awaiting a non-promise value returns it directly."""
        result, output = run("""
            async fn f() {
                return await 42;
            }
            print await f();
        """)
        assert output == ["42"]

    def test_await_resolved_promise(self):
        result, output = run("""
            async fn f() {
                let p = Promise.resolve(99);
                return await p;
            }
            print await f();
        """)
        assert output == ["99"]

    def test_await_in_sequence(self):
        result, output = run("""
            async fn step1() { return 1; }
            async fn step2() { return 2; }
            async fn step3() { return 3; }
            async fn main() {
                let a = await step1();
                let b = await step2();
                let c = await step3();
                return a + b + c;
            }
            print await main();
        """)
        assert output == ["6"]

    def test_await_chain(self):
        result, output = run("""
            async fn double(x) { return x * 2; }
            async fn chain() {
                let val = await double(await double(await double(1)));
                return val;
            }
            print await chain();
        """)
        assert output == ["8"]

    def test_top_level_await(self):
        result, output = run("""
            async fn getValue() { return 42; }
            let result = await getValue();
            print result;
        """)
        assert output == ["42"]

    def test_await_null(self):
        result, output = run("""
            async fn f() {
                return await null;
            }
            print await f();
        """)
        assert output == ["null"]

    def test_await_string(self):
        result, output = run("""
            async fn f() {
                return await "hello";
            }
            print await f();
        """)
        assert output == ["hello"]


# ============================================================
# Promise static methods
# ============================================================

class TestPromiseStatics:
    def test_promise_resolve(self):
        result, output = run("""
            let p = Promise.resolve(42);
            print type(p);
            print await p;
        """)
        assert output == ["promise", "42"]

    def test_promise_reject(self):
        result, output = run("""
            let p = Promise.reject("error!");
            let caught = false;
            try {
                await p;
            } catch(e) {
                caught = true;
                print e;
            }
            print caught;
        """)
        assert output == ["error!", "true"]

    def test_promise_all_empty(self):
        result, output = run("""
            let p = Promise.all([]);
            let results = await p;
            print len(results);
        """)
        assert output == ["0"]

    def test_promise_all_resolved(self):
        result, output = run("""
            async fn a() { return 1; }
            async fn b() { return 2; }
            async fn c() { return 3; }
            let results = await Promise.all([a(), b(), c()]);
            print results[0];
            print results[1];
            print results[2];
        """)
        assert output == ["1", "2", "3"]

    def test_promise_all_with_values(self):
        result, output = run("""
            let results = await Promise.all([
                Promise.resolve(10),
                Promise.resolve(20),
                Promise.resolve(30)
            ]);
            print results[0] + results[1] + results[2];
        """)
        assert output == ["60"]

    def test_promise_all_reject(self):
        result, output = run("""
            let ok = false;
            try {
                await Promise.all([
                    Promise.resolve(1),
                    Promise.reject("fail"),
                    Promise.resolve(3)
                ]);
            } catch(e) {
                ok = true;
                print e;
            }
            print ok;
        """)
        assert output == ["fail", "true"]

    def test_promise_race_resolved(self):
        result, output = run("""
            let result = await Promise.race([
                Promise.resolve(42),
                Promise.resolve(99)
            ]);
            print result;
        """)
        assert output == ["42"]

    def test_promise_race_reject(self):
        result, output = run("""
            let ok = false;
            try {
                await Promise.race([
                    Promise.reject("first-error"),
                    Promise.resolve(99)
                ]);
            } catch(e) {
                ok = true;
                print e;
            }
            print ok;
        """)
        assert output == ["first-error", "true"]


# ============================================================
# Error handling with async/await
# ============================================================

class TestAsyncErrors:
    def test_try_catch_await_rejection(self):
        result, output = run("""
            async fn failing() {
                throw "async error";
            }
            async fn main() {
                try {
                    await failing();
                } catch(e) {
                    return e;
                }
            }
            print await main();
        """)
        assert output == ["async error"]

    def test_throw_in_async(self):
        result, output = run("""
            async fn bomb() {
                throw "boom";
            }
            let ok = false;
            try {
                await bomb();
            } catch(e) {
                ok = true;
                print e;
            }
            print ok;
        """)
        assert output == ["boom", "true"]

    def test_try_catch_in_async(self):
        result, output = run("""
            async fn safe() {
                try {
                    throw "inner error";
                } catch(e) {
                    return "caught: " + e;
                }
            }
            print await safe();
        """)
        assert output == ["caught: inner error"]

    def test_async_rejection_propagates(self):
        result, output = run("""
            async fn inner() {
                throw "deep error";
            }
            async fn outer() {
                return await inner();
            }
            let caught = false;
            try {
                await outer();
            } catch(e) {
                caught = true;
                print e;
            }
            print caught;
        """)
        assert output == ["deep error", "true"]

    def test_try_finally_in_async(self):
        result, output = run("""
            async fn f() {
                try {
                    return 42;
                } finally {
                    print "finally ran";
                }
            }
            print await f();
        """)
        assert output == ["finally ran", "42"]

    def test_try_catch_finally_in_async(self):
        result, output = run("""
            async fn f() {
                try {
                    throw "err";
                } catch(e) {
                    print "caught";
                } finally {
                    print "finally";
                }
                return "done";
            }
            print await f();
        """)
        assert output == ["caught", "finally", "done"]


# ============================================================
# Async with closures
# ============================================================

class TestAsyncClosures:
    def test_async_captures_outer_var(self):
        result, output = run("""
            let x = 10;
            async fn add(y) {
                return x + y;
            }
            print await add(5);
        """)
        assert output == ["15"]

    def test_async_fn_stored_in_var(self):
        result, output = run("""
            async fn make() { return 42; }
            let f = make;
            print await f();
        """)
        assert output == ["42"]

    def test_async_with_for_in(self):
        result, output = run("""
            async fn sumArray(arr) {
                let total = 0;
                for (x in arr) {
                    total = total + x;
                }
                return total;
            }
            print await sumArray([1, 2, 3, 4, 5]);
        """)
        assert output == ["15"]

    def test_async_with_array_ops(self):
        result, output = run("""
            async fn process() {
                let arr = [1, 2, 3];
                push(arr, 4);
                return len(arr);
            }
            print await process();
        """)
        assert output == ["4"]


# ============================================================
# Async with classes
# ============================================================

class TestAsyncWithClasses:
    def test_async_method_not_directly_supported(self):
        """Classes can call async functions."""
        result, output = run("""
            async fn fetchData() { return 42; }
            class Service {
                getData() {
                    return fetchData();
                }
            }
            let s = Service();
            print await s.getData();
        """)
        assert output == ["42"]

    def test_async_fn_returns_class_instance(self):
        result, output = run("""
            class Point {
                init(x, y) {
                    this.x = x;
                    this.y = y;
                }
            }
            async fn makePoint() {
                return Point(3, 4);
            }
            let p = await makePoint();
            print p.x;
            print p.y;
        """)
        assert output == ["3", "4"]


# ============================================================
# Async with string interpolation and other features
# ============================================================

class TestAsyncWithFeatures:
    def test_async_with_fstring(self):
        result, output = run("""
            async fn greet(name) {
                return f"Hello, ${name}!";
            }
            print await greet("World");
        """)
        assert output == ["Hello, World!"]

    def test_async_with_destructuring(self):
        result, output = run("""
            async fn getCoords() {
                return [10, 20];
            }
            async fn main() {
                let [x, y] = await getCoords();
                return x + y;
            }
            print await main();
        """)
        assert output == ["30"]

    def test_async_with_spread(self):
        result, output = run("""
            async fn getItems() {
                return [1, 2, 3];
            }
            async fn main() {
                let items = await getItems();
                let all = [0, ...items, 4];
                return len(all);
            }
            print await main();
        """)
        assert output == ["5"]

    def test_async_with_hash_maps(self):
        result, output = run("""
            async fn getConfig() {
                return {name: "test", value: 42};
            }
            async fn main() {
                let config = await getConfig();
                return config.name;
            }
            print await main();
        """)
        assert output == ["test"]

    def test_async_with_optional_chaining(self):
        result, output = run("""
            async fn getData() {
                return {user: {name: "Alice"}};
            }
            async fn main() {
                let data = await getData();
                return data?.user?.name;
            }
            print await main();
        """)
        assert output == ["Alice"]

    def test_async_with_null_coalescing(self):
        result, output = run("""
            async fn maybeNull() {
                return null;
            }
            async fn main() {
                let val = await maybeNull();
                return val ?? "default";
            }
            print await main();
        """)
        assert output == ["default"]

    def test_async_with_pipe(self):
        result, output = run("""
            fn double(x) { return x * 2; }
            fn addOne(x) { return x + 1; }
            async fn compute(x) {
                return x |> double |> addOne;
            }
            print await compute(5);
        """)
        assert output == ["11"]


# ============================================================
# Multiple concurrent async operations
# ============================================================

class TestAsyncConcurrency:
    def test_multiple_awaits_order(self):
        result, output = run("""
            async fn task(name, val) {
                print f"start ${name}";
                return val;
            }
            async fn main() {
                let a = task("A", 1);
                let b = task("B", 2);
                let c = task("C", 3);
                let ra = await a;
                let rb = await b;
                let rc = await c;
                print f"results: ${ra} ${rb} ${rc}";
            }
            await main();
        """)
        assert "start A" in output
        assert "start B" in output
        assert "start C" in output
        assert "results: 1 2 3" in output

    def test_promise_all_with_async_fns(self):
        result, output = run("""
            async fn task(x) { return x * 10; }
            async fn main() {
                let results = await Promise.all([
                    task(1), task(2), task(3)
                ]);
                return results;
            }
            let r = await main();
            print r[0];
            print r[1];
            print r[2];
        """)
        assert output == ["10", "20", "30"]

    def test_fire_and_forget_async(self):
        """Async functions that are called but never awaited still run."""
        result, output = run("""
            async fn sideEffect() {
                print "ran";
                return null;
            }
            sideEffect();
        """)
        assert output == ["ran"]


# ============================================================
# Async with modules (export async fn)
# ============================================================

class TestAsyncModules:
    def test_export_async_fn(self):
        from async_await import ModuleRegistry
        reg = ModuleRegistry()
        reg.register("mathmod", """
            export async fn asyncDouble(x) {
                return x * 2;
            }
        """)
        result, output = run("""
            import { asyncDouble } from "mathmod";
            print await asyncDouble(21);
        """, registry=reg)
        assert output == ["42"]


# ============================================================
# Edge cases
# ============================================================

class TestAsyncEdgeCases:
    def test_await_already_resolved(self):
        result, output = run("""
            let p = Promise.resolve(42);
            print await p;
        """)
        assert output == ["42"]

    def test_await_already_rejected(self):
        result, output = run("""
            let p = Promise.reject("err");
            try {
                await p;
            } catch(e) {
                print e;
            }
        """)
        assert output == ["err"]

    def test_nested_async_calls(self):
        result, output = run("""
            async fn a() { return 1; }
            async fn b() { return await a() + 2; }
            async fn c() { return await b() + 3; }
            print await c();
        """)
        assert output == ["6"]

    def test_async_returning_promise(self):
        result, output = run("""
            async fn inner() { return 42; }
            async fn outer() {
                return inner();
            }
            let p = outer();
            let result = await p;
            // result is a promise (from inner()), need to await it
            print await result;
        """)
        assert output == ["42"]

    def test_async_void_side_effects(self):
        """Async fn print is a real side effect (output is shared)."""
        result, output = run("""
            async fn sayHello(name) {
                print f"hello ${name}";
            }
            await sayHello("A");
            await sayHello("B");
            await sayHello("C");
        """)
        assert output == ["hello A", "hello B", "hello C"]

    def test_recursive_async(self):
        result, output = run("""
            async fn factorial(n) {
                if (n <= 1) { return 1; }
                let sub = await factorial(n - 1);
                return n * sub;
            }
            print await factorial(5);
        """)
        assert output == ["120"]

    def test_async_with_while_loop(self):
        result, output = run("""
            async fn countdown(n) {
                let result = [];
                while (n > 0) {
                    push(result, n);
                    n = n - 1;
                }
                return result;
            }
            let r = await countdown(3);
            print r[0];
            print r[1];
            print r[2];
        """)
        assert output == ["3", "2", "1"]

    def test_async_with_break(self):
        result, output = run("""
            async fn findFirst(arr, target) {
                for (x in arr) {
                    if (x == target) {
                        return x;
                    }
                }
                return null;
            }
            print await findFirst([1, 2, 3, 4], 3);
        """)
        assert output == ["3"]

    def test_async_promise_state(self):
        result, output = run("""
            async fn f() { return 42; }
            let p = f();
            // Promise should be resolved immediately since no awaits inside
            print type(p);
        """)
        assert output == ["promise"]


# ============================================================
# Interaction between async and generators
# ============================================================

class TestAsyncAndGenerators:
    def test_async_fn_using_generator(self):
        result, output = run("""
            fn* range_gen(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            async fn sumRange(n) {
                let gen = range_gen(n);
                let total = 0;
                let val = next(gen, -1);
                while (val != -1) {
                    total = total + val;
                    val = next(gen, -1);
                }
                return total;
            }
            print await sumRange(5);
        """)
        assert output == ["10"]

    def test_generator_cannot_be_async(self):
        """fn* and async should not be combined -- async fn* is not supported."""
        # We just test that async fn (without *) works, and fn* works independently
        result, output = run("""
            async fn asyncFn() { return 1; }
            fn* genFn() { yield 2; }
            print await asyncFn();
            print next(genFn());
        """)
        assert output == ["1", "2"]


# ============================================================
# Promise chaining patterns
# ============================================================

class TestPromiseChaining:
    def test_sequential_awaits(self):
        result, output = run("""
            async fn step1() { return 1; }
            async fn step2(prev) { return prev + 2; }
            async fn step3(prev) { return prev + 3; }
            async fn pipeline() {
                let v = await step1();
                v = await step2(v);
                v = await step3(v);
                return v;
            }
            print await pipeline();
        """)
        assert output == ["6"]

    def test_parallel_then_combine(self):
        result, output = run("""
            async fn fetchA() { return 10; }
            async fn fetchB() { return 20; }
            async fn main() {
                let pa = fetchA();
                let pb = fetchB();
                let a = await pa;
                let b = await pb;
                return a + b;
            }
            print await main();
        """)
        assert output == ["30"]


# ============================================================
# Stress tests
# ============================================================

class TestAsyncStress:
    def test_many_sequential_awaits(self):
        result, output = run("""
            async fn identity(x) { return x; }
            async fn main() {
                let sum = 0;
                let i = 0;
                while (i < 20) {
                    sum = sum + await identity(i);
                    i = i + 1;
                }
                return sum;
            }
            print await main();
        """)
        assert output == ["190"]

    def test_promise_all_many(self):
        result, output = run("""
            async fn val(x) { return x; }
            let promises = [];
            let i = 0;
            while (i < 10) {
                push(promises, val(i));
                i = i + 1;
            }
            let results = await Promise.all(promises);
            let sum = 0;
            for (r in results) {
                sum = sum + r;
            }
            print sum;
        """)
        assert output == ["45"]


# ============================================================
# All previous features still work
# ============================================================

class TestBackwardsCompat:
    def test_generators_still_work(self):
        result, output = run("""
            fn* counter(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            let gen = counter(3);
            print next(gen);
            print next(gen);
            print next(gen);
        """)
        assert output == ["0", "1", "2"]

    def test_classes_still_work(self):
        result, output = run("""
            class Animal {
                init(name) { this.name = name; }
                speak() { return f"${this.name} speaks"; }
            }
            class Dog < Animal {
                speak() { return f"${this.name} barks"; }
            }
            let d = Dog("Rex");
            print d.speak();
        """)
        assert output == ["Rex barks"]

    def test_closures_still_work(self):
        """Closures capture env by copy, so each call sees the original state."""
        result, output = run("""
            fn makeAdder(x) {
                return fn(y) {
                    return x + y;
                };
            }
            let add5 = makeAdder(5);
            print add5(1);
            print add5(10);
            print add5(100);
        """)
        assert output == ["6", "15", "105"]

    def test_try_catch_finally_still_works(self):
        result, output = run("""
            try {
                throw "err";
            } catch(e) {
                print e;
            } finally {
                print "done";
            }
        """)
        assert output == ["err", "done"]

    def test_destructuring_still_works(self):
        result, output = run("""
            let [a, b, c] = [1, 2, 3];
            print a + b + c;
        """)
        assert output == ["6"]

    def test_spread_still_works(self):
        result, output = run("""
            let a = [1, 2];
            let b = [...a, 3, 4];
            print len(b);
        """)
        assert output == ["4"]

    def test_optional_chaining_still_works(self):
        result, output = run("""
            let obj = null;
            print obj?.name;
        """)
        assert output == ["null"]

    def test_null_coalescing_still_works(self):
        result, output = run("""
            let x = null;
            print x ?? 42;
        """)
        assert output == ["42"]

    def test_pipe_operator_still_works(self):
        result, output = run("""
            fn double(x) { return x * 2; }
            print 5 |> double;
        """)
        assert output == ["10"]

    def test_for_in_still_works(self):
        result, output = run("""
            let sum = 0;
            for (x in [1, 2, 3]) {
                sum = sum + x;
            }
            print sum;
        """)
        assert output == ["6"]

    def test_modules_still_work(self):
        from async_await import ModuleRegistry
        reg = ModuleRegistry()
        reg.register("math", """
            export fn add(a, b) { return a + b; }
        """)
        result, output = run("""
            import { add } from "math";
            print add(3, 4);
        """, registry=reg)
        assert output == ["7"]

    def test_string_interpolation_still_works(self):
        result, output = run("""
            let name = "World";
            print f"Hello, ${name}!";
        """)
        assert output == ["Hello, World!"]


# ============================================================
# Async with error recovery patterns
# ============================================================

class TestAsyncErrorRecovery:
    def test_retry_pattern(self):
        """Use array for shared mutable state across async calls."""
        result, output = run("""
            let state = [0];
            async fn unreliable(state) {
                state[0] = state[0] + 1;
                if (state[0] < 3) {
                    throw "not ready";
                }
                return "success";
            }
            async fn retry(maxAttempts) {
                let i = 0;
                while (i < maxAttempts) {
                    try {
                        return await unreliable(state);
                    } catch(e) {
                        i = i + 1;
                    }
                }
                return "failed";
            }
            print await retry(5);
        """)
        assert output == ["success"]

    def test_fallback_pattern(self):
        result, output = run("""
            async fn primary() { throw "down"; }
            async fn fallback() { return "backup"; }
            async fn withFallback() {
                try {
                    return await primary();
                } catch(e) {
                    return await fallback();
                }
            }
            print await withFallback();
        """)
        assert output == ["backup"]


# ============================================================
# Promise.all with mixed resolved/pending
# ============================================================

class TestPromiseAllMixed:
    def test_promise_all_non_promise_values(self):
        result, output = run("""
            let results = await Promise.all([1, 2, 3]);
            print results[0];
            print results[1];
            print results[2];
        """)
        assert output == ["1", "2", "3"]

    def test_promise_all_mixed(self):
        result, output = run("""
            async fn makeVal(x) { return x * 10; }
            let results = await Promise.all([
                makeVal(1),
                Promise.resolve(20),
                makeVal(3)
            ]);
            print results[0];
            print results[1];
            print results[2];
        """)
        assert output == ["10", "20", "30"]


# ============================================================
# Additional edge cases and patterns
# ============================================================

class TestAsyncAdditional:
    def test_async_with_instanceof(self):
        result, output = run("""
            class Animal {}
            async fn makeAnimal() {
                return Animal();
            }
            let a = await makeAnimal();
            print instanceof(a, Animal);
        """)
        assert output == ["true"]

    def test_async_with_try_catch_rethrow(self):
        result, output = run("""
            async fn f() {
                try {
                    throw "original";
                } catch(e) {
                    throw "modified: " + e;
                }
            }
            try {
                await f();
            } catch(e) {
                print e;
            }
        """)
        assert output == ["modified: original"]

    def test_await_promise_resolve_null(self):
        result, output = run("""
            let p = Promise.resolve(null);
            let r = await p;
            print r;
        """)
        assert output == ["null"]

    def test_await_promise_resolve_array(self):
        result, output = run("""
            let p = Promise.resolve([1, 2, 3]);
            let arr = await p;
            print len(arr);
            print arr[1];
        """)
        assert output == ["3", "2"]

    def test_await_promise_resolve_hash(self):
        result, output = run("""
            let p = Promise.resolve({x: 10, y: 20});
            let h = await p;
            print h.x + h.y;
        """)
        assert output == ["30"]

    def test_async_fn_type(self):
        result, output = run("""
            async fn f() { return 1; }
            print type(f);
        """)
        assert output == ["function"]

    def test_async_fn_returns_bool(self):
        result, output = run("""
            async fn isEven(n) {
                return n % 2 == 0;
            }
            print await isEven(4);
            print await isEven(3);
        """)
        assert output == ["true", "false"]

    def test_async_fn_returns_float(self):
        result, output = run("""
            async fn half(n) {
                return n / 2.0;
            }
            print await half(5.0);
        """)
        assert output == ["2.5"]

    def test_promise_race_with_async(self):
        result, output = run("""
            async fn fast() { return "fast"; }
            async fn slow() { return "slow"; }
            let result = await Promise.race([fast(), slow()]);
            print result;
        """)
        assert output == ["fast"]

    def test_multiple_promise_all(self):
        result, output = run("""
            async fn val(x) { return x; }
            let r1 = await Promise.all([val(1), val(2)]);
            let r2 = await Promise.all([val(3), val(4)]);
            print r1[0] + r1[1] + r2[0] + r2[1];
        """)
        assert output == ["10"]

    def test_async_with_string_ops(self):
        result, output = run("""
            async fn greet(first, last) {
                return f"${first} ${last}";
            }
            let name = await greet("John", "Doe");
            print len(name);
            print name;
        """)
        assert output == ["8", "John Doe"]

    def test_async_with_nested_hash(self):
        result, output = run("""
            async fn makeConfig() {
                return {
                    db: {host: "localhost", port: 5432},
                    app: {name: "test"}
                };
            }
            let cfg = await makeConfig();
            print cfg.db.host;
            print cfg.db.port;
            print cfg.app.name;
        """)
        assert output == ["localhost", "5432", "test"]

    def test_async_early_return(self):
        result, output = run("""
            async fn findPositive(arr) {
                for (x in arr) {
                    if (x > 0) {
                        return x;
                    }
                }
                return null;
            }
            print await findPositive([-3, -1, 0, 5, 8]);
        """)
        assert output == ["5"]

    def test_async_with_reduce(self):
        result, output = run("""
            async fn asyncSum(arr) {
                return reduce(arr, fn(acc, x) { return acc + x; }, 0);
            }
            print await asyncSum([1, 2, 3, 4, 5]);
        """)
        assert output == ["15"]

    def test_async_with_map(self):
        result, output = run("""
            async fn processArr() {
                let arr = [1, 2, 3];
                return map(arr, fn(x) { return x * 10; });
            }
            let r = await processArr();
            print r[0];
            print r[1];
            print r[2];
        """)
        assert output == ["10", "20", "30"]

    def test_async_with_filter(self):
        result, output = run("""
            async fn getEvens(arr) {
                return filter(arr, fn(x) { return x % 2 == 0; });
            }
            let r = await getEvens([1, 2, 3, 4, 5, 6]);
            print len(r);
            print r[0];
            print r[1];
        """)
        assert output == ["3", "2", "4"]

    def test_async_nested_await_in_expression(self):
        result, output = run("""
            async fn a() { return 10; }
            async fn b() { return 20; }
            async fn c() {
                return await a() + await b();
            }
            print await c();
        """)
        assert output == ["30"]

    def test_async_with_null_check(self):
        result, output = run("""
            async fn maybeGet(obj, key) {
                return obj?.[key] ?? "missing";
            }
            print await maybeGet({name: "Alice"}, "name");
            print await maybeGet(null, "name");
        """)
        assert output == ["Alice", "missing"]

    def test_promise_all_single(self):
        result, output = run("""
            let r = await Promise.all([Promise.resolve(42)]);
            print r[0];
        """)
        assert output == ["42"]

    def test_async_fn_prints_during_execution(self):
        result, output = run("""
            async fn logAndReturn(msg, val) {
                print msg;
                return val;
            }
            let r = await logAndReturn("computing", 42);
            print r;
        """)
        assert output == ["computing", "42"]

    def test_async_chained_error_handling(self):
        result, output = run("""
            async fn step1() { return 1; }
            async fn step2(x) {
                if (x == 1) { throw "step2 failed"; }
                return x + 1;
            }
            async fn pipeline() {
                try {
                    let v = await step1();
                    let w = await step2(v);
                    return w;
                } catch(e) {
                    return "caught: " + e;
                }
            }
            print await pipeline();
        """)
        assert output == ["caught: step2 failed"]

    def test_async_with_classes_and_methods(self):
        result, output = run("""
            class Calculator {
                init(base) { this.base = base; }
                add(x) { return this.base + x; }
            }
            async fn compute() {
                let calc = Calculator(100);
                return calc.add(42);
            }
            print await compute();
        """)
        assert output == ["142"]

    def test_async_with_inheritance(self):
        result, output = run("""
            class Shape {
                area() { return 0; }
            }
            class Circle < Shape {
                init(r) { this.r = r; }
                area() { return this.r * this.r * 3; }
            }
            async fn getArea() {
                let c = Circle(5);
                return c.area();
            }
            print await getArea();
        """)
        assert output == ["75"]


class TestAsyncWithModuleSystem:
    def test_async_fn_in_module_calling_other_module(self):
        from async_await import ModuleRegistry
        reg = ModuleRegistry()
        reg.register("utils", """
            export fn double(x) { return x * 2; }
        """)
        reg.register("service", """
            import { double } from "utils";
            export async fn transform(x) {
                return double(x) + 1;
            }
        """)
        result, output = run("""
            import { transform } from "service";
            print await transform(5);
        """, registry=reg)
        assert output == ["11"]


class TestPromiseEdgeCases:
    def test_promise_resolve_with_promise(self):
        """Promise.resolve with a non-promise value."""
        result, output = run("""
            let p = Promise.resolve(Promise.resolve(42));
            // Returns a promise wrapping a resolved promise
            let inner = await p;
            print await inner;
        """)
        assert output == ["42"]

    def test_empty_async_fn(self):
        result, output = run("""
            async fn noop() {}
            let r = await noop();
            print r;
        """)
        assert output == ["null"]

    def test_async_fn_returns_string(self):
        result, output = run("""
            async fn hello() { return "world"; }
            print await hello();
        """)
        assert output == ["world"]

    def test_multiple_awaits_different_types(self):
        result, output = run("""
            async fn getInt() { return 42; }
            async fn getStr() { return "hello"; }
            async fn getBool() { return true; }
            async fn getArr() { return [1, 2]; }
            print await getInt();
            print await getStr();
            print await getBool();
            let a = await getArr();
            print a[0];
        """)
        assert output == ["42", "hello", "true", "1"]

    def test_deeply_nested_async(self):
        result, output = run("""
            async fn level4() { return 99; }
            async fn level3() { return await level4(); }
            async fn level2() { return await level3(); }
            async fn level1() { return await level2(); }
            print await level1();
        """)
        assert output == ["99"]

    def test_async_error_in_promise_all(self):
        result, output = run("""
            async fn ok() { return 1; }
            async fn fail() { throw "bad"; }
            try {
                await Promise.all([ok(), fail(), ok()]);
            } catch(e) {
                print e;
            }
        """)
        assert output == ["bad"]

    def test_promise_race_all_resolved(self):
        result, output = run("""
            let r = await Promise.race([
                Promise.resolve(1),
                Promise.resolve(2),
                Promise.resolve(3)
            ]);
            print r;
        """)
        assert output == ["1"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
