"""Tests for C065: For-Await Loops.

Extends C064 (async generators) with `for await (x in asyncGen()) { ... }` syntax.
Lazy async iteration -- values consumed one at a time, not eagerly collected.
"""
import pytest
from for_await import run, execute, ParseError, CompileError, VMError


# ============================================================
# Section 1: Basic for-await
# ============================================================

class TestBasicForAwait:
    def test_basic_iteration(self):
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; yield 3; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [1, 2, 3]

    def test_single_yield(self):
        r, _ = run("""
            async fn* gen() { yield 42; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [42]

    def test_empty_generator(self):
        r, _ = run("""
            async fn* gen() {}
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == []

    def test_string_values(self):
        r, _ = run("""
            async fn* gen() { yield "hello"; yield "world"; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == ["hello", "world"]

    def test_mixed_types(self):
        r, _ = run("""
            async fn* gen() { yield 1; yield "two"; yield true; yield [3, 4]; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [1, "two", True, [3, 4]]

    def test_yield_null(self):
        """Null values should be yielded correctly, not treated as done."""
        r, _ = run("""
            async fn* gen() { yield null; yield 1; yield null; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [None, 1, None]

    def test_yield_false_and_zero(self):
        r, _ = run("""
            async fn* gen() { yield false; yield 0; yield ""; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [False, 0, ""]


# ============================================================
# Section 2: Await inside async generator
# ============================================================

class TestAwaitInside:
    def test_await_resolved_promise(self):
        r, _ = run("""
            async fn delay(v) { return v; }
            async fn* gen() {
                yield await delay(10);
                yield await delay(20);
            }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [10, 20]

    def test_await_between_yields(self):
        r, _ = run("""
            async fn compute(a, b) { return a + b; }
            async fn* gen() {
                let x = await compute(1, 2);
                yield x;
                let y = await compute(x, 3);
                yield y;
            }
            let results = [];
            for await (v in gen()) { results = results + [v]; }
            results;
        """)
        assert r == [3, 6]

    def test_await_promise_all(self):
        r, _ = run("""
            async fn val(x) { return x; }
            async fn* gen() {
                let arr = await Promise.all([val(1), val(2), val(3)]);
                for (x in arr) { yield x; }
            }
            let results = [];
            for await (v in gen()) { results = results + [v]; }
            results;
        """)
        assert r == [1, 2, 3]


# ============================================================
# Section 3: Control flow
# ============================================================

class TestControlFlow:
    def test_break(self):
        r, _ = run("""
            async fn* count() {
                let i = 0;
                while (true) { yield i; i = i + 1; }
            }
            let results = [];
            for await (x in count()) {
                if (x >= 3) { break; }
                results = results + [x];
            }
            results;
        """)
        assert r == [0, 1, 2]

    def test_continue(self):
        r, _ = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) { yield i; i = i + 1; }
            }
            let results = [];
            for await (x in range(6)) {
                if (x % 2 == 0) { continue; }
                results = results + [x];
            }
            results;
        """)
        assert r == [1, 3, 5]

    def test_early_return_in_body(self):
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; yield 3; }
            fn find_first() {
                for await (x in gen()) {
                    if (x == 2) { return x; }
                }
                return -1;
            }
            find_first();
        """)
        assert r == 2

    def test_nested_for_await(self):
        r, _ = run("""
            async fn* outer() { yield 1; yield 2; }
            async fn* inner(n) { yield n * 10; yield n * 20; }
            let results = [];
            for await (x in outer()) {
                for await (y in inner(x)) {
                    results = results + [y];
                }
            }
            results;
        """)
        assert r == [10, 20, 20, 40]

    def test_if_else_in_body(self):
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; yield 3; yield 4; }
            let evens = [];
            let odds = [];
            for await (x in gen()) {
                if (x % 2 == 0) { evens = evens + [x]; }
                else { odds = odds + [x]; }
            }
            [odds, evens];
        """)
        assert r == [[1, 3], [2, 4]]


# ============================================================
# Section 4: Destructuring
# ============================================================

class TestDestructuring:
    def test_array_destructure(self):
        r, _ = run("""
            async fn* pairs() { yield [1, 2]; yield [3, 4]; }
            let sums = [];
            for await ([a, b] in pairs()) { sums = sums + [a + b]; }
            sums;
        """)
        assert r == [3, 7]

    def test_hash_destructure(self):
        r, _ = run("""
            async fn* records() {
                yield {name: "alice", age: 30};
                yield {name: "bob", age: 25};
            }
            let names = [];
            for await ({name, age} in records()) { names = names + [name]; }
            names;
        """)
        assert r == ["alice", "bob"]

    def test_nested_destructure(self):
        r, _ = run("""
            async fn* gen() {
                yield [1, [2, 3]];
                yield [4, [5, 6]];
            }
            let results = [];
            for await ([a, [b, c]] in gen()) {
                results = results + [a + b + c];
            }
            results;
        """)
        assert r == [6, 15]

    def test_rest_destructure(self):
        r, _ = run("""
            async fn* gen() {
                yield [1, 2, 3, 4];
                yield [5, 6, 7];
            }
            let results = [];
            for await ([first, ...rest] in gen()) {
                results = results + [rest];
            }
            results;
        """)
        assert r == [[2, 3, 4], [6, 7]]


# ============================================================
# Section 5: Parameters and closures
# ============================================================

class TestParamsClosures:
    def test_generator_with_params(self):
        r, _ = run("""
            async fn* range(start, end) {
                let i = start;
                while (i < end) { yield i; i = i + 1; }
            }
            let results = [];
            for await (x in range(3, 7)) { results = results + [x]; }
            results;
        """)
        assert r == [3, 4, 5, 6]

    def test_closure_capture(self):
        r, _ = run("""
            async fn* scaled(multiplier) {
                yield 1 * multiplier;
                yield 2 * multiplier;
                yield 3 * multiplier;
            }
            let results = [];
            for await (x in scaled(10)) { results = results + [x]; }
            results;
        """)
        assert r == [10, 20, 30]

    def test_rest_params(self):
        r, _ = run("""
            async fn* from_args(...args) {
                for (x in args) { yield x; }
            }
            let results = [];
            for await (x in from_args(10, 20, 30)) { results = results + [x]; }
            results;
        """)
        assert r == [10, 20, 30]


# ============================================================
# Section 6: Error handling
# ============================================================

class TestErrorHandling:
    def test_try_catch_around_for_await(self):
        r, _ = run("""
            async fn* gen() {
                yield 1;
                throw "error";
            }
            let result = "none";
            try {
                for await (x in gen()) {
                    result = x;
                }
            } catch (e) {
                result = e;
            }
            result;
        """)
        assert r == "error"

    def test_try_catch_inside_body(self):
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; yield 3; }
            let results = [];
            for await (x in gen()) {
                try {
                    if (x == 2) { throw "skip"; }
                    results = results + [x];
                } catch (e) {
                    results = results + ["caught"];
                }
            }
            results;
        """)
        assert r == [1, "caught", 3]

    def test_generator_throws(self):
        """Throw in generator during ASYNC_ITER_NEXT propagates to catch.
        Note: try-catch restores env snapshot, so in-loop mutations are lost."""
        r, _ = run("""
            async fn* gen() {
                yield 1;
                throw "gen_error";
                yield 3;
            }
            let caught = "none";
            try {
                for await (x in gen()) {}
            } catch (e) {
                caught = e;
            }
            caught;
        """)
        assert r == "gen_error"

    def test_error_in_await(self):
        """Await rejection in generator propagates to outer catch."""
        r, _ = run("""
            async fn fail() { throw "async_fail"; }
            async fn* gen() {
                yield 1;
                yield await fail();
            }
            let caught = "none";
            try {
                for await (x in gen()) {}
            } catch (e) {
                caught = e;
            }
            caught;
        """)
        assert r == "async_fail"


# ============================================================
# Section 7: State and accumulation
# ============================================================

class TestStateAccumulation:
    def test_accumulate_sum(self):
        r, _ = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) { yield i; i = i + 1; }
            }
            let sum = 0;
            for await (x in range(5)) { sum = sum + x; }
            sum;
        """)
        assert r == 10

    def test_state_across_iterations(self):
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; yield 3; }
            let prev = 0;
            let diffs = [];
            for await (x in gen()) {
                diffs = diffs + [x - prev];
                prev = x;
            }
            diffs;
        """)
        assert r == [1, 1, 1]

    def test_collect_to_hash(self):
        r, _ = run("""
            async fn* entries() {
                yield ["a", 1];
                yield ["b", 2];
            }
            let result = {};
            for await ([k, v] in entries()) {
                result[k] = v;
            }
            result;
        """)
        assert r == {"a": 1, "b": 2}


# ============================================================
# Section 8: Composition patterns
# ============================================================

class TestComposition:
    def test_map_with_for_await(self):
        r, _ = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) { yield i; i = i + 1; }
            }
            async fn* map(gen, f) {
                for await (x in gen) { yield f(x); }
            }
            let results = [];
            for await (x in map(range(5), fn(x) { return x * x; })) {
                results = results + [x];
            }
            results;
        """)
        assert r == [0, 1, 4, 9, 16]

    def test_filter_with_for_await(self):
        r, _ = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) { yield i; i = i + 1; }
            }
            async fn* filter(gen, pred) {
                for await (x in gen) {
                    if (pred(x)) { yield x; }
                }
            }
            let results = [];
            for await (x in filter(range(6), fn(x) { return x % 2 == 0; })) {
                results = results + [x];
            }
            results;
        """)
        assert r == [0, 2, 4]

    def test_take_with_for_await(self):
        r, _ = run("""
            async fn* naturals() {
                let i = 0;
                while (true) { yield i; i = i + 1; }
            }
            async fn* take(gen, n) {
                let count = 0;
                for await (x in gen) {
                    if (count >= n) { break; }
                    yield x;
                    count = count + 1;
                }
            }
            let results = [];
            for await (x in take(naturals(), 5)) {
                results = results + [x];
            }
            results;
        """)
        assert r == [0, 1, 2, 3, 4]

    def test_chained_composition(self):
        """map |> filter |> take pipeline using for await."""
        r, _ = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) { yield i; i = i + 1; }
            }
            async fn* map(gen, f) {
                for await (x in gen) { yield f(x); }
            }
            async fn* filter(gen, pred) {
                for await (x in gen) {
                    if (pred(x)) { yield x; }
                }
            }
            async fn* take(gen, n) {
                let count = 0;
                for await (x in gen) {
                    if (count >= n) { break; }
                    yield x;
                    count = count + 1;
                }
            }
            let pipeline = take(
                filter(
                    map(range(100), fn(x) { return x * x; }),
                    fn(x) { return x % 2 == 1; }
                ),
                4
            );
            let results = [];
            for await (x in pipeline) { results = results + [x]; }
            results;
        """)
        assert r == [1, 9, 25, 49]

    def test_zip_with_for_await(self):
        r, _ = run("""
            async fn* gen_a() { yield 1; yield 2; yield 3; }
            async fn* gen_b() { yield "a"; yield "b"; yield "c"; }
            let a_items = [];
            for await (x in gen_a()) { a_items = a_items + [x]; }
            let b_items = [];
            for await (x in gen_b()) { b_items = b_items + [x]; }
            let zipped = [];
            let i = 0;
            while (i < len(a_items)) {
                zipped = zipped + [[a_items[i], b_items[i]]];
                i = i + 1;
            }
            zipped;
        """)
        assert r == [[1, "a"], [2, "b"], [3, "c"]]


# ============================================================
# Section 9: Print and side effects
# ============================================================

class TestSideEffects:
    def test_print_in_body(self):
        r, out = run("""
            async fn* gen() { yield 1; yield 2; }
            for await (x in gen()) { print x; }
        """)
        assert out == ["1", "2"]

    def test_side_effects_order(self):
        r, out = run("""
            async fn* gen() {
                print "yield 1";
                yield 1;
                print "yield 2";
                yield 2;
            }
            for await (x in gen()) {
                print x;
            }
        """)
        assert out == ["yield 1", "1", "yield 2", "2"]


# ============================================================
# Section 10: With classes
# ============================================================

class TestWithClasses:
    def test_class_with_for_await(self):
        r, _ = run("""
            class Point {
                init(x, y) { this.x = x; this.y = y; }
            }
            async fn* points() {
                yield Point(1, 2);
                yield Point(3, 4);
            }
            let sums = [];
            for await (p in points()) { sums = sums + [p.x + p.y]; }
            sums;
        """)
        assert r == [3, 7]

    def test_class_method_async_gen(self):
        """Async gen defined outside class, called with class data."""
        r, _ = run("""
            class Range {
                init(s, e) { this.s = s; this.e = e; }
            }
            async fn* iter_range(range) {
                let i = range.s;
                while (i < range.e) { yield i; i = i + 1; }
            }
            let range = Range(1, 4);
            let results = [];
            for await (x in iter_range(range)) { results = results + [x]; }
            results;
        """)
        assert r == [1, 2, 3]


# ============================================================
# Section 11: With other language features
# ============================================================

class TestLanguageFeatures:
    def test_with_string_interpolation(self):
        r, _ = run("""
            async fn* names() { yield "alice"; yield "bob"; }
            let results = [];
            for await (name in names()) {
                results = results + [f"hello ${name}"];
            }
            results;
        """)
        assert r == ["hello alice", "hello bob"]

    def test_with_optional_chaining(self):
        r, _ = run("""
            async fn* gen() {
                yield {name: "alice"};
                yield null;
                yield {name: "bob"};
            }
            let results = [];
            for await (item in gen()) {
                results = results + [item?.name];
            }
            results;
        """)
        assert r == ["alice", None, "bob"]

    def test_with_null_coalescing(self):
        r, _ = run("""
            async fn* gen() { yield null; yield 1; yield null; yield 2; }
            let results = [];
            for await (x in gen()) {
                results = results + [x ?? 0];
            }
            results;
        """)
        assert r == [0, 1, 0, 2]

    def test_with_spread(self):
        r, _ = run("""
            async fn* gen() { yield [1, 2]; yield [3, 4]; }
            let all = [];
            for await (arr in gen()) {
                all = [...all, ...arr];
            }
            all;
        """)
        assert r == [1, 2, 3, 4]

    def test_with_pipe_operator(self):
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; yield 3; }
            fn double(x) { return x * 2; }
            let results = [];
            for await (x in gen()) {
                results = results + [x |> double];
            }
            results;
        """)
        assert r == [2, 4, 6]


# ============================================================
# Section 12: Edge cases
# ============================================================

class TestEdgeCases:
    def test_generator_with_return(self):
        """Return in generator ends iteration, value is discarded."""
        r, _ = run("""
            async fn* gen() { yield 1; return 99; yield 3; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [1]

    def test_multiple_for_await_same_gen_fn(self):
        """Each call creates a fresh generator."""
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; }
            let a = [];
            for await (x in gen()) { a = a + [x]; }
            let b = [];
            for await (x in gen()) { b = b + [x]; }
            [a, b];
        """)
        assert r == [[1, 2], [1, 2]]

    def test_yield_arrays(self):
        r, _ = run("""
            async fn* gen() { yield [1, 2]; yield [3]; yield []; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [[1, 2], [3], []]

    def test_yield_hashes(self):
        r, _ = run("""
            async fn* gen() { yield {a: 1}; yield {b: 2}; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [{"a": 1}, {"b": 2}]

    def test_large_iteration(self):
        r, _ = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) { yield i; i = i + 1; }
            }
            let sum = 0;
            for await (x in range(100)) { sum = sum + x; }
            sum;
        """)
        assert r == 4950

    def test_yield_boolean_values(self):
        r, _ = run("""
            async fn* gen() { yield true; yield false; yield true; }
            let results = [];
            for await (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [True, False, True]


# ============================================================
# Section 13: For-await inside async functions
# ============================================================

class TestInsideAsync:
    def test_for_await_in_async_fn(self):
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; yield 3; }
            async fn collect() {
                let results = [];
                for await (x in gen()) { results = results + [x]; }
                return results;
            }
            await collect();
        """)
        assert r == [1, 2, 3]

    def test_for_await_in_async_with_await(self):
        r, _ = run("""
            async fn double(x) { return x * 2; }
            async fn* gen() { yield 1; yield 2; yield 3; }
            async fn process() {
                let results = [];
                for await (x in gen()) {
                    let doubled = await double(x);
                    results = results + [doubled];
                }
                return results;
            }
            await process();
        """)
        assert r == [2, 4, 6]


# ============================================================
# Section 14: Fibonacci and algorithmic patterns
# ============================================================

class TestAlgorithmic:
    def test_fibonacci(self):
        r, _ = run("""
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
            async fn* take(gen, n) {
                let count = 0;
                for await (x in gen) {
                    if (count >= n) { break; }
                    yield x;
                    count = count + 1;
                }
            }
            let results = [];
            for await (x in take(fib(), 8)) { results = results + [x]; }
            results;
        """)
        assert r == [0, 1, 1, 2, 3, 5, 8, 13]

    def test_sieve_like(self):
        """Generate primes using a simple trial division approach."""
        r, _ = run("""
            async fn* range(start, end) {
                let i = start;
                while (i < end) { yield i; i = i + 1; }
            }
            fn is_prime(n) {
                if (n < 2) { return false; }
                let i = 2;
                while (i * i <= n) {
                    if (n % i == 0) { return false; }
                    i = i + 1;
                }
                return true;
            }
            async fn* primes(max) {
                for await (n in range(2, max)) {
                    if (is_prime(n)) { yield n; }
                }
            }
            let results = [];
            for await (p in primes(20)) { results = results + [p]; }
            results;
        """)
        assert r == [2, 3, 5, 7, 11, 13, 17, 19]

    def test_running_average(self):
        r, _ = run("""
            async fn* data() { yield 10; yield 20; yield 30; yield 40; }
            let sum = 0;
            let count = 0;
            let avgs = [];
            for await (x in data()) {
                sum = sum + x;
                count = count + 1;
                avgs = avgs + [sum / count];
            }
            avgs;
        """)
        assert r == [10.0, 15.0, 20.0, 25.0]


# ============================================================
# Section 15: Finally blocks
# ============================================================

class TestFinally:
    def test_try_finally_around_for_await(self):
        r, out = run("""
            async fn* gen() { yield 1; yield 2; }
            let results = [];
            try {
                for await (x in gen()) { results = results + [x]; }
            } finally {
                print "done";
            }
            results;
        """)
        assert r == [1, 2]
        assert out == ["done"]

    def test_break_with_finally(self):
        r, out = run("""
            async fn* count() {
                let i = 0;
                while (true) { yield i; i = i + 1; }
            }
            let results = [];
            try {
                for await (x in count()) {
                    if (x >= 2) { break; }
                    results = results + [x];
                }
            } finally {
                print "cleanup";
            }
            results;
        """)
        assert r == [0, 1]
        assert out == ["cleanup"]


# ============================================================
# Section 16: Backward compatibility (C064 tests)
# ============================================================

class TestBackwardCompat:
    """Verify that existing for-in and async generator features still work."""

    def test_regular_for_in(self):
        r, _ = run("""
            let results = [];
            for (x in [1, 2, 3]) { results = results + [x]; }
            results;
        """)
        assert r == [1, 2, 3]

    def test_for_in_with_key_value(self):
        r, _ = run("""
            let h = {a: 1, b: 2};
            let keys = [];
            for (k, v in h) { keys = keys + [k]; }
            keys;
        """)
        assert sorted(r) == ["a", "b"]

    def test_sync_generator(self):
        r, _ = run("""
            fn* gen() { yield 1; yield 2; yield 3; }
            let results = [];
            for (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [1, 2, 3]

    def test_async_fn_still_works(self):
        r, _ = run("""
            async fn add(a, b) { return a + b; }
            await add(1, 2);
        """)
        assert r == 3

    def test_async_gen_with_next(self):
        """Old-style manual next() still works."""
        r, _ = run("""
            async fn* gen() { yield 10; yield 20; }
            let g = gen();
            let a = await next(g);
            let b = await next(g);
            [a, b];
        """)
        assert r == [10, 20]

    def test_eager_for_in_async_gen(self):
        """Regular for-in on async gen still eagerly collects."""
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; yield 3; }
            let results = [];
            for (x in gen()) { results = results + [x]; }
            results;
        """)
        assert r == [1, 2, 3]

    def test_destructuring_for_in(self):
        r, _ = run("""
            let pairs = [[1, 2], [3, 4]];
            let sums = [];
            for ([a, b] in pairs) { sums = sums + [a + b]; }
            sums;
        """)
        assert r == [3, 7]

    def test_closures_still_work(self):
        r, _ = run("""
            fn make_adder(n) {
                return fn(x) { return x + n; };
            }
            let add5 = make_adder(5);
            add5(10);
        """)
        assert r == 15

    def test_classes_still_work(self):
        r, _ = run("""
            class Dog {
                init(name) { this.name = name; }
                speak() { return f"${this.name} says woof"; }
            }
            let d = Dog("Rex");
            d.speak();
        """)
        assert r == "Rex says woof"

    def test_modules_still_work(self):
        r, _ = run("""
            let x = 42;
            export fn get_x() { return x; }
            x;
        """)
        assert r == 42


# ============================================================
# Section 17: Complex patterns
# ============================================================

class TestComplexPatterns:
    def test_async_gen_yielding_async_gen_results(self):
        """An async gen that consumes another async gen via for-await and re-yields."""
        r, _ = run("""
            async fn* source() { yield 1; yield 2; yield 3; }
            async fn* doubled(gen) {
                for await (x in gen) { yield x * 2; }
            }
            async fn* plus_one(gen) {
                for await (x in gen) { yield x + 1; }
            }
            let results = [];
            for await (x in plus_one(doubled(source()))) {
                results = results + [x];
            }
            results;
        """)
        assert r == [3, 5, 7]

    def test_multiple_generators_interleaved(self):
        r, _ = run("""
            async fn* gen_a() { yield "a1"; yield "a2"; }
            async fn* gen_b() { yield "b1"; yield "b2"; }
            let results = [];
            let ga = gen_a();
            let gb = gen_b();
            let a1 = await next(ga);
            let b1 = await next(gb);
            let a2 = await next(ga);
            let b2 = await next(gb);
            [a1, b1, a2, b2];
        """)
        assert r == ["a1", "b1", "a2", "b2"]

    def test_recursive_async_gen(self):
        r, _ = run("""
            async fn* countdown(n) {
                if (n <= 0) { return; }
                yield n;
                for await (x in countdown(n - 1)) { yield x; }
            }
            let results = [];
            for await (x in countdown(4)) { results = results + [x]; }
            results;
        """)
        assert r == [4, 3, 2, 1]

    def test_flat_map(self):
        r, _ = run("""
            async fn* of(...args) {
                for (x in args) { yield x; }
            }
            async fn* flat_map(gen, f) {
                for await (x in gen) {
                    for await (y in f(x)) { yield y; }
                }
            }
            async fn* expand(n) { yield n; yield n * 10; }
            let results = [];
            for await (x in flat_map(of(1, 2, 3), expand)) {
                results = results + [x];
            }
            results;
        """)
        assert r == [1, 10, 2, 20, 3, 30]

    def test_enumerate_pattern(self):
        r, _ = run("""
            async fn* items() { yield "a"; yield "b"; yield "c"; }
            async fn* enumerate(gen) {
                let i = 0;
                for await (x in gen) {
                    yield [i, x];
                    i = i + 1;
                }
            }
            let results = [];
            for await ([idx, val] in enumerate(items())) {
                results = results + [f"${idx}:${val}"];
            }
            results;
        """)
        assert r == ["0:a", "1:b", "2:c"]


# ============================================================
# Section 18: Stress/regression
# ============================================================

class TestStress:
    def test_deep_nesting(self):
        r, _ = run("""
            async fn* gen() { yield 1; yield 2; }
            let total = 0;
            for await (a in gen()) {
                for await (b in gen()) {
                    for await (c in gen()) {
                        total = total + a * 100 + b * 10 + c;
                    }
                }
            }
            total;
        """)
        # a=1: b=1: c=1,2 -> 111+112; b=2: c=1,2 -> 121+122
        # a=2: b=1: c=1,2 -> 211+212; b=2: c=1,2 -> 221+222
        assert r == 111 + 112 + 121 + 122 + 211 + 212 + 221 + 222

    def test_many_yields(self):
        r, _ = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) { yield i; i = i + 1; }
            }
            let count = 0;
            for await (x in range(200)) { count = count + 1; }
            count;
        """)
        assert r == 200

    def test_generator_reuse_after_exhaustion(self):
        r, _ = run("""
            async fn* gen() { yield 1; }
            let g = gen();
            let results = [];
            for await (x in g) { results = results + [x]; }
            for await (x in g) { results = results + [x]; }
            results;
        """)
        # Second loop should produce nothing (generator already done)
        assert r == [1]

    def test_break_in_nested_for_await(self):
        r, _ = run("""
            async fn* outer() { yield 1; yield 2; yield 3; }
            async fn* inner() { yield 10; yield 20; yield 30; }
            let results = [];
            for await (a in outer()) {
                for await (b in inner()) {
                    if (b >= 20) { break; }
                    results = results + [a * 100 + b];
                }
            }
            results;
        """)
        assert r == [110, 210, 310]
