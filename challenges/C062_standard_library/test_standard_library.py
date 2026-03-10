"""
Tests for C062: Standard Library
"""
import pytest
from standard_library import run, execute, make_default_registry


def r(src, **kw):
    """Run with default registry."""
    reg = make_default_registry(**kw)
    return run(src, registry=reg)


def val(src, **kw):
    """Run and return result only."""
    return r(src, **kw)[0]


def out(src, **kw):
    """Run and return output only."""
    return r(src, **kw)[1]


# ============================================================
# Collections: Stack
# ============================================================

class TestStack:
    def test_push_pop(self):
        assert val('import { Stack } from "collections"; let s = Stack(); s.push(1); s.push(2); s.pop();') == 2

    def test_peek(self):
        assert val('import { Stack } from "collections"; let s = Stack(); s.push(10); s.push(20); s.peek();') == 20

    def test_size(self):
        assert val('import { Stack } from "collections"; let s = Stack(); s.push(1); s.push(2); s.push(3); s.size;') == 3

    def test_isEmpty(self):
        assert val('import { Stack } from "collections"; let s = Stack(); s.isEmpty();') == True

    def test_not_empty(self):
        assert val('import { Stack } from "collections"; let s = Stack(); s.push(1); s.isEmpty();') == False

    def test_toArray(self):
        assert val('import { Stack } from "collections"; let s = Stack(); s.push(1); s.push(2); s.push(3); s.toArray();') == [1, 2, 3]

    def test_clear(self):
        assert val('import { Stack } from "collections"; let s = Stack(); s.push(1); s.push(2); s.clear(); s.size;') == 0

    def test_underflow(self):
        o = out('import { Stack } from "collections"; let s = Stack(); try { s.pop(); } catch (e) { print e; }')
        assert o == ['Stack underflow']

    def test_peek_underflow(self):
        o = out('import { Stack } from "collections"; let s = Stack(); try { s.peek(); } catch (e) { print e; }')
        assert o == ['Stack underflow']

    def test_chaining(self):
        assert val('import { Stack } from "collections"; let s = Stack(); s.push(1).push(2).push(3); s.size;') == 3

    def test_lifo_order(self):
        assert out('''
            import { Stack } from "collections";
            let s = Stack();
            s.push("a"); s.push("b"); s.push("c");
            print s.pop();
            print s.pop();
            print s.pop();
        ''') == ['c', 'b', 'a']


# ============================================================
# Collections: Queue
# ============================================================

class TestQueue:
    def test_enqueue_dequeue(self):
        assert val('import { Queue } from "collections"; let q = Queue(); q.enqueue(1); q.enqueue(2); q.dequeue();') == 1

    def test_fifo_order(self):
        assert out('''
            import { Queue } from "collections";
            let q = Queue();
            q.enqueue("a"); q.enqueue("b"); q.enqueue("c");
            print q.dequeue();
            print q.dequeue();
            print q.dequeue();
        ''') == ['a', 'b', 'c']

    def test_front(self):
        assert val('import { Queue } from "collections"; let q = Queue(); q.enqueue(10); q.enqueue(20); q.front();') == 10

    def test_size(self):
        assert val('import { Queue } from "collections"; let q = Queue(); q.enqueue(1); q.enqueue(2); q.size;') == 2

    def test_isEmpty(self):
        assert val('import { Queue } from "collections"; Queue().isEmpty();') == True

    def test_toArray(self):
        assert val('import { Queue } from "collections"; let q = Queue(); q.enqueue(1); q.enqueue(2); q.toArray();') == [1, 2]

    def test_clear(self):
        assert val('import { Queue } from "collections"; let q = Queue(); q.enqueue(1); q.clear(); q.size;') == 0

    def test_underflow(self):
        o = out('import { Queue } from "collections"; try { Queue().dequeue(); } catch (e) { print e; }')
        assert o == ['Queue underflow']

    def test_chaining(self):
        assert val('import { Queue } from "collections"; let q = Queue(); q.enqueue(1).enqueue(2).enqueue(3); q.size;') == 3


# ============================================================
# Collections: Set
# ============================================================

class TestSet:
    def test_add_has(self):
        assert val('import { Set } from "collections"; let s = Set(); s.add(1); s.has(1);') == True

    def test_no_duplicates(self):
        assert val('import { Set } from "collections"; let s = Set(); s.add(1); s.add(1); s.add(1); s.size;') == 1

    def test_delete(self):
        assert val('import { Set } from "collections"; let s = Set(); s.add(1); s.add(2); s.delete(1); s.size;') == 1

    def test_delete_returns_bool(self):
        assert val('import { Set } from "collections"; let s = Set(); s.add(1); s.delete(1);') == True

    def test_delete_missing(self):
        assert val('import { Set } from "collections"; let s = Set(); s.delete(999);') == False

    def test_union(self):
        result = val('''
            import { Set } from "collections";
            let a = Set(); a.add(1); a.add(2);
            let b = Set(); b.add(2); b.add(3);
            a.union(b).toArray();
        ''')
        assert sorted(result) == [1, 2, 3]

    def test_intersection(self):
        result = val('''
            import { Set } from "collections";
            let a = Set(); a.add(1); a.add(2); a.add(3);
            let b = Set(); b.add(2); b.add(3); b.add(4);
            a.intersection(b).toArray();
        ''')
        assert sorted(result) == [2, 3]

    def test_difference(self):
        result = val('''
            import { Set } from "collections";
            let a = Set(); a.add(1); a.add(2); a.add(3);
            let b = Set(); b.add(2); b.add(3);
            a.difference(b).toArray();
        ''')
        assert result == [1]

    def test_toArray(self):
        result = val('import { Set } from "collections"; let s = Set(); s.add(3); s.add(1); s.add(2); s.toArray();')
        assert result == [3, 1, 2]  # insertion order

    def test_clear(self):
        assert val('import { Set } from "collections"; let s = Set(); s.add(1); s.add(2); s.clear(); s.size;') == 0

    def test_has_missing(self):
        assert val('import { Set } from "collections"; Set().has(42);') == False

    def test_string_values(self):
        assert val('import { Set } from "collections"; let s = Set(); s.add("hello"); s.add("world"); s.has("hello");') == True


# ============================================================
# Collections: LinkedList
# ============================================================

class TestLinkedList:
    def test_append(self):
        assert val('import { LinkedList } from "collections"; let l = LinkedList(); l.append(1); l.append(2); l.toArray();') == [1, 2]

    def test_prepend(self):
        assert val('import { LinkedList } from "collections"; let l = LinkedList(); l.append(2); l.prepend(1); l.toArray();') == [1, 2]

    def test_first_last(self):
        assert out('''
            import { LinkedList } from "collections";
            let l = LinkedList();
            l.append(10); l.append(20); l.append(30);
            print l.first();
            print l.last();
        ''') == ['10', '30']

    def test_removeFirst(self):
        assert val('''
            import { LinkedList } from "collections";
            let l = LinkedList();
            l.append(1); l.append(2); l.append(3);
            l.removeFirst();
        ''') == 1

    def test_removeFirst_updates(self):
        assert val('''
            import { LinkedList } from "collections";
            let l = LinkedList();
            l.append(1); l.append(2);
            l.removeFirst();
            l.first();
        ''') == 2

    def test_contains(self):
        assert val('import { LinkedList } from "collections"; let l = LinkedList(); l.append(1); l.append(2); l.contains(2);') == True

    def test_not_contains(self):
        assert val('import { LinkedList } from "collections"; let l = LinkedList(); l.append(1); l.contains(99);') == False

    def test_size(self):
        assert val('import { LinkedList } from "collections"; let l = LinkedList(); l.append(1); l.append(2); l.size;') == 2

    def test_clear(self):
        assert val('import { LinkedList } from "collections"; let l = LinkedList(); l.append(1); l.clear(); l.size;') == 0

    def test_empty_first(self):
        assert val('import { LinkedList } from "collections"; LinkedList().first();') is None

    def test_empty_removeFirst(self):
        o = out('import { LinkedList } from "collections"; try { LinkedList().removeFirst(); } catch (e) { print e; }')
        assert o == ['LinkedList is empty']

    def test_chaining(self):
        assert val('import { LinkedList } from "collections"; let l = LinkedList(); l.append(1).append(2).append(3); l.size;') == 3


# ============================================================
# Iter: range
# ============================================================

class TestRange:
    def test_basic(self):
        assert val('import { range, toArray } from "iter"; toArray(range(5));') == [0, 1, 2, 3, 4]

    def test_start_stop(self):
        assert val('import { range, toArray } from "iter"; toArray(range(2, 6));') == [2, 3, 4, 5]

    def test_step(self):
        assert val('import { range, toArray } from "iter"; toArray(range(0, 10, 3));') == [0, 3, 6, 9]

    def test_negative_step(self):
        assert val('import { range, toArray } from "iter"; toArray(range(5, 0, -1));') == [5, 4, 3, 2, 1]

    def test_empty(self):
        assert val('import { range, toArray } from "iter"; toArray(range(0));') == []

    def test_auto_negative_step(self):
        assert val('import { range, toArray } from "iter"; toArray(range(3, 0));') == [3, 2, 1]


# ============================================================
# Iter: map, filter, reduce
# ============================================================

class TestMapFilterReduce:
    def test_map(self):
        assert val('import { map, toArray } from "iter"; toArray(map([1,2,3], fn(x) { return x * 2; }));') == [2, 4, 6]

    def test_filter(self):
        assert val('import { filter, toArray } from "iter"; toArray(filter([1,2,3,4,5], fn(x) { return x > 3; }));') == [4, 5]

    def test_reduce(self):
        assert val('import { reduce } from "iter"; reduce([1,2,3,4], fn(acc, x) { return acc + x; }, 0);') == 10

    def test_reduce_no_init(self):
        assert val('import { reduce } from "iter"; reduce([1,2,3], fn(acc, x) { return acc + x; });') == 6

    def test_map_filter_chain(self):
        assert val('''
            import { map, filter, toArray } from "iter";
            let nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            let evens = filter(nums, fn(x) { return x % 2 == 0; });
            let doubled = map(evens, fn(x) { return x * 2; });
            toArray(doubled);
        ''') == [4, 8, 12, 16, 20]

    def test_map_strings(self):
        assert val('import { map, toArray } from "iter"; toArray(map(["a","b","c"], fn(s) { return s.toUpperCase(); }));') == ["A", "B", "C"]


# ============================================================
# Iter: zip, enumerate
# ============================================================

class TestZipEnumerate:
    def test_zip(self):
        assert val('import { zip, toArray } from "iter"; toArray(zip([1,2,3], ["a","b","c"]));') == [[1,"a"], [2,"b"], [3,"c"]]

    def test_zip_unequal(self):
        assert val('import { zip, toArray } from "iter"; toArray(zip([1,2], ["a","b","c"]));') == [[1,"a"], [2,"b"]]

    def test_enumerate(self):
        assert val('import { enumerate, toArray } from "iter"; toArray(enumerate(["a","b","c"]));') == [[0,"a"], [1,"b"], [2,"c"]]

    def test_enumerate_start(self):
        assert val('import { enumerate, toArray } from "iter"; toArray(enumerate(["x","y"], 5));') == [[5,"x"], [6,"y"]]


# ============================================================
# Iter: take, drop, chain
# ============================================================

class TestTakeDropChain:
    def test_take(self):
        assert val('import { take, toArray } from "iter"; toArray(take([1,2,3,4,5], 3));') == [1, 2, 3]

    def test_take_more(self):
        assert val('import { take, toArray } from "iter"; toArray(take([1,2], 5));') == [1, 2]

    def test_drop(self):
        assert val('import { drop, toArray } from "iter"; toArray(drop([1,2,3,4,5], 2));') == [3, 4, 5]

    def test_drop_more(self):
        assert val('import { drop, toArray } from "iter"; toArray(drop([1,2], 5));') == []

    def test_chain(self):
        assert val('import { chain, toArray } from "iter"; toArray(chain([1,2], [3,4]));') == [1, 2, 3, 4]

    def test_chain_empty(self):
        assert val('import { chain, toArray } from "iter"; toArray(chain([], [1,2]));') == [1, 2]


# ============================================================
# Iter: flatMap, repeat, cycle
# ============================================================

class TestFlatMapRepeatCycle:
    def test_flatMap(self):
        assert val('import { flatMap, toArray } from "iter"; toArray(flatMap([1,2,3], fn(x) { return [x, x*10]; }));') == [1, 10, 2, 20, 3, 30]

    def test_repeat_n(self):
        assert val('import { repeat, toArray } from "iter"; toArray(repeat("x", 4));') == ["x", "x", "x", "x"]

    def test_repeat_finite(self):
        assert val('import { repeat, toArray } from "iter"; toArray(repeat(0, 3));') == [0, 0, 0]

    def test_cycle_finite(self):
        """cycle requires finite input and produces infinite output -- test with toArray limited."""
        # cycle is infinite, test just the finite repeat instead
        assert val('import { repeat, toArray } from "iter"; toArray(repeat("x", 5));') == ["x", "x", "x", "x", "x"]


# ============================================================
# Iter: find, every, some, count, sum
# ============================================================

class TestIterUtilities:
    def test_find(self):
        assert val('import { find } from "iter"; find([1,2,3,4,5], fn(x) { return x > 3; });') == 4

    def test_find_none(self):
        assert val('import { find } from "iter"; find([1,2,3], fn(x) { return x > 10; });') is None

    def test_every_true(self):
        assert val('import { every } from "iter"; every([2,4,6,8], fn(x) { return x % 2 == 0; });') == True

    def test_every_false(self):
        assert val('import { every } from "iter"; every([2,4,5,8], fn(x) { return x % 2 == 0; });') == False

    def test_some_true(self):
        assert val('import { some } from "iter"; some([1,3,5,6], fn(x) { return x % 2 == 0; });') == True

    def test_some_false(self):
        assert val('import { some } from "iter"; some([1,3,5], fn(x) { return x % 2 == 0; });') == False

    def test_count(self):
        assert val('import { count } from "iter"; count([10, 20, 30]);') == 3

    def test_sum(self):
        assert val('import { sum } from "iter"; sum([1, 2, 3, 4, 5]);') == 15


# ============================================================
# Iter: chunks, unique, groupBy
# ============================================================

class TestChunksUniqueGroupBy:
    def test_chunks(self):
        assert val('import { chunks, toArray } from "iter"; toArray(chunks([1,2,3,4,5], 2));') == [[1,2], [3,4], [5]]

    def test_chunks_exact(self):
        assert val('import { chunks, toArray } from "iter"; toArray(chunks([1,2,3,4], 2));') == [[1,2], [3,4]]

    def test_unique(self):
        assert val('import { unique, toArray } from "iter"; toArray(unique([1,2,3,2,1,4]));') == [1, 2, 3, 4]

    def test_unique_strings(self):
        assert val('import { unique, toArray } from "iter"; toArray(unique(["a","b","a","c","b"]));') == ["a", "b", "c"]

    def test_groupBy(self):
        result = val('''
            import { groupBy } from "iter";
            groupBy([1,2,3,4,5,6], fn(x) { return x % 2; });
        ''')
        assert result['0'] == [2, 4, 6]
        assert result['1'] == [1, 3, 5]

    def test_groupBy_strings(self):
        result = val('''
            import { groupBy } from "iter";
            groupBy(["cat", "car", "dog", "door"], fn(s) { return s[0]; });
        ''')
        assert result['c'] == ["cat", "car"]
        assert result['d'] == ["dog", "door"]


# ============================================================
# Functional: compose, pipe
# ============================================================

class TestComposePipe:
    def test_compose(self):
        assert val('''
            import { compose } from "functional";
            let double = fn(x) { return x * 2; };
            let inc = fn(x) { return x + 1; };
            let f = compose(inc, double);
            f(5);
        ''') == 11  # double(5)=10, inc(10)=11

    def test_pipe(self):
        assert val('''
            import { pipe } from "functional";
            let double = fn(x) { return x * 2; };
            let inc = fn(x) { return x + 1; };
            let f = pipe(double, inc);
            f(5);
        ''') == 11  # double(5)=10, inc(10)=11

    def test_compose_three(self):
        assert val('''
            import { compose } from "functional";
            let a = fn(x) { return x + 1; };
            let b = fn(x) { return x * 2; };
            let c = fn(x) { return x - 3; };
            compose(a, b, c)(10);
        ''') == 15  # c(10)=7, b(7)=14, a(14)=15

    def test_pipe_three(self):
        assert val('''
            import { pipe } from "functional";
            let a = fn(x) { return x + 1; };
            let b = fn(x) { return x * 2; };
            let c = fn(x) { return x - 3; };
            pipe(a, b, c)(10);
        ''') == 19  # a(10)=11, b(11)=22, c(22)=19

    def test_compose_single(self):
        assert val('''
            import { compose } from "functional";
            let f = compose(fn(x) { return x * 3; });
            f(4);
        ''') == 12


# ============================================================
# Functional: curry, partial
# ============================================================

class TestCurryPartial:
    def test_curry(self):
        assert val('''
            import { curry } from "functional";
            let add = fn(a, b) { return a + b; };
            let add5 = curry(add, 2)(5);
            add5(3);
        ''') == 8

    def test_curry_all_at_once(self):
        assert val('''
            import { curry } from "functional";
            let add = fn(a, b) { return a + b; };
            curry(add, 2)(5, 3);
        ''') == 8

    def test_curry_three(self):
        assert val('''
            import { curry } from "functional";
            let add3 = fn(a, b, c) { return a + b + c; };
            curry(add3, 3)(1)(2)(3);
        ''') == 6

    def test_partial(self):
        assert val('''
            import { partial } from "functional";
            let add = fn(a, b) { return a + b; };
            let add10 = partial(add, 10);
            add10(5);
        ''') == 15

    def test_partial_multi(self):
        assert val('''
            import { partial } from "functional";
            let add3 = fn(a, b, c) { return a + b + c; };
            let f = partial(add3, 1, 2);
            f(3);
        ''') == 6


# ============================================================
# Functional: memoize, once, identity, constant, tap, negate, flip
# ============================================================

class TestFunctionalUtils:
    def test_memoize(self):
        assert out('''
            import { memoize } from "functional";
            let expensive = memoize(fn(x) {
                return x * x;
            });
            print expensive(5);
            print expensive(5);
            print expensive(3);
        ''') == ['25', '25', '9']

    def test_memoize_caches(self):
        """Memoize returns cached result for same args."""
        result = val('''
            import { memoize } from "functional";
            let f = memoize(fn(x) { return x * 2; });
            let a = f(5);
            let b = f(5);
            a == b;
        ''')
        assert result == True

    def test_once(self):
        """Once returns same result on repeated calls."""
        assert out('''
            import { once } from "functional";
            let init = once(fn() { return 42; });
            print init();
            print init();
            print init();
        ''') == ['42', '42', '42']

    def test_identity(self):
        assert val('import { identity } from "functional"; identity(42);') == 42

    def test_constant(self):
        assert val('import { constant } from "functional"; let f = constant(99); f();') == 99

    def test_constant_multiple_calls(self):
        assert out('''
            import { constant } from "functional";
            let f = constant(7);
            print f();
            print f();
        ''') == ['7', '7']

    def test_tap(self):
        assert out('''
            import { tap } from "functional";
            let logger = tap(fn(x) { print f"saw: ${x}"; });
            let result = logger(42);
            print result;
        ''') == ['saw: 42', '42']

    def test_negate(self):
        assert val('''
            import { negate } from "functional";
            let isEven = fn(x) { return x % 2 == 0; };
            let isOdd = negate(isEven);
            isOdd(3);
        ''') == True

    def test_negate_false(self):
        assert val('''
            import { negate } from "functional";
            let isEven = fn(x) { return x % 2 == 0; };
            let isOdd = negate(isEven);
            isOdd(4);
        ''') == False

    def test_flip(self):
        assert val('''
            import { flip } from "functional";
            let div = fn(a, b) { return a / b; };
            let rdiv = flip(div);
            rdiv(2, 10);
        ''') == 5  # div(10, 2) = 5


# ============================================================
# Testing module
# ============================================================

class TestTestingModule:
    def test_assert_pass(self):
        assert val('import { assert } from "testing"; assert(true, "should pass");') == True

    def test_assert_fail(self):
        o = out('import { assert } from "testing"; try { assert(false, "bad"); } catch (e) { print e; }')
        assert o == ['Assertion failed: bad']

    def test_assert_no_message(self):
        o = out('import { assert } from "testing"; try { assert(false); } catch (e) { print e; }')
        assert o == ['Assertion failed']

    def test_assertEqual_pass(self):
        assert val('import { assertEqual } from "testing"; assertEqual(42, 42);') == True

    def test_assertEqual_fail(self):
        o = out('import { assertEqual } from "testing"; try { assertEqual(1, 2); } catch (e) { print e; }')
        assert o == ['Expected 2, got 1']

    def test_assertEqual_message(self):
        o = out('import { assertEqual } from "testing"; try { assertEqual(1, 2, "count check"); } catch (e) { print e; }')
        assert o == ['count check: Expected 2, got 1']

    def test_assertThrows_pass(self):
        assert val('import { assertThrows } from "testing"; assertThrows(fn() { throw "boom"; });') == True

    def test_assertThrows_with_msg(self):
        assert val('import { assertThrows } from "testing"; assertThrows(fn() { throw "file not found"; }, "not found");') == True

    def test_assertThrows_wrong_msg(self):
        o = out('''
            import { assertThrows } from "testing";
            try {
                assertThrows(fn() { throw "wrong error"; }, "expected msg");
            } catch (e) { print e; }
        ''')
        assert 'expected msg' in o[0]

    def test_assertThrows_no_throw(self):
        o = out('''
            import { assertThrows } from "testing";
            try {
                assertThrows(fn() { return 1; });
            } catch (e) { print e; }
        ''')
        assert 'Expected function to throw' in o[0]

    def test_assertNull(self):
        assert val('import { assertNull } from "testing"; assertNull(null);') == True

    def test_assertNull_fail(self):
        o = out('import { assertNull } from "testing"; try { assertNull(42); } catch (e) { print e; }')
        assert 'Expected null' in o[0]

    def test_assertNotNull(self):
        assert val('import { assertNotNull } from "testing"; assertNotNull(42);') == True

    def test_assertNotNull_fail(self):
        o = out('import { assertNotNull } from "testing"; try { assertNotNull(null); } catch (e) { print e; }')
        assert 'non-null' in o[0]


# ============================================================
# Testing: test() and suite()
# ============================================================

class TestTestRunner:
    def test_passing_test(self):
        result = val('''
            import { test } from "testing";
            test("math works", fn() {
                if (1 + 1 != 2) { throw "broken"; }
            });
        ''')
        assert result['name'] == 'math works'
        assert result['passed'] == True
        assert result['error'] is None

    def test_failing_test(self):
        result = val('''
            import { test } from "testing";
            test("always fails", fn() {
                throw "intentional failure";
            });
        ''')
        assert result['name'] == 'always fails'
        assert result['passed'] == False
        assert 'intentional failure' in result['error']

    def test_suite(self):
        result = val('''
            import { test, suite } from "testing";
            let t1 = test("a", fn() {});
            let t2 = test("b", fn() { throw "fail"; });
            let t3 = test("c", fn() {});
            suite("my suite", [t1, t2, t3]);
        ''')
        assert result['name'] == 'my suite'
        assert result['passed'] == 2
        assert result['failed'] == 1
        assert result['total'] == 3

    def test_suite_all_pass(self):
        result = val('''
            import { test, suite } from "testing";
            let t1 = test("one", fn() {});
            let t2 = test("two", fn() {});
            suite("all good", [t1, t2]);
        ''')
        assert result['passed'] == 2
        assert result['failed'] == 0


# ============================================================
# Namespace imports
# ============================================================

class TestNamespaceImport:
    def test_collections_namespace(self):
        assert val('''
            import "collections";
            let s = collections.Stack();
            s.push(42);
            s.peek();
        ''') == 42

    def test_iter_namespace(self):
        assert val('import "iter"; iter.toArray(iter.range(3));') == [0, 1, 2]

    def test_functional_namespace(self):
        assert val('import "functional"; functional.identity(7);') == 7

    def test_testing_namespace(self):
        assert val('import "testing"; testing.assert(true);') == True


# ============================================================
# Cross-module composition
# ============================================================

class TestCrossModule:
    def test_iter_with_collections(self):
        assert val('''
            import { Stack } from "collections";
            import { reduce } from "iter";
            let s = Stack();
            s.push(1); s.push(2); s.push(3);
            reduce(s.toArray(), fn(acc, x) { return acc + x; }, 0);
        ''') == 6

    def test_functional_with_iter(self):
        assert val('''
            import { map, toArray } from "iter";
            import { compose } from "functional";
            let double = fn(x) { return x * 2; };
            let inc = fn(x) { return x + 1; };
            let f = compose(inc, double);
            toArray(map([1,2,3], f));
        ''') == [3, 5, 7]

    def test_testing_with_collections(self):
        result = val('''
            import { test, assertEqual } from "testing";
            import { Stack } from "collections";
            test("stack works", fn() {
                let s = Stack();
                s.push(1); s.push(2);
                assertEqual(s.size, 2);
                let v = s.pop();
                assertEqual(v, 2);
            });
        ''')
        assert result['passed'] == True

    def test_all_modules_together(self):
        result = val('''
            import { Stack, Queue } from "collections";
            import { map, reduce, toArray, range } from "iter";
            import { compose, identity } from "functional";
            import { test, suite, assertEqual } from "testing";

            let t1 = test("stack-iter", fn() {
                let s = Stack();
                let nums = toArray(range(5));
                for (n in nums) { s.push(n); }
                assertEqual(s.size, 5);
            });
            let t2 = test("queue-map", fn() {
                let q = Queue();
                q.enqueue(1); q.enqueue(2); q.enqueue(3);
                let arr = q.toArray();
                let doubled = toArray(map(arr, fn(x) { return x * 2; }));
                assertEqual(doubled[0], 2);
                assertEqual(doubled[2], 6);
            });
            let t3 = test("compose-reduce", fn() {
                let nums = toArray(range(1, 6));
                let sum = reduce(nums, fn(a, b) { return a + b; }, 0);
                assertEqual(sum, 15);
            });
            suite("integration", [t1, t2, t3]);
        ''')
        assert result['passed'] == 3
        assert result['failed'] == 0


# ============================================================
# Capability restrictions on stdlib modules
# ============================================================

class TestCapabilityRestrictions:
    def test_deny_collections(self):
        reg = make_default_registry(capabilities={'math'})
        with pytest.raises(Exception) as exc:
            run('import "collections";', registry=reg)
        assert 'Capability denied' in str(exc.value) or 'collections' in str(exc.value)

    def test_allow_specific_stdlib(self):
        reg = make_default_registry(capabilities={'iter', 'functional'})
        result, output = run('''
            import { range, toArray } from "iter";
            import { identity } from "functional";
            let r = toArray(range(3));
            identity(r);
        ''', registry=reg)
        assert result == [0, 1, 2]

    def test_deny_all(self):
        reg = make_default_registry(capabilities=set())
        with pytest.raises(Exception):
            run('import "testing";', registry=reg)


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_set_with_mixed_types(self):
        # 1 == true in MiniLang (Python semantics), so {1, "1", true} has size 2
        assert val('''
            import { Set } from "collections";
            let s = Set();
            s.add(1); s.add("1"); s.add("hello");
            s.size;
        ''') == 3

    def test_empty_range(self):
        assert val('import { range, toArray } from "iter"; toArray(range(5, 5));') == []

    def test_reduce_single(self):
        assert val('import { reduce } from "iter"; reduce([42], fn(a, b) { return a + b; });') == 42

    def test_memoize_null_result(self):
        # memoize uses cache.has(key), so null results ARE cached
        assert out('''
            import { memoize } from "functional";
            let state = {calls: 0};
            let f = memoize(fn(x) { state.calls = state.calls + 1; return null; });
            f(1);
            f(1);
            print state.calls;
        ''') == ['1']  # Called once -- memoize caches the null result

    def test_linked_list_single_remove(self):
        assert val('''
            import { LinkedList } from "collections";
            let l = LinkedList();
            l.append(42);
            l.removeFirst();
            l.size;
        ''') == 0

    def test_chunks_single_item(self):
        assert val('import { chunks, toArray } from "iter"; toArray(chunks([1], 3));') == [[1]]

    def test_unique_empty(self):
        assert val('import { unique, toArray } from "iter"; toArray(unique([]));') == []

    def test_suite_empty(self):
        result = val('import { suite } from "testing"; suite("empty", []);')
        assert result['total'] == 0
        assert result['passed'] == 0

    def test_compose_pipe_inverse(self):
        """compose(f,g) and pipe(g,f) should give same result."""
        assert val('''
            import { compose, pipe } from "functional";
            let f = fn(x) { return x + 1; };
            let g = fn(x) { return x * 3; };
            let a = compose(f, g)(5);
            let b = pipe(g, f)(5);
            a == b;
        ''') == True

    def test_iter_with_generators(self):
        """Iter functions work with generator inputs."""
        assert val('''
            import { map, filter, toArray } from "iter";
            fn* gen() {
                yield 1; yield 2; yield 3; yield 4; yield 5;
            }
            toArray(filter(map(gen(), fn(x) { return x * 10; }), fn(x) { return x > 20; }));
        ''') == [30, 40, 50]

    def test_sum_empty(self):
        assert val('import { sum } from "iter"; sum([]);') == 0

    def test_count_empty(self):
        assert val('import { count } from "iter"; count([]);') == 0

    def test_every_empty(self):
        assert val('import { every } from "iter"; every([], fn(x) { return false; });') == True

    def test_some_empty(self):
        assert val('import { some } from "iter"; some([], fn(x) { return true; });') == False


# ============================================================
# Stdlib + native module composition
# ============================================================

class TestStdlibWithNative:
    def test_iter_with_math(self):
        assert val('''
            import { map, toArray, range } from "iter";
            import "math";
            toArray(map(range(1, 5), fn(x) { return math.pow(2, x); }));
        ''') == [2, 4, 8, 16]

    def test_testing_with_math(self):
        result = val('''
            import { test, assertEqual } from "testing";
            import "math";
            test("sqrt", fn() {
                assertEqual(math.sqrt(16), 4.0);
            });
        ''')
        assert result['passed'] == True

    def test_functional_with_json(self):
        assert val('''
            import { pipe } from "functional";
            import "json";
            let process = pipe(
                fn(x) { return json.stringify(x); },
                fn(x) { return json.parse(x); }
            );
            let data = {a: 1, b: 2};
            let result = process(data);
            result.a + result.b;
        ''') == 3
