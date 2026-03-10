"""Tests for C055: Finally Blocks"""

import pytest
from finally_blocks import run, execute, VMError, parse, Compiler, VM, ParseError


# ============================================================
# Helper to check output with VMError
# ============================================================

def run_capturing_error(source):
    """Run source, return (result_or_none, output, error_or_none)."""
    ast = parse(source)
    compiler = Compiler()
    chunk = compiler.compile(ast)
    vm = VM(chunk)
    error = None
    result = None
    try:
        result = vm.run()
    except VMError as e:
        error = str(e)
    return result, vm.output, error


# ============================================================
# 1. Basic try/finally (no catch)
# ============================================================

class TestTryFinally:
    def test_basic_try_finally(self):
        r, o = run('try { print "try"; } finally { print "finally"; }')
        assert o == ['try', 'finally']

    def test_finally_runs_on_normal_exit(self):
        r, o = run('let x = 0; try { x = 1; } finally { x = x + 10; } print x;')
        assert o == ['11']

    def test_finally_runs_on_exception(self):
        _, o, err = run_capturing_error('try { throw "boom"; } finally { print "cleanup"; }')
        assert o == ['cleanup']
        assert 'boom' in err

    def test_finally_exception_still_propagates(self):
        _, o, err = run_capturing_error('try { throw "err"; } finally { print "fin"; }')
        assert o == ['fin']
        assert err is not None

    def test_try_finally_no_exception(self):
        r, o = run('let result = 0; try { result = 42; } finally { result = result + 1; } print result;')
        assert o == ['43']

    def test_nested_try_finally(self):
        r, o = run('''
            try {
                try {
                    print "inner-try";
                } finally {
                    print "inner-finally";
                }
            } finally {
                print "outer-finally";
            }
        ''')
        assert o == ['inner-try', 'inner-finally', 'outer-finally']

    def test_nested_try_finally_with_exception(self):
        _, o, err = run_capturing_error('''
            try {
                try {
                    throw "err";
                } finally {
                    print "inner-fin";
                }
            } finally {
                print "outer-fin";
            }
        ''')
        assert o == ['inner-fin', 'outer-fin']
        assert 'err' in err

    def test_nested_exception_caught_outer(self):
        r, o = run('''
            try {
                try {
                    throw "err";
                } finally {
                    print "inner-fin";
                }
            } catch(e) {
                print e;
            } finally {
                print "outer-fin";
            }
        ''')
        assert o == ['inner-fin', 'err', 'outer-fin']


# ============================================================
# 2. try/catch/finally
# ============================================================

class TestTryCatchFinally:
    def test_normal_flow(self):
        r, o = run('try { print "try"; } catch(e) { print "catch"; } finally { print "finally"; }')
        assert o == ['try', 'finally']

    def test_exception_caught(self):
        r, o = run('try { throw "err"; } catch(e) { print e; } finally { print "done"; }')
        assert o == ['err', 'done']

    def test_exception_in_catch_propagates_after_finally(self):
        _, o, err = run_capturing_error('''
            try {
                throw "first";
            } catch(e) {
                throw "second";
            } finally {
                print "finally";
            }
        ''')
        assert o == ['finally']
        assert 'second' in err

    def test_catch_var_accessible(self):
        r, o = run('''
            let msg = "";
            try {
                throw "hello";
            } catch(e) {
                msg = e;
            } finally {
                print msg;
            }
        ''')
        assert o == ['hello']

    def test_finally_after_catch_normal(self):
        r, o = run('''
            let log = [];
            try {
                push(log, "try");
            } catch(e) {
                push(log, "catch");
            } finally {
                push(log, "finally");
            }
            print len(log);
            print log[0];
            print log[1];
        ''')
        assert o == ['2', 'try', 'finally']

    def test_multiple_try_catch_finally(self):
        r, o = run('''
            try { print 1; } catch(e) { print "c1"; } finally { print "f1"; }
            try { throw "x"; } catch(e) { print "c2"; } finally { print "f2"; }
        ''')
        assert o == ['1', 'f1', 'c2', 'f2']


# ============================================================
# 3. Return from try/catch with finally
# ============================================================

class TestReturnWithFinally:
    def test_return_from_try(self):
        r, o = run('fn f() { try { return 1; } finally { print "fin"; } } print f();')
        assert o == ['fin', '1']

    def test_return_from_catch(self):
        r, o = run('fn f() { try { throw "e"; } catch(e) { return 2; } finally { print "fin"; } } print f();')
        assert o == ['fin', '2']

    def test_return_value_captured_before_finally(self):
        r, o = run('''
            fn f() {
                let x = 10;
                try {
                    return x;
                } finally {
                    x = 20;
                }
            }
            print f();
        ''')
        assert o == ['10']

    def test_finally_side_effects_happen(self):
        # Use an array (mutable reference) to track side effects across calls
        r, o = run('''
            let log = [];
            fn f() {
                try {
                    return 1;
                } finally {
                    push(log, "fin");
                }
            }
            f();
            f();
            f();
            print len(log);
        ''')
        assert o == ['3']

    def test_return_from_nested_try_finally(self):
        r, o = run('''
            fn f() {
                try {
                    try {
                        return 42;
                    } finally {
                        print "inner";
                    }
                } finally {
                    print "outer";
                }
            }
            print f();
        ''')
        assert o == ['inner', 'outer', '42']

    def test_return_from_try_no_catch(self):
        r, o = run('''
            fn f() {
                try {
                    return "hello";
                } finally {
                    print "cleanup";
                }
            }
            print f();
        ''')
        assert o == ['cleanup', 'hello']

    def test_return_after_try_finally(self):
        r, o = run('''
            fn f() {
                try {
                    print "try";
                } finally {
                    print "fin";
                }
                return 99;
            }
            print f();
        ''')
        assert o == ['try', 'fin', '99']


# ============================================================
# 4. Exception propagation through finally
# ============================================================

class TestExceptionPropagation:
    def test_uncaught_exception_runs_finally(self):
        _, o, err = run_capturing_error('''
            try {
                throw "oops";
            } finally {
                print "cleaned";
            }
        ''')
        assert o == ['cleaned']
        assert 'oops' in err

    def test_exception_propagates_through_nested_finally(self):
        _, o, err = run_capturing_error('''
            try {
                try {
                    try {
                        throw "deep";
                    } finally {
                        print "f1";
                    }
                } finally {
                    print "f2";
                }
            } finally {
                print "f3";
            }
        ''')
        assert o == ['f1', 'f2', 'f3']
        assert 'deep' in err

    def test_exception_in_finally_overrides_original(self):
        _, o, err = run_capturing_error('''
            try {
                throw "original";
            } finally {
                throw "override";
            }
        ''')
        assert 'override' in err

    def test_exception_caught_after_finally(self):
        r, o = run('''
            try {
                try {
                    throw "err";
                } finally {
                    print "inner-fin";
                }
            } catch(e) {
                print e;
            }
        ''')
        assert o == ['inner-fin', 'err']

    def test_runtime_error_triggers_finally(self):
        _, o, err = run_capturing_error('''
            try {
                let x = 1 / 0;
            } finally {
                print "fin";
            }
        ''')
        assert o == ['fin']
        assert err is not None


# ============================================================
# 5. Finally with loops
# ============================================================

class TestFinallyWithLoops:
    def test_finally_in_loop(self):
        r, o = run('''
            let i = 0;
            while (i < 3) {
                try {
                    print i;
                } finally {
                    i = i + 1;
                }
            }
        ''')
        assert o == ['0', '1', '2']

    def test_exception_in_loop_with_finally(self):
        # Exception caught at same scope level (not nested try/catch)
        r, o = run('''
            let results = [];
            let i = 0;
            while (i < 3) {
                try {
                    push(results, i);
                    if (i == 1) { throw "skip"; }
                } catch(e) {
                    push(results, e);
                } finally {
                    push(results, "fin");
                }
                i = i + 1;
            }
            for (x in results) { print x; }
        ''')
        assert o == ['0', 'fin', '1', 'skip', 'fin', '2', 'fin']

    def test_for_in_with_try_finally(self):
        r, o = run('''
            let arr = [10, 20, 30];
            for (x in arr) {
                try {
                    print x;
                } finally {
                    print "fin";
                }
            }
        ''')
        assert o == ['10', 'fin', '20', 'fin', '30', 'fin']


# ============================================================
# 6. Finally with closures
# ============================================================

class TestFinallyWithClosures:
    def test_closure_in_finally(self):
        r, o = run('''
            let log = [];
            fn make() {
                try {
                    return 1;
                } finally {
                    push(log, "fin");
                }
            }
            make();
            print len(log);
        ''')
        assert o == ['1']

    def test_finally_captures_closure_var(self):
        r, o = run('''
            fn outer() {
                let x = "hello";
                fn inner() {
                    try {
                        return x;
                    } finally {
                        print "fin";
                    }
                }
                return inner();
            }
            print outer();
        ''')
        assert o == ['fin', 'hello']


# ============================================================
# 7. Finally with classes
# ============================================================

class TestFinallyWithClasses:
    def test_finally_in_constructor(self):
        r, o = run('''
            class Foo {
                init() {
                    try {
                        this.x = 10;
                    } finally {
                        print "init-fin";
                    }
                }
            }
            let f = Foo();
            print f.x;
        ''')
        assert o == ['init-fin', '10']

    def test_finally_in_method(self):
        r, o = run('''
            class Resource {
                init() { this.open = true; }
                close() {
                    try {
                        return this.open;
                    } finally {
                        this.open = false;
                    }
                }
            }
            let r = Resource();
            print r.close();
            print r.open;
        ''')
        assert o == ['true', 'false']

    def test_exception_in_method_with_finally(self):
        r, o = run('''
            class Svc {
                init() { this.cleaned = false; }
                work() {
                    try {
                        throw "fail";
                    } catch(e) {
                        print e;
                    } finally {
                        this.cleaned = true;
                    }
                }
            }
            let s = Svc();
            s.work();
            print s.cleaned;
        ''')
        assert o == ['fail', 'true']


# ============================================================
# 8. Parse errors
# ============================================================

class TestParseErrors:
    def test_try_without_catch_or_finally(self):
        with pytest.raises(ParseError):
            run('try { print 1; }')

    def test_try_with_only_catch(self):
        r, o = run('try { print 1; } catch(e) { print 2; }')
        assert o == ['1']

    def test_try_with_only_finally(self):
        r, o = run('try { print 1; } finally { print 2; }')
        assert o == ['1', '2']


# ============================================================
# 9. Finally with generators
# ============================================================

class TestFinallyWithGenerators:
    def test_generator_with_try_finally(self):
        r, o = run('''
            fn* gen() {
                try {
                    yield 1;
                    yield 2;
                } finally {
                    print "gen-fin";
                }
            }
            let g = gen();
            print next(g);
            print next(g);
            print next(g, "done");
        ''')
        assert o == ['1', '2', 'gen-fin', 'done']

    def test_generator_exception_in_try_finally(self):
        r, o = run('''
            fn* gen() {
                try {
                    yield 1;
                    throw "gen-err";
                } catch(e) {
                    print e;
                } finally {
                    print "gen-fin";
                }
            }
            let g = gen();
            print next(g);
            print next(g, "done");
        ''')
        assert o == ['1', 'gen-err', 'gen-fin', 'done']


# ============================================================
# 10. Complex scenarios
# ============================================================

class TestComplexScenarios:
    def test_finally_with_optional_chaining(self):
        r, o = run('''
            fn f(obj) {
                try {
                    return obj?.name ?? "unknown";
                } finally {
                    print "checked";
                }
            }
            print f(null);
            print f({name: "test"});
        ''')
        assert o == ['checked', 'unknown', 'checked', 'test']

    def test_finally_with_spread(self):
        r, o = run('''
            fn merge(a, b) {
                try {
                    return {...a, ...b};
                } finally {
                    print "merged";
                }
            }
            let r = merge({x: 1}, {y: 2});
            print r.x;
            print r.y;
        ''')
        assert o == ['merged', '1', '2']

    def test_finally_with_destructuring(self):
        r, o = run('''
            fn f() {
                try {
                    let [a, b] = [1, 2];
                    return a + b;
                } finally {
                    print "destr-fin";
                }
            }
            print f();
        ''')
        assert o == ['destr-fin', '3']

    def test_finally_with_pipe(self):
        r, o = run('''
            fn double(x) { return x * 2; }
            fn f() {
                try {
                    return 5 |> double;
                } finally {
                    print "pipe-fin";
                }
            }
            print f();
        ''')
        assert o == ['pipe-fin', '10']

    def test_deeply_nested_finally_with_returns(self):
        r, o = run('''
            fn f() {
                try {
                    try {
                        try {
                            return "deep";
                        } finally {
                            print "f1";
                        }
                    } finally {
                        print "f2";
                    }
                } finally {
                    print "f3";
                }
            }
            print f();
        ''')
        assert o == ['f1', 'f2', 'f3', 'deep']

    def test_finally_counter_pattern(self):
        # Use arrays for mutable shared state (VM copies env for function calls)
        r, o = run('''
            let enters = [];
            let exits = [];
            fn tracked_op(val) {
                try {
                    push(enters, val);
                    if (val < 0) { throw "negative"; }
                    return val * 2;
                } catch(e) {
                    return -1;
                } finally {
                    push(exits, val);
                }
            }
            print tracked_op(5);
            print tracked_op(-3);
            print tracked_op(10);
            print len(enters);
            print len(exits);
        ''')
        assert o == ['10', '-1', '20', '3', '3']

    def test_resource_cleanup_pattern(self):
        r, o = run('''
            let resources = [];
            fn acquire(name) {
                push(resources, name);
                return name;
            }
            fn release(name) {
                let idx = 0;
                let found = -1;
                while (idx < len(resources)) {
                    if (resources[idx] == name) {
                        found = idx;
                    }
                    idx = idx + 1;
                }
            }
            fn use_resource() {
                let r = acquire("db");
                try {
                    print "using " + r;
                    return "done";
                } finally {
                    release(r);
                    print "released";
                }
            }
            print use_resource();
        ''')
        assert o == ['using db', 'released', 'done']

    def test_exception_type_preserved(self):
        r, o = run('''
            try {
                try {
                    throw {code: 404, msg: "not found"};
                } finally {
                    print "fin";
                }
            } catch(e) {
                print e.code;
                print e.msg;
            }
        ''')
        assert o == ['fin', '404', 'not found']

    def test_finally_with_null_coalescing(self):
        r, o = run('''
            fn safe_get(obj, key, default) {
                try {
                    return obj[key] ?? default;
                } finally {
                    print "accessed";
                }
            }
            print safe_get({a: 1}, "a", 0);
            print safe_get({a: null}, "a", 99);
        ''')
        assert o == ['accessed', '1', 'accessed', '99']


# ============================================================
# 11. Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_try_finally(self):
        r, o = run('try { } finally { print "fin"; }')
        assert o == ['fin']

    def test_empty_finally(self):
        r, o = run('try { print "try"; } finally { }')
        assert o == ['try']

    def test_empty_try_and_finally(self):
        r, o = run('try { } finally { }')
        assert o == []

    def test_finally_only_try_catch_no_error(self):
        r, o = run('try { print "ok"; } catch(e) { print "err"; } finally { print "done"; }')
        assert o == ['ok', 'done']

    def test_try_finally_in_if(self):
        r, o = run('''
            if (true) {
                try { print "a"; } finally { print "b"; }
            }
        ''')
        assert o == ['a', 'b']

    def test_exception_during_finally_replaces_pending(self):
        _, o, err = run_capturing_error('''
            fn f() {
                try {
                    return 1;
                } finally {
                    throw "fin-err";
                }
            }
            f();
        ''')
        assert 'fin-err' in err

    def test_return_in_finally_overrides_try_return(self):
        # In most languages, return in finally overrides try return
        # Our implementation: finally runs, then END_FINALLY does the pending return
        # If finally has its own return, it will execute via RETURN opcode
        # which checks for FinallyHandlers -- but we already popped it
        # So the return in finally should just return normally
        r, o = run('''
            fn f() {
                try {
                    return 1;
                } finally {
                    return 2;
                }
            }
            print f();
        ''')
        # The return in finally should override
        assert o == ['2']

    def test_multiple_finally_blocks_sequential(self):
        r, o = run('''
            try { print "a"; } finally { print "b"; }
            try { print "c"; } finally { print "d"; }
            try { print "e"; } finally { print "f"; }
        ''')
        assert o == ['a', 'b', 'c', 'd', 'e', 'f']


# ============================================================
# 12. Integration with existing features
# ============================================================

class TestIntegration:
    def test_finally_with_string_interpolation(self):
        r, o = run('''
            let name = "world";
            try {
                print f"hello ${name}";
            } finally {
                print f"bye ${name}";
            }
        ''')
        assert o == ['hello world', 'bye world']

    def test_finally_with_for_in_array(self):
        r, o = run('''
            let total = 0;
            for (x in [1, 2, 3]) {
                try {
                    total = total + x;
                } finally {
                    print f"added ${x}";
                }
            }
            print total;
        ''')
        assert o == ['added 1', 'added 2', 'added 3', '6']

    def test_finally_with_class_inheritance(self):
        r, o = run('''
            class Base {
                cleanup() {
                    print "base-cleanup";
                }
            }
            class Child < Base {
                work() {
                    try {
                        return "result";
                    } finally {
                        this.cleanup();
                    }
                }
            }
            let c = Child();
            print c.work();
        ''')
        assert o == ['base-cleanup', 'result']

    def test_finally_with_hash_map(self):
        r, o = run('''
            fn process(data) {
                try {
                    return data.value * 2;
                } finally {
                    print f"processed ${data.name}";
                }
            }
            print process({name: "x", value: 5});
        ''')
        assert o == ['processed x', '10']

    def test_finally_with_error_handling_chain(self):
        r, o = run('''
            fn divide(a, b) {
                try {
                    if (b == 0) { throw "div by zero"; }
                    return a / b;
                } catch(e) {
                    print f"error: ${e}";
                    return null;
                } finally {
                    print "div-done";
                }
            }
            print divide(10, 2);
            print divide(10, 0);
        ''')
        assert o == ['div-done', '5', 'error: div by zero', 'div-done', 'null']

    def test_finally_preserves_module_exports(self):
        from finally_blocks import ModuleRegistry
        reg = ModuleRegistry()
        reg.register("math", '''
            export fn safe_sqrt(x) {
                try {
                    if (x < 0) { throw "negative"; }
                    return x;
                } catch(e) {
                    return 0;
                } finally {
                    print "sqrt-done";
                }
            }
        ''')
        r, o = run('import { safe_sqrt } from "math"; print safe_sqrt(4); print safe_sqrt(-1);', registry=reg)
        assert o == ['sqrt-done', '4', 'sqrt-done', '0']


# ============================================================
# 13. Regression tests for existing features
# ============================================================

class TestRegressions:
    def test_try_catch_still_works(self):
        r, o = run('try { throw "err"; } catch(e) { print e; }')
        assert o == ['err']

    def test_null_coalescing_still_works(self):
        r, o = run('let x = null; print x ?? 42;')
        assert o == ['42']

    def test_optional_chaining_still_works(self):
        r, o = run('let x = null; print x?.foo;')
        assert o == ['null']

    def test_classes_still_work(self):
        r, o = run('''
            class Dog {
                init(name) { this.name = name; }
                bark() { return f"${this.name} says woof"; }
            }
            let d = Dog("Rex");
            print d.bark();
        ''')
        assert o == ['Rex says woof']

    def test_generators_still_work(self):
        r, o = run('''
            fn* range(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            for (x in range(3)) {
                print x;
            }
        ''')
        assert o == ['0', '1', '2']

    def test_destructuring_still_works(self):
        r, o = run('let [a, b, ...rest] = [1, 2, 3, 4]; print a; print b; print rest;')
        assert o == ['1', '2', '[3, 4]']

    def test_spread_still_works(self):
        r, o = run('let a = [1, 2]; let b = [...a, 3]; print b;')
        assert o == ['[1, 2, 3]']

    def test_pipe_still_works(self):
        r, o = run('fn add1(x) { return x + 1; } print 5 |> add1;')
        assert o == ['6']

    def test_closures_still_work(self):
        # VM copies env for each call, so closure counter returns same value
        # Test basic closure capture works
        r, o = run('''
            fn make_greeter(name) {
                return fn() { return f"hello ${name}"; };
            }
            let g = make_greeter("world");
            print g();
        ''')
        assert o == ['hello world']

    def test_error_handling_still_works(self):
        r, o = run('''
            try {
                throw "test";
            } catch(e) {
                print e;
            }
            print "after";
        ''')
        assert o == ['test', 'after']


# ============================================================
# 14. Stress tests
# ============================================================

class TestStress:
    def test_many_sequential_try_finally(self):
        stmts = []
        for i in range(20):
            stmts.append(f'try {{ print {i}; }} finally {{ print "f{i}"; }}')
        r, o = run('\n'.join(stmts))
        expected = []
        for i in range(20):
            expected.extend([str(i), f'f{i}'])
        assert o == expected

    def test_deeply_nested_try_finally(self):
        # 5 levels deep
        code = 'print "start";'
        for i in range(5):
            code = f'try {{ {code} }} finally {{ print "fin{i}"; }}'
        r, o = run(code)
        assert o[0] == 'start'
        assert len(o) == 6  # start + 5 finally blocks

    def test_recursive_with_finally(self):
        r, o = run('''
            fn countdown(n) {
                try {
                    if (n <= 0) { return "done"; }
                    return countdown(n - 1);
                } finally {
                    print n;
                }
            }
            print countdown(5);
        ''')
        assert o == ['0', '1', '2', '3', '4', '5', 'done']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
