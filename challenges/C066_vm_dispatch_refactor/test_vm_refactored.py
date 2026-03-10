"""Tests for C066: VM Dispatch Table Refactor.

Verifies that the _execute_op if/elif chain has been refactored into a dispatch
table with individual handler methods, while maintaining 100% backward compatibility.

Test categories:
1. Structural tests -- dispatch table exists, handlers are methods, coverage is complete
2. Behavioral regression tests -- all language features still work
3. Performance sanity -- dispatch overhead is negligible
"""
import pytest
import time
import inspect
from vm_refactored import (
    run, execute, ParseError, CompileError, VMError,
    VM, Op, Compiler, Chunk, CapabilityError, ModuleRegistry,
)


# ============================================================
# 1. STRUCTURAL TESTS -- Dispatch table architecture
# ============================================================

class TestDispatchTableStructure:
    """Verify the dispatch table exists and is correctly wired."""

    def test_dispatch_table_exists(self):
        """VM has a _dispatch dict attribute."""
        chunk = Chunk()
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        assert hasattr(vm, '_dispatch')
        assert isinstance(vm._dispatch, dict)

    def test_dispatch_table_has_all_opcodes(self):
        """Every Op enum value has a handler in the dispatch table."""
        chunk = Chunk()
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        for op in Op:
            assert op in vm._dispatch, f"Op.{op.name} missing from dispatch table"

    def test_dispatch_handlers_are_methods(self):
        """Each dispatch entry is a bound method."""
        chunk = Chunk()
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        for op, handler in vm._dispatch.items():
            assert callable(handler), f"Handler for Op.{op.name} is not callable"

    def test_handler_naming_convention(self):
        """Each handler follows _op_<name> naming convention."""
        chunk = Chunk()
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        for op, handler in vm._dispatch.items():
            expected = f'_op_{op.name.lower()}'
            assert handler.__name__ == expected, \
                f"Op.{op.name}: expected {expected}, got {handler.__name__}"

    def test_execute_op_is_short(self):
        """_execute_op should be a short dispatch function (< 10 lines)."""
        source = inspect.getsource(VM._execute_op)
        line_count = len(source.strip().split('\n'))
        assert line_count <= 10, f"_execute_op is {line_count} lines, expected <= 10"

    def test_no_if_elif_chain_in_execute_op(self):
        """_execute_op should not contain elif (no more if/elif chain)."""
        source = inspect.getsource(VM._execute_op)
        assert 'elif' not in source, "_execute_op still contains elif chain"

    def test_dispatch_table_count(self):
        """Dispatch table should have exactly as many entries as Op enum values."""
        chunk = Chunk()
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        assert len(vm._dispatch) == len(Op), \
            f"Dispatch has {len(vm._dispatch)} entries but Op has {len(Op)} values"

    def test_unknown_opcode_raises(self):
        """Dispatch with invalid opcode raises VMError."""
        chunk = Chunk()
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        with pytest.raises(VMError, match="Unknown opcode"):
            vm._execute_op(999)


# ============================================================
# 2. BEHAVIORAL REGRESSION -- Core language features
# ============================================================

class TestArithmetic:
    def test_add(self):
        result, _ = run("1 + 2;")
        assert result == 3

    def test_sub(self):
        result, _ = run("10 - 3;")
        assert result == 7

    def test_mul(self):
        result, _ = run("4 * 5;")
        assert result == 20

    def test_div(self):
        result, _ = run("10 / 3;")
        assert result == 3

    def test_float_div(self):
        result, _ = run("10.0 / 3;")
        assert abs(result - 3.333333) < 0.001

    def test_mod(self):
        result, _ = run("10 % 3;")
        assert result == 1

    def test_neg(self):
        result, _ = run("-5;")
        assert result == -5

    def test_div_by_zero_error(self):
        with pytest.raises(VMError, match="Division by zero"):
            run("1 / 0;")


class TestComparison:
    def test_eq(self):
        result, _ = run("1 == 1;")
        assert result is True

    def test_ne(self):
        result, _ = run("1 != 2;")
        assert result is True

    def test_lt(self):
        result, _ = run("1 < 2;")
        assert result is True

    def test_gt(self):
        result, _ = run("2 > 1;")
        assert result is True

    def test_le(self):
        result, _ = run("2 <= 2;")
        assert result is True

    def test_ge(self):
        result, _ = run("3 >= 2;")
        assert result is True


class TestLogic:
    def test_not(self):
        result, _ = run("!true;")
        assert result is False

    def test_and(self):
        result, _ = run("true && false;")
        assert result is False

    def test_or(self):
        result, _ = run("false || true;")
        assert result is True

    def test_null_coalesce(self):
        result, _ = run("let x = null; x ?? 42;")
        assert result == 42


class TestVariables:
    def test_let_and_load(self):
        result, _ = run("let x = 10; x;")
        assert result == 10

    def test_store(self):
        result, _ = run("let x = 1; x = 2; x;")
        assert result == 2

    def test_undefined_var(self):
        with pytest.raises(VMError, match="Undefined variable"):
            run("x;")


class TestControlFlow:
    def test_if_true(self):
        _, out = run("if (true) { print 1; } else { print 2; }")
        assert out == ["1"]

    def test_if_false(self):
        _, out = run("if (false) { print 1; } else { print 2; }")
        assert out == ["2"]

    def test_while(self):
        _, out = run("let i = 0; while (i < 3) { print i; i = i + 1; }")
        assert out == ["0", "1", "2"]


class TestFunctions:
    def test_fn_call(self):
        result, _ = run("fn add(a, b) { return a + b; } add(3, 4);")
        assert result == 7

    def test_closure(self):
        result, _ = run("""
            fn make_adder(n) { return fn(x) { return n + x; }; }
            let add5 = make_adder(5);
            add5(10);
        """)
        assert result == 15

    def test_recursion(self):
        result, _ = run("""
            fn fib(n) {
                if (n <= 1) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
            fib(10);
        """)
        assert result == 55


class TestArrays:
    def test_array_literal(self):
        result, _ = run("[1, 2, 3];")
        assert result == [1, 2, 3]

    def test_index_get(self):
        result, _ = run("let a = [10, 20, 30]; a[1];")
        assert result == 20

    def test_index_set(self):
        result, _ = run("let a = [1, 2, 3]; a[0] = 99; a[0];")
        assert result == 99

    def test_array_length(self):
        result, _ = run("[1, 2, 3].length;")
        assert result == 3

    def test_array_push(self):
        _, out = run("let a = [1]; a.push(2); print a;")
        assert out == ["[1, 2]"]


class TestHashes:
    def test_hash_literal(self):
        result, _ = run('{"a": 1, "b": 2};')
        assert result == {"a": 1, "b": 2}

    def test_hash_get(self):
        result, _ = run('let h = {"x": 42}; h.x;')
        assert result == 42

    def test_hash_set(self):
        result, _ = run('let h = {"x": 1}; h.x = 99; h.x;')
        assert result == 99


class TestForIn:
    def test_for_in_array(self):
        _, out = run("for (x in [1, 2, 3]) { print x; }")
        assert out == ["1", "2", "3"]

    def test_for_in_string(self):
        _, out = run('for (c in "abc") { print c; }')
        assert out == ["a", "b", "c"]


class TestErrorHandling:
    def test_try_catch(self):
        _, out = run('try { throw "boom"; } catch (e) { print e; }')
        assert out == ["boom"]

    def test_finally(self):
        _, out = run("try { print 1; } finally { print 2; }")
        assert out == ["1", "2"]


class TestClasses:
    def test_class_basic(self):
        result, _ = run("""
            class Point {
                init(x, y) { this.x = x; this.y = y; }
                sum() { return this.x + this.y; }
            }
            let p = Point(3, 4);
            p.sum();
        """)
        assert result == 7

    def test_inheritance(self):
        result, _ = run("""
            class Animal { init(name) { this.name = name; } }
            class Dog < Animal {
                init(name) { super.init(name); this.type = "dog"; }
                desc() { return this.name + " is a " + this.type; }
            }
            let d = Dog("Rex");
            d.desc();
        """)
        assert result == "Rex is a dog"


class TestGenerators:
    def test_generator(self):
        _, out = run("""
            fn* range(n) { let i = 0; while (i < n) { yield i; i = i + 1; } }
            let g = range(3);
            print next(g);
            print next(g);
            print next(g);
        """)
        assert out == ["0", "1", "2"]


class TestAsync:
    def test_async_await(self):
        _, out = run("""
            async fn greet() { return "hello"; }
            let p = greet();
            print await p;
        """)
        assert out == ["hello"]


class TestSpread:
    def test_array_spread(self):
        result, _ = run("let a = [1, 2]; let b = [...a, 3]; b;")
        assert result == [1, 2, 3]

    def test_hash_spread(self):
        result, _ = run('let a = {"x": 1}; let b = {...a, "y": 2}; b;')
        assert result == {"x": 1, "y": 2}


class TestPipe:
    def test_pipe_operator(self):
        result, _ = run("""
            fn double(x) { return x * 2; }
            fn add1(x) { return x + 1; }
            5 |> double |> add1;
        """)
        assert result == 11


class TestDestructuring:
    def test_array_destructuring(self):
        _, out = run("let [a, b] = [1, 2]; print a; print b;")
        assert out == ["1", "2"]

    def test_hash_destructuring(self):
        _, out = run('let {x, y} = {"x": 10, "y": 20}; print x; print y;')
        assert out == ["10", "20"]


class TestStringInterp:
    def test_fstring(self):
        result, _ = run('let name = "world"; f"hello ${name}";')
        assert result == "hello world"


class TestModules:
    def test_import_export(self):
        reg = ModuleRegistry()
        reg.register("math_mod", "export fn add(a, b) { return a + b; }")
        d = execute("""
            import "math_mod";
            add(1, 2);
        """, registry=reg)
        assert d['result'] == 3


class TestEnums:
    def test_enum(self):
        _, out = run("""
            enum Color { Red, Green, Blue }
            print Color.Red.name;
            print Color.Green.ordinal;
        """)
        assert out == ["Red", "1"]


class TestIterLength:
    def test_iter_length(self):
        result, _ = run("let a = [1, 2, 3]; len(a);")
        assert result == 3


class TestOptionalChaining:
    def test_optional_chain(self):
        result, _ = run("let x = null; x?.foo;")
        assert result is None


class TestAsyncGenerators:
    def test_async_gen(self):
        _, out = run("""
            async fn* nums() { yield 1; yield 2; yield 3; }
            let g = nums();
            print await next(g);
            print await next(g);
            print await next(g);
        """)
        assert out == ["1", "2", "3"]


class TestForAwait:
    def test_for_await_basic(self):
        _, out = run("""
            async fn* range(n) {
                let i = 0;
                while (i < n) { yield i; i = i + 1; }
            }
            for await (x in range(3)) { print x; }
        """)
        assert out == ["0", "1", "2"]


class TestMakeClass:
    def test_make_class_with_static(self):
        result, _ = run("""
            class Math {
                static max(a, b) { if (a > b) { return a; } return b; }
            }
            Math.max(3, 7);
        """)
        assert result == 7

    def test_getter(self):
        result, _ = run("""
            class Circle {
                init(r) { this.r = r; }
                get area() { return this.r * this.r * 3; }
            }
            let c = Circle(5);
            c.area;
        """)
        assert result == 75


class TestSuperInvoke:
    def test_super_invoke(self):
        result, _ = run("""
            class Base {
                value() { return 10; }
            }
            class Child < Base {
                value() { return super.value() + 5; }
            }
            let c = Child();
            c.value();
        """)
        assert result == 15


# ============================================================
# 3. PERFORMANCE SANITY
# ============================================================

class TestPerformance:
    def test_dispatch_overhead_negligible(self):
        """Dispatch table should not significantly slow execution."""
        code = """
            let sum = 0;
            let i = 0;
            while (i < 1000) {
                sum = sum + i;
                i = i + 1;
            }
            sum;
        """
        start = time.perf_counter()
        for _ in range(5):
            result, _ = run(code)
        elapsed = time.perf_counter() - start
        assert result == 499500
        # Should complete in well under 10 seconds for 5 runs
        assert elapsed < 10.0, f"5 runs took {elapsed:.2f}s -- dispatch overhead too high"

    def test_handler_isolation(self):
        """Each handler can be called independently (for future optimization)."""
        chunk = Chunk()
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        # Directly call a handler
        vm.push(5)
        vm.push(3)
        vm._op_add()
        assert vm.pop() == 8


# ============================================================
# 4. EDGE CASES
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        result, out = run("")
        assert result is None
        assert out == []

    def test_deeply_nested_calls(self):
        result, _ = run("""
            fn a(x) { return x + 1; }
            fn b(x) { return a(x) + 1; }
            fn c(x) { return b(x) + 1; }
            fn d(x) { return c(x) + 1; }
            d(0);
        """)
        assert result == 4

    def test_complex_expression(self):
        result, _ = run("(1 + 2) * (3 + 4) - 5;")
        assert result == 16

    def test_string_operations(self):
        result, _ = run('"hello" + " " + "world";')
        assert result == "hello world"

    def test_boolean_truthy(self):
        result, _ = run("!false;")
        assert result is True

    def test_null_handling(self):
        result, _ = run("null ?? 42;")
        assert result == 42

    def test_rest_params(self):
        _, out = run("""
            fn sum(...nums) {
                let total = 0;
                for (n in nums) { total = total + n; }
                return total;
            }
            print sum(1, 2, 3, 4, 5);
        """)
        assert out == ["15"]

    def test_computed_properties(self):
        result, _ = run("""
            let key = "x";
            let h = {[key]: 42};
            h.x;
        """)
        assert result == 42
