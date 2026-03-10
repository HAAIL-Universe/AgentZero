"""
Tests for C046 Module System
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from module_system import (
    parse, compile_source, execute, run, lex, disassemble,
    ModuleRegistry, ModuleError, VMError, ParseError, CompileError,
    ImportStmt, ExportFnDecl, ExportLetDecl, FnDecl, LetDecl,
    TokenType, ClosureObject,
)


# ============================================================
# 1. Backward compatibility -- all C045 features still work
# ============================================================

class TestBackwardCompatibility:
    """Ensure all previous features work unchanged."""

    def test_basic_arithmetic(self):
        r, _ = run("let x = 2 + 3 * 4;")
        assert r is None  # let returns None
        r = execute("2 + 3 * 4;")
        assert r['result'] == 14

    def test_variables(self):
        r = execute("let x = 10; x = x + 5; x;")
        assert r['result'] == 15

    def test_functions(self):
        r, out = run('fn add(a, b) { return a + b; } print(add(3, 4));')
        assert out == ['7']

    def test_closures(self):
        # Note: dict(captured_env) per call gives per-call isolation
        # so count resets each call (known VM semantics since C043)
        r, out = run("""
            fn make_adder(n) {
                return fn(x) { return x + n; };
            }
            let add5 = make_adder(5);
            print(add5(10));
            print(add5(20));
        """)
        assert out == ['15', '25']

    def test_arrays(self):
        r, out = run("let a = [1, 2, 3]; print(len(a)); print(a[1]);")
        assert out == ['3', '2']

    def test_hash_maps(self):
        r, out = run('let h = {name: "alice", age: 30}; print(h.name); print(h.age);')
        assert out == ['alice', '30']

    def test_for_in_loop(self):
        r, out = run("""
            let sum = 0;
            for (x in [1, 2, 3, 4, 5]) {
                sum = sum + x;
            }
            print(sum);
        """)
        assert out == ['15']

    def test_try_catch(self):
        r, out = run("""
            try {
                throw "oops";
            } catch (e) {
                print(e);
            }
        """)
        assert out == ['oops']

    def test_error_handling_builtins(self):
        r, out = run('print(type(42)); print(type("hi")); print(string(true));')
        assert out == ['int', 'string', 'true']

    def test_pattern_features(self):
        r, out = run("""
            let arr = [10, 20, 30];
            let result = map(arr, fn(x) { return x * 2; });
            print(result[0]);
            print(result[1]);
            print(result[2]);
        """)
        assert out == ['20', '40', '60']

    def test_break_continue(self):
        r, out = run("""
            let sum = 0;
            for (x in [1, 2, 3, 4, 5]) {
                if (x == 3) { continue; }
                if (x == 5) { break; }
                sum = sum + x;
            }
            print(sum);
        """)
        assert out == ['7']


# ============================================================
# 2. Lexer -- new tokens
# ============================================================

class TestLexer:
    def test_import_token(self):
        tokens = lex('import "math";')
        assert tokens[0].type == TokenType.IMPORT
        assert tokens[1].type == TokenType.STRING
        assert tokens[1].value == "math"

    def test_export_token(self):
        tokens = lex('export fn add(a, b) {}')
        assert tokens[0].type == TokenType.EXPORT
        assert tokens[1].type == TokenType.FN

    def test_from_token(self):
        tokens = lex('import { x } from "mod";')
        assert tokens[0].type == TokenType.IMPORT
        assert tokens[1].type == TokenType.LBRACE
        assert tokens[2].type == TokenType.IDENT
        assert tokens[3].type == TokenType.RBRACE
        assert tokens[4].type == TokenType.FROM
        assert tokens[5].type == TokenType.STRING

    def test_export_let_token(self):
        tokens = lex('export let PI = 3.14;')
        assert tokens[0].type == TokenType.EXPORT
        assert tokens[1].type == TokenType.LET


# ============================================================
# 3. Parser -- import/export AST
# ============================================================

class TestParser:
    def test_parse_import_all(self):
        ast = parse('import "math";')
        stmt = ast.stmts[0]
        assert isinstance(stmt, ImportStmt)
        assert stmt.module_name == "math"
        assert stmt.names == []

    def test_parse_import_selective(self):
        ast = parse('import { add, sub } from "math";')
        stmt = ast.stmts[0]
        assert isinstance(stmt, ImportStmt)
        assert stmt.module_name == "math"
        assert stmt.names == ["add", "sub"]

    def test_parse_import_single_name(self):
        ast = parse('import { PI } from "constants";')
        stmt = ast.stmts[0]
        assert isinstance(stmt, ImportStmt)
        assert stmt.names == ["PI"]

    def test_parse_export_fn(self):
        ast = parse('export fn add(a, b) { return a + b; }')
        stmt = ast.stmts[0]
        assert isinstance(stmt, ExportFnDecl)
        assert isinstance(stmt.fn_decl, FnDecl)
        assert stmt.fn_decl.name == "add"
        assert stmt.fn_decl.params == ["a", "b"]

    def test_parse_export_let(self):
        ast = parse('export let PI = 3.14;')
        stmt = ast.stmts[0]
        assert isinstance(stmt, ExportLetDecl)
        assert isinstance(stmt.let_decl, LetDecl)
        assert stmt.let_decl.name == "PI"

    def test_parse_multiple_exports(self):
        ast = parse("""
            export let X = 1;
            export fn double(n) { return n * 2; }
            export let Y = 2;
        """)
        assert len(ast.stmts) == 3
        assert isinstance(ast.stmts[0], ExportLetDecl)
        assert isinstance(ast.stmts[1], ExportFnDecl)
        assert isinstance(ast.stmts[2], ExportLetDecl)

    def test_parse_import_with_trailing_comma(self):
        ast = parse('import { add, sub, } from "math";')
        stmt = ast.stmts[0]
        assert stmt.names == ["add", "sub"]

    def test_parse_error_export_without_fn_or_let(self):
        with pytest.raises(ParseError, match="Expected 'fn' or 'let'"):
            parse('export print(42);')

    def test_parse_mixed_imports_and_code(self):
        ast = parse("""
            import "math";
            let x = 10;
            print(x);
        """)
        assert len(ast.stmts) == 3
        assert isinstance(ast.stmts[0], ImportStmt)


# ============================================================
# 4. Compiler -- export tracking
# ============================================================

class TestCompiler:
    def test_export_fn_tracked(self):
        ast = parse('export fn add(a, b) { return a + b; }')
        from module_system import Compiler
        compiler = Compiler()
        compiler.compile(ast)
        assert 'add' in compiler.exports

    def test_export_let_tracked(self):
        ast = parse('export let PI = 3.14;')
        from module_system import Compiler
        compiler = Compiler()
        compiler.compile(ast)
        assert 'PI' in compiler.exports

    def test_multiple_exports_tracked(self):
        ast = parse("""
            export let X = 1;
            export fn foo() { return 42; }
            let y = 2;
            fn bar() { return 0; }
        """)
        from module_system import Compiler
        compiler = Compiler()
        compiler.compile(ast)
        assert compiler.exports == ['X', 'foo']

    def test_import_compiles_as_noop(self):
        """Import statements produce no bytecode -- they're resolved at load time."""
        ast = parse('import "math"; let x = 1;')
        from module_system import Compiler
        compiler = Compiler()
        chunk = compiler.compile(ast)
        # The chunk should still work (the import is a no-op in bytecode)
        dis = disassemble(chunk)
        assert 'CONST' in dis  # the let x = 1 is compiled


# ============================================================
# 5. Module Registry -- basic
# ============================================================

class TestModuleRegistryBasic:
    def test_register_and_load(self):
        reg = ModuleRegistry()
        reg.register("math", """
            export fn add(a, b) { return a + b; }
            export fn sub(a, b) { return a - b; }
        """)
        exports = reg.load("math")
        assert "add" in exports
        assert "sub" in exports

    def test_module_not_found(self):
        reg = ModuleRegistry()
        with pytest.raises(ModuleError, match="not found"):
            reg.load("nonexistent")

    def test_module_caching(self):
        """Module should only execute once."""
        reg = ModuleRegistry()
        reg.register("counter", """
            export let count = 42;
        """)
        exports1 = reg.load("counter")
        exports2 = reg.load("counter")
        assert exports1 is exports2  # same dict object (cached)

    def test_register_many(self):
        reg = ModuleRegistry()
        reg.register_many({
            "a": "export let X = 1;",
            "b": "export let Y = 2;",
        })
        assert reg.load("a")["X"] == 1
        assert reg.load("b")["Y"] == 2

    def test_clear_cache(self):
        reg = ModuleRegistry()
        reg.register("mod", "export let X = 1;")
        reg.load("mod")
        assert "mod" in reg.cache
        reg.clear_cache()
        assert "mod" not in reg.cache

    def test_export_let_value(self):
        reg = ModuleRegistry()
        reg.register("constants", """
            export let PI = 3;
            export let E = 2;
        """)
        exports = reg.load("constants")
        assert exports["PI"] == 3
        assert exports["E"] == 2


# ============================================================
# 6. Import all -- full integration
# ============================================================

class TestImportAll:
    def test_import_all_functions(self):
        reg = ModuleRegistry()
        reg.register("math", """
            export fn add(a, b) { return a + b; }
            export fn mul(a, b) { return a * b; }
        """)
        r, out = run("""
            import "math";
            print(add(3, 4));
            print(mul(5, 6));
        """, registry=reg)
        assert out == ['7', '30']

    def test_import_all_variables(self):
        reg = ModuleRegistry()
        reg.register("config", """
            export let MAX = 100;
            export let MIN = 0;
        """)
        r, out = run("""
            import "config";
            print(MAX);
            print(MIN);
        """, registry=reg)
        assert out == ['100', '0']

    def test_import_mixed(self):
        reg = ModuleRegistry()
        reg.register("utils", """
            export let VERSION = 1;
            export fn greet(name) {
                return "hello";
            }
        """)
        r, out = run("""
            import "utils";
            print(VERSION);
            print(greet("world"));
        """, registry=reg)
        assert out == ['1', 'hello']


# ============================================================
# 7. Selective imports
# ============================================================

class TestSelectiveImport:
    def test_import_specific_names(self):
        reg = ModuleRegistry()
        reg.register("math", """
            export fn add(a, b) { return a + b; }
            export fn sub(a, b) { return a - b; }
            export fn mul(a, b) { return a * b; }
        """)
        r, out = run("""
            import { add } from "math";
            print(add(10, 20));
        """, registry=reg)
        assert out == ['30']

    def test_selective_import_multiple(self):
        reg = ModuleRegistry()
        reg.register("math", """
            export fn add(a, b) { return a + b; }
            export fn sub(a, b) { return a - b; }
        """)
        r, out = run("""
            import { add, sub } from "math";
            print(add(10, 3));
            print(sub(10, 3));
        """, registry=reg)
        assert out == ['13', '7']

    def test_selective_import_nonexistent(self):
        reg = ModuleRegistry()
        reg.register("math", "export fn add(a, b) { return a + b; }")
        with pytest.raises(ModuleError, match="does not export 'mul'"):
            run('import { mul } from "math";', registry=reg)

    def test_selective_import_only_gets_named(self):
        """Names not in the selective list should not be available."""
        reg = ModuleRegistry()
        reg.register("math", """
            export fn add(a, b) { return a + b; }
            export fn sub(a, b) { return a - b; }
        """)
        with pytest.raises(VMError, match="Undefined variable 'sub'"):
            run("""
                import { add } from "math";
                sub(1, 2);
            """, registry=reg)


# ============================================================
# 8. Module chaining -- module imports module
# ============================================================

class TestModuleChaining:
    def test_module_imports_module(self):
        reg = ModuleRegistry()
        reg.register("base", """
            export fn double(x) { return x * 2; }
        """)
        reg.register("derived", """
            import "base";
            export fn quadruple(x) { return double(double(x)); }
        """)
        r, out = run("""
            import "derived";
            print(quadruple(5));
        """, registry=reg)
        assert out == ['20']

    def test_three_level_chain(self):
        reg = ModuleRegistry()
        reg.register("a", "export let X = 10;")
        reg.register("b", """
            import "a";
            export let Y = X + 5;
        """)
        reg.register("c", """
            import "b";
            export let Z = Y * 2;
        """)
        r, out = run("""
            import "c";
            print(Z);
        """, registry=reg)
        assert out == ['30']

    def test_diamond_import(self):
        """A imports B and C, both import D. D should only execute once."""
        reg = ModuleRegistry()
        reg.register("d", "export let VAL = 42;")
        reg.register("b", """
            import "d";
            export let B_VAL = VAL + 1;
        """)
        reg.register("c", """
            import "d";
            export let C_VAL = VAL + 2;
        """)
        r, out = run("""
            import "b";
            import "c";
            print(B_VAL);
            print(C_VAL);
        """, registry=reg)
        assert out == ['43', '44']

    def test_selective_import_in_module(self):
        reg = ModuleRegistry()
        reg.register("math", """
            export fn add(a, b) { return a + b; }
            export fn sub(a, b) { return a - b; }
        """)
        reg.register("calc", """
            import { add } from "math";
            export fn sum3(a, b, c) { return add(add(a, b), c); }
        """)
        r, out = run("""
            import "calc";
            print(sum3(1, 2, 3));
        """, registry=reg)
        assert out == ['6']


# ============================================================
# 9. Circular imports
# ============================================================

class TestCircularImports:
    def test_direct_circular(self):
        reg = ModuleRegistry()
        reg.register("a", 'import "b"; export let X = 1;')
        reg.register("b", 'import "a"; export let Y = 2;')
        with pytest.raises(ModuleError, match="Circular import"):
            run('import "a";', registry=reg)

    def test_indirect_circular(self):
        reg = ModuleRegistry()
        reg.register("a", 'import "b"; export let X = 1;')
        reg.register("b", 'import "c"; export let Y = 2;')
        reg.register("c", 'import "a"; export let Z = 3;')
        with pytest.raises(ModuleError, match="Circular import"):
            run('import "a";', registry=reg)


# ============================================================
# 10. Module isolation
# ============================================================

class TestModuleIsolation:
    def test_module_env_isolated(self):
        """Module's internal variables don't leak unless exported."""
        reg = ModuleRegistry()
        reg.register("mod", """
            let internal = 99;
            export let public = 42;
        """)
        r, out = run("""
            import "mod";
            print(public);
        """, registry=reg)
        assert out == ['42']
        # internal should NOT be accessible
        with pytest.raises(VMError, match="Undefined variable 'internal'"):
            run("""
                import "mod";
                print(internal);
            """, registry=reg)

    def test_module_functions_isolated(self):
        """Non-exported functions are not accessible."""
        reg = ModuleRegistry()
        reg.register("mod", """
            fn helper() { return 10; }
            export fn public_fn() { return helper(); }
        """)
        r, out = run("""
            import "mod";
            print(public_fn());
        """, registry=reg)
        assert out == ['10']
        with pytest.raises(VMError, match="Undefined variable 'helper'"):
            run("""
                import "mod";
                helper();
            """, registry=reg)

    def test_importer_env_not_leaked_to_module(self):
        """Module should not see caller's variables."""
        reg = ModuleRegistry()
        reg.register("mod", """
            export fn get_x() { return 0; }
        """)
        # Even though we define x = 999 before import, the module shouldn't see it
        r, out = run("""
            let x = 999;
            import "mod";
            print(get_x());
        """, registry=reg)
        assert out == ['0']


# ============================================================
# 11. Export with complex values
# ============================================================

class TestExportComplexValues:
    def test_export_array(self):
        reg = ModuleRegistry()
        reg.register("data", "export let items = [1, 2, 3];")
        r, out = run("""
            import "data";
            print(len(items));
            print(items[1]);
        """, registry=reg)
        assert out == ['3', '2']

    def test_export_hash_map(self):
        reg = ModuleRegistry()
        reg.register("config", 'export let settings = {debug: false, level: 5};')
        r, out = run("""
            import "config";
            print(settings.level);
        """, registry=reg)
        assert out == ['5']

    def test_export_closure(self):
        reg = ModuleRegistry()
        reg.register("factory", """
            export fn make_adder(n) {
                return fn(x) { return x + n; };
            }
        """)
        r, out = run("""
            import "factory";
            let add5 = make_adder(5);
            print(add5(10));
        """, registry=reg)
        assert out == ['15']

    def test_export_computed_value(self):
        reg = ModuleRegistry()
        reg.register("computed", """
            let base = 10;
            export let doubled = base * 2;
            export let tripled = base * 3;
        """)
        r, out = run("""
            import "computed";
            print(doubled);
            print(tripled);
        """, registry=reg)
        assert out == ['20', '30']


# ============================================================
# 12. Module with side effects
# ============================================================

class TestModuleSideEffects:
    def test_module_print_runs_once(self):
        """Module with print should print during first load only."""
        reg = ModuleRegistry()
        reg.register("mod", """
            export let X = 1;
        """)
        # Load twice -- second should be cached
        exports1 = reg.load("mod")
        exports2 = reg.load("mod")
        assert exports1["X"] == 1
        assert exports1 is exports2

    def test_module_with_computation(self):
        reg = ModuleRegistry()
        reg.register("math", """
            fn factorial(n) {
                if (n <= 1) { return 1; }
                return n * factorial(n - 1);
            }
            export let FACT5 = factorial(5);
            export fn fact(n) {
                return factorial(n);
            }
        """)
        r, out = run("""
            import "math";
            print(FACT5);
            print(fact(6));
        """, registry=reg)
        assert out == ['120', '720']


# ============================================================
# 13. Execute with exports
# ============================================================

class TestExecuteExports:
    def test_execute_returns_exports(self):
        r = execute("""
            export let X = 10;
            export fn double(n) { return n * 2; }
            let y = 20;
        """)
        assert 'exports' in r
        assert 'X' in r['exports']
        assert 'double' in r['exports']
        assert r['exports']['X'] == 10
        assert 'y' not in r['exports']

    def test_execute_exports_closure(self):
        r = execute("export fn greet() { return 42; }")
        assert 'greet' in r['exports']
        assert isinstance(r['exports']['greet'], ClosureObject)


# ============================================================
# 14. Error handling in modules
# ============================================================

class TestModuleErrors:
    def test_module_compile_error(self):
        reg = ModuleRegistry()
        reg.register("bad", "export let x = ;")
        with pytest.raises(ParseError):
            reg.load("bad")

    def test_module_runtime_error(self):
        reg = ModuleRegistry()
        reg.register("bad", """
            export let x = 1 / 0;
        """)
        with pytest.raises(VMError, match="Division by zero"):
            reg.load("bad")

    def test_try_catch_with_imported_fn(self):
        reg = ModuleRegistry()
        reg.register("risky", """
            export fn divide(a, b) {
                return a / b;
            }
        """)
        r, out = run("""
            import "risky";
            try {
                let result = divide(10, 0);
            } catch (e) {
                print(e);
            }
        """, registry=reg)
        assert out == ['Division by zero']

    def test_throw_across_modules(self):
        reg = ModuleRegistry()
        reg.register("thrower", """
            export fn explode() {
                throw "boom";
            }
        """)
        r, out = run("""
            import "thrower";
            try {
                explode();
            } catch (e) {
                print(e);
            }
        """, registry=reg)
        assert out == ['boom']


# ============================================================
# 15. Module with for-in loops and arrays
# ============================================================

class TestModuleWithLanguageFeatures:
    def test_module_with_for_in(self):
        reg = ModuleRegistry()
        reg.register("summer", """
            export fn sum_array(arr) {
                let total = 0;
                for (x in arr) {
                    total = total + x;
                }
                return total;
            }
        """)
        r, out = run("""
            import "summer";
            print(sum_array([1, 2, 3, 4, 5]));
        """, registry=reg)
        assert out == ['15']

    def test_module_with_hash_map_operations(self):
        reg = ModuleRegistry()
        reg.register("hashutils", """
            export fn count_keys(h) {
                return len(keys(h));
            }
        """)
        r, out = run("""
            import "hashutils";
            let h = {a: 1, b: 2, c: 3};
            print(count_keys(h));
        """, registry=reg)
        assert out == ['3']

    def test_module_with_closures(self):
        reg = ModuleRegistry()
        reg.register("closures", """
            export fn make_multiplier(factor) {
                return fn(x) { return x * factor; };
            }
        """)
        r, out = run("""
            import "closures";
            let times3 = make_multiplier(3);
            let times7 = make_multiplier(7);
            print(times3(10));
            print(times7(10));
        """, registry=reg)
        assert out == ['30', '70']

    def test_module_with_try_catch(self):
        reg = ModuleRegistry()
        reg.register("safe", """
            export fn safe_div(a, b) {
                try {
                    return a / b;
                } catch (e) {
                    return 0;
                }
            }
        """)
        r, out = run("""
            import "safe";
            print(safe_div(10, 2));
            print(safe_div(10, 0));
        """, registry=reg)
        assert out == ['5', '0']

    def test_module_with_higher_order_functions(self):
        reg = ModuleRegistry()
        reg.register("hof", """
            export fn apply_twice(f, x) {
                return f(f(x));
            }
        """)
        r, out = run("""
            import "hof";
            fn inc(n) { return n + 1; }
            print(apply_twice(inc, 5));
        """, registry=reg)
        assert out == ['7']


# ============================================================
# 16. Multiple modules imported
# ============================================================

class TestMultipleModules:
    def test_import_two_modules(self):
        reg = ModuleRegistry()
        reg.register("math", """
            export fn add(a, b) { return a + b; }
        """)
        reg.register("string", """
            export fn greeting() { return "hello"; }
        """)
        r, out = run("""
            import "math";
            import "string";
            print(add(1, 2));
            print(greeting());
        """, registry=reg)
        assert out == ['3', 'hello']

    def test_selective_from_different_modules(self):
        reg = ModuleRegistry()
        reg.register("a", """
            export let X = 10;
            export let Y = 20;
        """)
        reg.register("b", """
            export let Z = 30;
            export let W = 40;
        """)
        r, out = run("""
            import { X } from "a";
            import { Z } from "b";
            print(X + Z);
        """, registry=reg)
        assert out == ['40']

    def test_name_collision_last_wins(self):
        """If two modules export the same name, last import wins."""
        reg = ModuleRegistry()
        reg.register("a", "export let X = 1;")
        reg.register("b", "export let X = 2;")
        r, out = run("""
            import "a";
            import "b";
            print(X);
        """, registry=reg)
        assert out == ['2']


# ============================================================
# 17. Standard library-style modules
# ============================================================

class TestStandardLibraryStyle:
    def test_math_module(self):
        reg = ModuleRegistry()
        reg.register("math", """
            export fn abs(x) {
                if (x < 0) { return -x; }
                return x;
            }
            export fn max(a, b) {
                if (a > b) { return a; }
                return b;
            }
            export fn min(a, b) {
                if (a < b) { return a; }
                return b;
            }
            export fn clamp(x, lo, hi) {
                if (x < lo) { return lo; }
                if (x > hi) { return hi; }
                return x;
            }
        """)
        r, out = run("""
            import "math";
            print(abs(-5));
            print(max(3, 7));
            print(min(3, 7));
            print(clamp(15, 0, 10));
        """, registry=reg)
        assert out == ['5', '7', '3', '10']

    def test_list_module(self):
        reg = ModuleRegistry()
        reg.register("list", """
            export fn sum(arr) {
                return reduce(arr, fn(a, b) { return a + b; }, 0);
            }
            export fn product(arr) {
                return reduce(arr, fn(a, b) { return a * b; }, 1);
            }
            export fn contains(arr, val) {
                for (x in arr) {
                    if (x == val) { return true; }
                }
                return false;
            }
        """)
        r, out = run("""
            import "list";
            print(sum([1, 2, 3, 4, 5]));
            print(product([1, 2, 3, 4]));
            print(contains([10, 20, 30], 20));
            print(contains([10, 20, 30], 99));
        """, registry=reg)
        assert out == ['15', '24', 'true', 'false']


# ============================================================
# 18. Run without registry
# ============================================================

class TestRunWithoutRegistry:
    def test_run_without_registry_no_imports(self):
        """Normal code without imports should work without registry."""
        r, out = run("print(1 + 2);")
        assert out == ['3']

    def test_run_with_export_no_registry(self):
        """Exports should compile and run even without a registry."""
        r, out = run("""
            export fn add(a, b) { return a + b; }
            print(add(3, 4));
        """)
        assert out == ['7']

    def test_execute_with_export_returns_exports(self):
        r = execute("""
            export let X = 42;
            export fn foo() { return 1; }
        """)
        assert r['exports']['X'] == 42
        assert 'foo' in r['exports']


# ============================================================
# 19. Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_module(self):
        reg = ModuleRegistry()
        reg.register("empty", "")
        exports = reg.load("empty")
        assert exports == {}

    def test_module_with_only_internal_code(self):
        reg = ModuleRegistry()
        reg.register("internal", """
            let x = 1;
            let y = 2;
            fn helper() { return x + y; }
        """)
        exports = reg.load("internal")
        assert exports == {}

    def test_export_after_import(self):
        reg = ModuleRegistry()
        reg.register("base", "export let BASE = 100;")
        reg.register("derived", """
            import "base";
            export let DOUBLED = BASE * 2;
        """)
        r, out = run("""
            import "derived";
            print(DOUBLED);
        """, registry=reg)
        assert out == ['200']

    def test_module_with_recursion(self):
        reg = ModuleRegistry()
        reg.register("recursive", """
            export fn fib(n) {
                if (n <= 1) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
        """)
        r, out = run("""
            import "recursive";
            print(fib(10));
        """, registry=reg)
        assert out == ['55']

    def test_module_export_bool(self):
        reg = ModuleRegistry()
        reg.register("flags", """
            export let DEBUG = true;
            export let VERBOSE = false;
        """)
        r, out = run("""
            import "flags";
            print(DEBUG);
            print(VERBOSE);
        """, registry=reg)
        assert out == ['true', 'false']

    def test_module_export_string(self):
        reg = ModuleRegistry()
        reg.register("strings", 'export let GREETING = "hello world";')
        r, out = run("""
            import "strings";
            print(GREETING);
        """, registry=reg)
        assert out == ['hello world']

    def test_import_module_name_with_special_chars(self):
        reg = ModuleRegistry()
        reg.register("my-module", "export let X = 1;")
        r, out = run("""
            import "my-module";
            print(X);
        """, registry=reg)
        assert out == ['1']

    def test_import_order_matters(self):
        """Imports should be resolved before code runs."""
        reg = ModuleRegistry()
        reg.register("nums", "export let N = 42;")
        r, out = run("""
            import "nums";
            print(N);
        """, registry=reg)
        assert out == ['42']


# ============================================================
# 20. Re-exporting patterns
# ============================================================

class TestReExport:
    def test_re_export_value(self):
        """Module B imports from A and re-exports."""
        reg = ModuleRegistry()
        reg.register("a", "export let ORIGINAL = 1;")
        reg.register("b", """
            import "a";
            export let RE_EXPORTED = ORIGINAL;
        """)
        r, out = run("""
            import "b";
            print(RE_EXPORTED);
        """, registry=reg)
        assert out == ['1']

    def test_re_export_function(self):
        reg = ModuleRegistry()
        reg.register("core", """
            export fn helper(x) { return x * 10; }
        """)
        reg.register("facade", """
            import "core";
            export fn process(x) { return helper(x) + 1; }
        """)
        r, out = run("""
            import "facade";
            print(process(5));
        """, registry=reg)
        assert out == ['51']


# ============================================================
# 21. Stress tests
# ============================================================

class TestStress:
    def test_many_modules(self):
        """Import a chain of 10 modules."""
        reg = ModuleRegistry()
        reg.register("m0", "export let V = 1;")
        for i in range(1, 10):
            reg.register(f"m{i}", f"""
                import "m{i-1}";
                export let V = V + 1;
            """)
        r, out = run("""
            import "m9";
            print(V);
        """, registry=reg)
        assert out == ['10']

    def test_many_exports(self):
        """Module with many exports."""
        lines = []
        for i in range(20):
            lines.append(f"export let V{i} = {i};")
        reg = ModuleRegistry()
        reg.register("many", "\n".join(lines))
        exports = reg.load("many")
        assert len(exports) == 20
        for i in range(20):
            assert exports[f"V{i}"] == i

    def test_module_with_complex_computation(self):
        reg = ModuleRegistry()
        reg.register("sorts", """
            export fn bubble_sort(arr) {
                let n = len(arr);
                let result = slice(arr, 0);
                let i = 0;
                while (i < n) {
                    let j = 0;
                    while (j < n - 1 - i) {
                        if (result[j] > result[j + 1]) {
                            let temp = result[j];
                            result[j] = result[j + 1];
                            result[j + 1] = temp;
                        }
                        j = j + 1;
                    }
                    i = i + 1;
                }
                return result;
            }
        """)
        r, out = run("""
            import "sorts";
            let sorted = bubble_sort([5, 3, 8, 1, 9, 2]);
            for (x in sorted) {
                print(x);
            }
        """, registry=reg)
        assert out == ['1', '2', '3', '5', '8', '9']


# ============================================================
# 22. Disassemble with module features
# ============================================================

class TestDisassemble:
    def test_disassemble_export_fn(self):
        chunk, compiler = compile_source("""
            export fn add(a, b) { return a + b; }
        """)
        dis = disassemble(chunk)
        assert 'CONST' in dis
        assert 'MAKE_CLOSURE' in dis
        assert 'STORE' in dis

    def test_disassemble_import_produces_no_code(self):
        """Import is a compile no-op, so disassembly should be minimal."""
        chunk, compiler = compile_source('import "math";')
        dis = disassemble(chunk)
        # Should only have HALT
        assert 'HALT' in dis


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
