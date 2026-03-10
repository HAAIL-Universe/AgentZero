"""Tests for C050: Spread Operator"""
import pytest
from spread_operator import run, execute, lex, parse, SpreadExpr, ArrayLit, HashLit

# ============================================================
# Section 1: Array Spread Basics
# ============================================================

class TestArraySpreadBasics:
    def test_spread_single_array(self):
        r, o = run('let a = [1, 2, 3]; let b = [...a]; print b;')
        assert o == ['[1, 2, 3]']

    def test_spread_into_new_array(self):
        r, o = run('let a = [1, 2]; let b = [0, ...a, 3]; print b;')
        assert o == ['[0, 1, 2, 3]']

    def test_spread_at_start(self):
        r, o = run('let a = [1, 2]; let b = [...a, 3, 4]; print b;')
        assert o == ['[1, 2, 3, 4]']

    def test_spread_at_end(self):
        r, o = run('let a = [3, 4]; let b = [1, 2, ...a]; print b;')
        assert o == ['[1, 2, 3, 4]']

    def test_spread_empty_array(self):
        r, o = run('let a = []; let b = [1, ...a, 2]; print b;')
        assert o == ['[1, 2]']

    def test_spread_only(self):
        r, o = run('let a = [1, 2, 3]; let b = [...a]; print b;')
        assert o == ['[1, 2, 3]']

    def test_spread_creates_copy(self):
        """Spread creates a new array, not a reference."""
        r, o = run('''
            let a = [1, 2, 3];
            let b = [...a];
            push(b, 4);
            print a;
            print b;
        ''')
        assert o == ['[1, 2, 3]', '[1, 2, 3, 4]']

    def test_spread_single_element(self):
        r, o = run('let a = [42]; let b = [...a]; print b;')
        assert o == ['[42]']


# ============================================================
# Section 2: Multiple Array Spreads
# ============================================================

class TestMultipleArraySpreads:
    def test_two_spreads(self):
        r, o = run('let a = [1, 2]; let b = [3, 4]; let c = [...a, ...b]; print c;')
        assert o == ['[1, 2, 3, 4]']

    def test_three_spreads(self):
        r, o = run('''
            let a = [1]; let b = [2]; let c = [3];
            let d = [...a, ...b, ...c];
            print d;
        ''')
        assert o == ['[1, 2, 3]']

    def test_spreads_with_elements_between(self):
        r, o = run('''
            let a = [1, 2]; let b = [5, 6];
            let c = [...a, 3, 4, ...b];
            print c;
        ''')
        assert o == ['[1, 2, 3, 4, 5, 6]']

    def test_spread_same_array_twice(self):
        r, o = run('let a = [1, 2]; let b = [...a, ...a]; print b;')
        assert o == ['[1, 2, 1, 2]']

    def test_empty_spreads(self):
        r, o = run('let a = []; let b = []; let c = [...a, ...b]; print c;')
        assert o == ['[]']


# ============================================================
# Section 3: Hash Spread Basics
# ============================================================

class TestHashSpreadBasics:
    def test_spread_single_hash(self):
        r, o = run('let h = {a: 1, b: 2}; let h2 = {...h}; print h2.a; print h2.b;')
        assert o == ['1', '2']

    def test_spread_with_extra_keys(self):
        r, o = run('let h = {a: 1}; let h2 = {...h, b: 2}; print h2.a; print h2.b;')
        assert o == ['1', '2']

    def test_spread_at_start_of_hash(self):
        r, o = run('let h = {a: 1}; let h2 = {...h, b: 2}; print h2.a; print h2.b;')
        assert o == ['1', '2']

    def test_spread_at_end_of_hash(self):
        r, o = run('let h = {b: 2}; let h2 = {a: 1, ...h}; print h2.a; print h2.b;')
        assert o == ['1', '2']

    def test_spread_empty_hash(self):
        r, o = run('let h = {}; let h2 = {a: 1, ...h}; print h2.a;')
        assert o == ['1']

    def test_spread_creates_hash_copy(self):
        """Spread creates a new hash, not a reference."""
        r, o = run('''
            let h = {a: 1, b: 2};
            let h2 = {...h};
            h2.c = 3;
            print has(h, "c");
            print has(h2, "c");
        ''')
        assert o == ['false', 'true']


# ============================================================
# Section 4: Hash Spread Override Behavior
# ============================================================

class TestHashSpreadOverrides:
    def test_later_key_overrides_spread(self):
        r, o = run('let h = {a: 1, b: 2}; let h2 = {...h, a: 10}; print h2.a; print h2.b;')
        assert o == ['10', '2']

    def test_spread_overrides_earlier_key(self):
        r, o = run('let h = {a: 10}; let h2 = {a: 1, ...h}; print h2.a;')
        assert o == ['10']

    def test_multiple_hash_spreads_override(self):
        r, o = run('''
            let defaults = {a: 1, b: 2, c: 3};
            let overrides = {b: 20, c: 30};
            let h = {...defaults, ...overrides};
            print h.a; print h.b; print h.c;
        ''')
        assert o == ['1', '20', '30']

    def test_sandwich_override(self):
        """Key, spread, key -- last writer wins at each position."""
        r, o = run('''
            let h = {x: 100};
            let h2 = {x: 1, ...h, y: 2};
            print h2.x; print h2.y;
        ''')
        assert o == ['100', '2']

    def test_triple_hash_spread(self):
        r, o = run('''
            let a = {x: 1}; let b = {y: 2}; let c = {z: 3};
            let h = {...a, ...b, ...c};
            print h.x; print h.y; print h.z;
        ''')
        assert o == ['1', '2', '3']


# ============================================================
# Section 5: Function Call Spread
# ============================================================

class TestCallSpread:
    def test_spread_args_basic(self):
        r, o = run('''
            fn add(a, b) { return a + b; }
            let args = [3, 4];
            print add(...args);
        ''')
        assert o == ['7']

    def test_spread_args_with_normal(self):
        r, o = run('''
            fn triple(a, b, c) { return a + b + c; }
            let rest = [2, 3];
            print triple(1, ...rest);
        ''')
        assert o == ['6']

    def test_spread_args_at_start(self):
        r, o = run('''
            fn triple(a, b, c) { return a + b + c; }
            let first = [1, 2];
            print triple(...first, 3);
        ''')
        assert o == ['6']

    def test_spread_multiple_arrays_in_call(self):
        r, o = run('''
            fn quad(a, b, c, d) { return a * b + c * d; }
            let x = [2, 3]; let y = [4, 5];
            print quad(...x, ...y);
        ''')
        assert o == ['26']

    def test_spread_into_builtin(self):
        r, o = run('''
            let args = [[1, 2, 3], [4, 5]];
            let result = concat(...args);
            print result;
        ''')
        assert o == ['[1, 2, 3, 4, 5]']

    def test_spread_into_closure(self):
        r, o = run('''
            fn make_adder(x) {
                return fn(a, b) { return x + a + b; };
            }
            let add10 = make_adder(10);
            let args = [3, 4];
            print add10(...args);
        ''')
        assert o == ['17']

    def test_spread_single_arg(self):
        r, o = run('''
            fn identity(x) { return x; }
            let args = [42];
            print identity(...args);
        ''')
        assert o == ['42']


# ============================================================
# Section 6: Spread with Expressions
# ============================================================

class TestSpreadExpressions:
    def test_spread_function_result(self):
        r, o = run('''
            fn get_items() { return [1, 2, 3]; }
            let b = [0, ...get_items(), 4];
            print b;
        ''')
        assert o == ['[0, 1, 2, 3, 4]']

    def test_spread_computed_array(self):
        r, o = run('''
            let b = [...range(3), ...range(3, 6)];
            print b;
        ''')
        assert o == ['[0, 1, 2, 3, 4, 5]']

    def test_spread_index_result(self):
        r, o = run('''
            let matrix = [[1, 2], [3, 4]];
            let flat = [...matrix[0], ...matrix[1]];
            print flat;
        ''')
        assert o == ['[1, 2, 3, 4]']

    def test_spread_hash_from_function(self):
        r, o = run('''
            fn defaults() { return {a: 1, b: 2}; }
            let h = {...defaults(), c: 3};
            print h.a; print h.b; print h.c;
        ''')
        assert o == ['1', '2', '3']

    def test_spread_conditional_array(self):
        r, o = run('''
            let x = true;
            let extra = if (x) [10, 20] else [];
            let arr = [1, ...extra, 2];
            print arr;
        ''')
        assert o == ['[1, 10, 20, 2]']


# ============================================================
# Section 7: Spread with Strings
# ============================================================

class TestSpreadStrings:
    def test_spread_string_into_array(self):
        r, o = run('let a = [..."abc"]; print a;')
        assert o == ['[a, b, c]']

    def test_spread_string_with_elements(self):
        r, o = run('let a = [1, ..."hi", 2]; print a;')
        assert o == ['[1, h, i, 2]']

    def test_spread_empty_string(self):
        r, o = run('let a = [1, ..."", 2]; print a;')
        assert o == ['[1, 2]']

    def test_spread_string_variable(self):
        r, o = run('let s = "xyz"; let a = [...s]; print a;')
        assert o == ['[x, y, z]']


# ============================================================
# Section 8: Nested Spread
# ============================================================

class TestNestedSpread:
    def test_spread_in_nested_array(self):
        r, o = run('''
            let inner = [3, 4];
            let outer = [[1, 2], [...inner, 5]];
            print outer;
        ''')
        assert o == ['[[1, 2], [3, 4, 5]]']

    def test_spread_in_hash_value(self):
        r, o = run('''
            let items = [1, 2, 3];
            let h = {data: [...items, 4]};
            print h.data;
        ''')
        assert o == ['[1, 2, 3, 4]']

    def test_spread_hash_in_array_of_hashes(self):
        r, o = run('''
            let base = {x: 1};
            let arr = [{...base, y: 2}, {...base, y: 3}];
            print arr[0].x; print arr[0].y;
            print arr[1].x; print arr[1].y;
        ''')
        assert o == ['1', '2', '1', '3']

    def test_deeply_nested_spread(self):
        r, o = run('''
            let a = [1]; let b = [2]; let c = [3];
            let result = [...a, [...b, ...c]];
            print result;
        ''')
        assert o == ['[1, [2, 3]]']


# ============================================================
# Section 9: Spread and Destructuring Interaction
# ============================================================

class TestSpreadDestructuring:
    def test_spread_with_destructuring_let(self):
        """Spread creates array, destructuring unpacks it."""
        r, o = run('''
            let a = [1, 2]; let b = [3, 4];
            let [x, y, z, w] = [...a, ...b];
            print x; print y; print z; print w;
        ''')
        assert o == ['1', '2', '3', '4']

    def test_spread_with_rest_destructuring(self):
        r, o = run('''
            let items = [1, 2, 3, 4, 5];
            let [first, ...rest] = [...items];
            print first; print rest;
        ''')
        assert o == ['1', '[2, 3, 4, 5]']

    def test_destructuring_assignment_with_spread_rhs(self):
        r, o = run('''
            let a = [1, 2];
            let x = 0; let y = 0; let z = 0;
            [x, y, z] = [...a, 3];
            print x; print y; print z;
        ''')
        assert o == ['1', '2', '3']


# ============================================================
# Section 10: Spread in Loops
# ============================================================

class TestSpreadInLoops:
    def test_spread_in_for_in(self):
        r, o = run('''
            let a = [1, 2]; let b = [3, 4];
            let result = 0;
            for (x in [...a, ...b]) {
                result = result + x;
            }
            print result;
        ''')
        assert o == ['10']

    def test_accumulate_with_spread(self):
        r, o = run('''
            let result = [];
            let i = 0;
            while (i < 3) {
                result = [...result, i];
                i = i + 1;
            }
            print result;
        ''')
        assert o == ['[0, 1, 2]']

    def test_hash_accumulate_with_spread(self):
        r, o = run('''
            let config = {};
            config = {...config, a: 1};
            config = {...config, b: 2};
            config = {...config, c: 3};
            print config.a; print config.b; print config.c;
        ''')
        assert o == ['1', '2', '3']


# ============================================================
# Section 11: Spread with Closures
# ============================================================

class TestSpreadClosures:
    def test_spread_captured_array(self):
        r, o = run('''
            let items = [1, 2, 3];
            fn extend() {
                return [...items, 4, 5];
            }
            print extend();
        ''')
        assert o == ['[1, 2, 3, 4, 5]']

    def test_spread_in_map_callback(self):
        r, o = run('''
            let prefix = [0];
            let arrays = [[1], [2], [3]];
            let result = map(arrays, fn(a) { return [...prefix, ...a]; });
            print result;
        ''')
        assert o == ['[[0, 1], [0, 2], [0, 3]]']

    def test_spread_in_reduce(self):
        r, o = run('''
            let arrays = [[1, 2], [3, 4], [5, 6]];
            let flat = reduce(arrays, fn(acc, a) { return [...acc, ...a]; }, []);
            print flat;
        ''')
        assert o == ['[1, 2, 3, 4, 5, 6]']


# ============================================================
# Section 12: Error Handling with Spread
# ============================================================

class TestSpreadErrors:
    def test_spread_non_array_into_array(self):
        with pytest.raises(Exception, match="Cannot spread"):
            run('let x = 42; let a = [...x];')

    def test_spread_non_hash_into_hash(self):
        with pytest.raises(Exception, match="Cannot spread"):
            run('let x = 42; let h = {...x};')

    def test_spread_array_into_hash(self):
        with pytest.raises(Exception, match="Cannot spread"):
            run('let a = [1, 2]; let h = {...a};')

    def test_spread_hash_into_array(self):
        with pytest.raises(Exception, match="Cannot spread"):
            run('let h = {a: 1}; let a = [...h];')

    def test_spread_wrong_arity(self):
        with pytest.raises(Exception, match="expects"):
            run('''
                fn add(a, b) { return a + b; }
                let args = [1, 2, 3];
                add(...args);
            ''')

    def test_spread_error_caught_by_try(self):
        r, o = run('''
            try {
                let x = 42;
                let a = [...x];
            } catch (e) {
                print "caught";
            }
        ''')
        assert o == ['caught']


# ============================================================
# Section 13: Spread with Other Features
# ============================================================

class TestSpreadIntegration:
    def test_spread_with_fstring(self):
        r, o = run('''
            let items = [1, 2, 3];
            let extended = [...items, 4];
            print f"items: ${string(extended)}";
        ''')
        assert o == ['items: [1, 2, 3, 4]']

    def test_spread_with_generators(self):
        """Spread a collected generator result."""
        r, o = run('''
            fn* gen() {
                yield 1;
                yield 2;
                yield 3;
            }
            let g = gen();
            let items = [];
            let v = next(g, "done");
            while (v != "done") {
                items = [...items, v];
                v = next(g, "done");
            }
            print items;
        ''')
        assert o == ['[1, 2, 3]']

    def test_spread_with_modules(self):
        from spread_operator import ModuleRegistry
        reg = ModuleRegistry()
        reg.register("utils", '''
            export fn get_defaults() {
                return {theme: "dark", lang: "en"};
            }
        ''')
        r, o = run('''
            import { get_defaults } from "utils";
            let config = {...get_defaults(), lang: "fr"};
            print config.theme;
            print config.lang;
        ''', registry=reg)
        assert o == ['dark', 'fr']

    def test_spread_with_for_in_destructuring(self):
        r, o = run('''
            let a = [1, 2]; let b = [3, 4];
            let pairs = [[...a], [...b]];
            for ([x, y] in pairs) {
                print x + y;
            }
        ''')
        assert o == ['3', '7']

    def test_spread_preserves_order(self):
        r, o = run('''
            let h1 = {a: 1, b: 2};
            let h2 = {c: 3, d: 4};
            let h3 = {...h1, ...h2};
            print keys(h3);
        ''')
        assert o == ['[a, b, c, d]']


# ============================================================
# Section 14: Edge Cases
# ============================================================

class TestSpreadEdgeCases:
    def test_spread_in_empty_array_context(self):
        r, o = run('let a = []; let b = [...a]; print b;')
        assert o == ['[]']

    def test_spread_in_empty_hash_context(self):
        r, o = run('let h = {}; let h2 = {...h}; print keys(h2);')
        assert o == ['[]']

    def test_spread_large_array(self):
        r, o = run('''
            let a = range(50);
            let b = range(50, 100);
            let c = [...a, ...b];
            print len(c);
        ''')
        assert o == ['100']

    def test_spread_into_builtin_concat(self):
        r, o = run('''
            let a = [1, 2]; let b = [3, 4];
            let c = concat(...[a, b]);
            print c;
        ''')
        # concat([1,2], [3,4]) = [1,2,3,4]
        # ...[[1,2],[3,4]] spreads the outer array into args
        assert o == ['[1, 2, 3, 4]']

    def test_spread_bool_values(self):
        r, o = run('let a = [true, false]; let b = [...a, true]; print b;')
        assert o == ['[true, false, true]']

    def test_spread_mixed_types(self):
        r, o = run('let a = [1, "hi", true]; let b = [...a]; print b;')
        assert o == ['[1, hi, true]']

    def test_spread_nested_arrays(self):
        """Spread is shallow -- nested arrays are not flattened."""
        r, o = run('''
            let a = [[1, 2], [3, 4]];
            let b = [...a];
            print b;
        ''')
        assert o == ['[[1, 2], [3, 4]]']

    def test_spread_hash_with_numeric_keys(self):
        r, o = run('''
            let h = {1: "one", 2: "two"};
            let h2 = {...h, 3: "three"};
            print size(h2);
        ''')
        assert o == ['3']

    def test_spread_hash_string_keys(self):
        r, o = run('''
            let h1 = {"name": "alice"};
            let h2 = {...h1, "age": 30};
            print h2.name; print h2.age;
        ''')
        assert o == ['alice', '30']

    def test_multiple_spread_operations(self):
        """Spread in multiple statements."""
        r, o = run('''
            let a = [1, 2];
            let b = [...a, 3];
            let c = [...b, 4];
            let d = [...c, 5];
            print d;
        ''')
        assert o == ['[1, 2, 3, 4, 5]']


# ============================================================
# Section 15: Parser Tests
# ============================================================

class TestSpreadParsing:
    def test_parse_array_spread(self):
        ast = parse('[...a, 1];')
        stmt = ast.stmts[0]
        assert isinstance(stmt, ArrayLit)
        assert isinstance(stmt.elements[0], SpreadExpr)

    def test_parse_hash_spread(self):
        ast = parse('{...h, a: 1};')
        stmt = ast.stmts[0]
        assert isinstance(stmt, HashLit)
        assert isinstance(stmt.pairs[0], SpreadExpr)

    def test_parse_call_spread(self):
        from spread_operator import CallExpr
        ast = parse('f(...args);')
        stmt = ast.stmts[0]
        assert isinstance(stmt, CallExpr)
        assert isinstance(stmt.args[0], SpreadExpr)

    def test_parse_multiple_spreads(self):
        ast = parse('[...a, ...b, ...c];')
        stmt = ast.stmts[0]
        assert isinstance(stmt, ArrayLit)
        assert all(isinstance(e, SpreadExpr) for e in stmt.elements)

    def test_parse_mixed_spread_elements(self):
        ast = parse('[1, ...a, 2, ...b, 3];')
        stmt = ast.stmts[0]
        assert isinstance(stmt, ArrayLit)
        assert len(stmt.elements) == 5
        assert not isinstance(stmt.elements[0], SpreadExpr)
        assert isinstance(stmt.elements[1], SpreadExpr)
        assert not isinstance(stmt.elements[2], SpreadExpr)
        assert isinstance(stmt.elements[3], SpreadExpr)
        assert not isinstance(stmt.elements[4], SpreadExpr)


# ============================================================
# Section 16: Spread with Try/Catch
# ============================================================

class TestSpreadTryCatch:
    def test_spread_error_in_try(self):
        r, o = run('''
            try {
                let h = {a: 1};
                let arr = [...h];
            } catch (e) {
                print "error caught";
            }
        ''')
        assert o == ['error caught']

    def test_spread_in_try_success(self):
        r, o = run('''
            try {
                let a = [1, 2]; let b = [3, 4];
                let c = [...a, ...b];
                print c;
            } catch (e) {
                print "error";
            }
        ''')
        assert o == ['[1, 2, 3, 4]']


# ============================================================
# Section 17: C049 Backward Compatibility
# ============================================================

class TestBackwardCompatibility:
    def test_basic_arithmetic(self):
        r, o = run('print 2 + 3;')
        assert o == ['5']

    def test_array_operations(self):
        r, o = run('let a = [1, 2, 3]; print len(a); print a[1];')
        assert o == ['3', '2']

    def test_hash_operations(self):
        r, o = run('let h = {x: 10, y: 20}; print h.x; print h.y;')
        assert o == ['10', '20']

    def test_closures(self):
        r, o = run('''
            fn make(x) { return fn(y) { return x + y; }; }
            let f = make(10);
            print f(5);
        ''')
        assert o == ['15']

    def test_destructuring(self):
        r, o = run('let [a, b, ...rest] = [1, 2, 3, 4, 5]; print a; print b; print rest;')
        assert o == ['1', '2', '[3, 4, 5]']

    def test_fstrings(self):
        r, o = run('let name = "world"; print f"hello ${name}";')
        assert o == ['hello world']

    def test_generators(self):
        r, o = run('''
            fn* nums() { yield 1; yield 2; yield 3; }
            let g = nums();
            print next(g); print next(g); print next(g);
        ''')
        assert o == ['1', '2', '3']

    def test_for_in(self):
        r, o = run('''
            let sum = 0;
            for (x in [1, 2, 3, 4]) { sum = sum + x; }
            print sum;
        ''')
        assert o == ['10']

    def test_try_catch(self):
        r, o = run('''
            try { throw "boom"; } catch (e) { print e; }
        ''')
        assert o == ['boom']

    def test_modules(self):
        from spread_operator import ModuleRegistry
        reg = ModuleRegistry()
        reg.register("math", 'export fn double(x) { return x * 2; }')
        r, o = run('import { double } from "math"; print double(21);', registry=reg)
        assert o == ['42']

    def test_if_expression(self):
        r, o = run('let x = if (true) 1 else 2; print x;')
        assert o == ['1']

    def test_print_without_parens(self):
        r, o = run('print 42;')
        assert o == ['42']
