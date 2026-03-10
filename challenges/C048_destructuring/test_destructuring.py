"""Tests for C048: Destructuring -- pattern-based binding for arrays and hashes."""

import pytest
from destructuring import run, execute, parse, lex, LexError, ParseError, VMError


# ============================================================
# Helper
# ============================================================

def result_of(source):
    r, _ = run(source)
    return r

def output_of(source):
    _, o = run(source)
    return o

def env_of(source):
    return execute(source)['env']


# ============================================================
# 1. Array Destructuring Basics
# ============================================================

class TestArrayDestructuringBasics:
    def test_simple_two_elements(self):
        e = env_of('let [a, b] = [1, 2];')
        assert e['a'] == 1
        assert e['b'] == 2

    def test_simple_three_elements(self):
        e = env_of('let [a, b, c] = [10, 20, 30];')
        assert e['a'] == 10
        assert e['b'] == 20
        assert e['c'] == 30

    def test_single_element(self):
        e = env_of('let [x] = [42];')
        assert e['x'] == 42

    def test_from_variable(self):
        e = env_of('let arr = [1, 2, 3]; let [a, b, c] = arr;')
        assert e['a'] == 1
        assert e['b'] == 2
        assert e['c'] == 3

    def test_string_elements(self):
        e = env_of('let [a, b] = ["hello", "world"];')
        assert e['a'] == "hello"
        assert e['b'] == "world"

    def test_mixed_types(self):
        e = env_of('let [a, b, c] = [1, "two", true];')
        assert e['a'] == 1
        assert e['b'] == "two"
        assert e['c'] is True

    def test_boolean_elements(self):
        e = env_of('let [a, b] = [true, false];')
        assert e['a'] is True
        assert e['b'] is False

    def test_from_function_return(self):
        e = env_of('fn pair() { return [3, 4]; } let [a, b] = pair();')
        assert e['a'] == 3
        assert e['b'] == 4

    def test_four_elements(self):
        e = env_of('let [a, b, c, d] = [1, 2, 3, 4];')
        assert e['a'] == 1
        assert e['d'] == 4

    def test_print_destructured(self):
        assert output_of('let [a, b] = [10, 20]; print(a + b);') == ['30']


# ============================================================
# 2. Hash Destructuring Basics
# ============================================================

class TestHashDestructuringBasics:
    def test_simple_two_keys(self):
        e = env_of('let {x, y} = {x: 1, y: 2};')
        assert e['x'] == 1
        assert e['y'] == 2

    def test_simple_three_keys(self):
        e = env_of('let {a, b, c} = {a: 10, b: 20, c: 30};')
        assert e['a'] == 10
        assert e['b'] == 20
        assert e['c'] == 30

    def test_single_key(self):
        e = env_of('let {x} = {x: 42, y: 99};')
        assert e['x'] == 42

    def test_from_variable(self):
        e = env_of('let h = {x: 1, y: 2}; let {x, y} = h;')
        assert e['x'] == 1
        assert e['y'] == 2

    def test_partial_extraction(self):
        """Only extract some keys from a larger hash."""
        e = env_of('let {a} = {a: 1, b: 2, c: 3};')
        assert e['a'] == 1

    def test_string_values(self):
        e = env_of('let {name, age} = {name: "Alice", age: 30};')
        assert e['name'] == "Alice"
        assert e['age'] == 30

    def test_from_function_return(self):
        e = env_of('fn point() { return {x: 5, y: 10}; } let {x, y} = point();')
        assert e['x'] == 5
        assert e['y'] == 10

    def test_print_destructured(self):
        assert output_of('let {x, y} = {x: 3, y: 4}; print(x + y);') == ['7']


# ============================================================
# 3. Hash Destructuring with Aliases
# ============================================================

class TestHashAliases:
    def test_simple_alias(self):
        e = env_of('let {x: a, y: b} = {x: 1, y: 2};')
        assert e['a'] == 1
        assert e['b'] == 2

    def test_mixed_alias_and_direct(self):
        e = env_of('let {x, y: b} = {x: 1, y: 2};')
        assert e['x'] == 1
        assert e['b'] == 2

    def test_alias_does_not_bind_key(self):
        e = env_of('let {x: renamed} = {x: 42};')
        assert e['renamed'] == 42
        assert 'x' not in e or e.get('x') != 42

    def test_multiple_aliases(self):
        e = env_of('let {a: x, b: y, c: z} = {a: 1, b: 2, c: 3};')
        assert e['x'] == 1
        assert e['y'] == 2
        assert e['z'] == 3


# ============================================================
# 4. Default Values
# ============================================================

class TestDefaultValues:
    def test_array_default_used(self):
        e = env_of('let [a, b = 99] = [1];')
        assert e['a'] == 1
        assert e['b'] == 99

    def test_array_default_not_used(self):
        e = env_of('let [a, b = 99] = [1, 2];')
        assert e['a'] == 1
        assert e['b'] == 2

    def test_multiple_defaults(self):
        e = env_of('let [a = 10, b = 20, c = 30] = [];')
        assert e['a'] == 10
        assert e['b'] == 20
        assert e['c'] == 30

    def test_default_expression(self):
        e = env_of('let [a, b = 3 + 4] = [1];')
        assert e['a'] == 1
        assert e['b'] == 7

    def test_hash_default_used(self):
        e = env_of('let {x, z = 99} = {x: 10};')
        assert e['x'] == 10
        assert e['z'] == 99

    def test_hash_default_not_used(self):
        e = env_of('let {x = 5, y = 10} = {x: 1, y: 2};')
        assert e['x'] == 1
        assert e['y'] == 2

    def test_hash_default_expression(self):
        e = env_of('let {x, y = 2 * 5} = {x: 1};')
        assert e['x'] == 1
        assert e['y'] == 10

    def test_hash_alias_with_default(self):
        e = env_of('let {x: a = 99} = {};')
        assert e['a'] == 99

    def test_array_default_with_short_array(self):
        """Missing element triggers default."""
        e = env_of('let [a = 42, b = 99, c = 7] = [1];')
        assert e['a'] == 1
        assert e['b'] == 99
        assert e['c'] == 7

    def test_default_string(self):
        e = env_of('let [a = "default"] = [];')
        assert e['a'] == "default"


# ============================================================
# 5. Rest Elements
# ============================================================

class TestRestElements:
    def test_simple_rest(self):
        e = env_of('let [a, ...rest] = [1, 2, 3, 4];')
        assert e['a'] == 1
        assert e['rest'] == [2, 3, 4]

    def test_rest_empty(self):
        e = env_of('let [a, b, ...rest] = [1, 2];')
        assert e['a'] == 1
        assert e['b'] == 2
        assert e['rest'] == []

    def test_rest_single_element(self):
        e = env_of('let [a, ...rest] = [1, 2];')
        assert e['a'] == 1
        assert e['rest'] == [2]

    def test_rest_all_elements(self):
        e = env_of('let [...all] = [1, 2, 3];')
        assert e['all'] == [1, 2, 3]

    def test_rest_with_two_named(self):
        e = env_of('let [a, b, ...rest] = [1, 2, 3, 4, 5];')
        assert e['a'] == 1
        assert e['b'] == 2
        assert e['rest'] == [3, 4, 5]

    def test_rest_preserves_types(self):
        e = env_of('let [a, ...rest] = [1, "two", true];')
        assert e['a'] == 1
        assert e['rest'] == ["two", True]


# ============================================================
# 6. Nested Destructuring
# ============================================================

class TestNestedDestructuring:
    def test_nested_array_in_array(self):
        e = env_of('let [a, [b, c]] = [1, [2, 3]];')
        assert e['a'] == 1
        assert e['b'] == 2
        assert e['c'] == 3

    def test_deeply_nested(self):
        e = env_of('let [a, [b, [c, d]]] = [1, [2, [3, 4]]];')
        assert e['a'] == 1
        assert e['b'] == 2
        assert e['c'] == 3
        assert e['d'] == 4

    def test_hash_in_array(self):
        e = env_of('let [a, {x, y}] = [1, {x: 2, y: 3}];')
        assert e['a'] == 1
        assert e['x'] == 2
        assert e['y'] == 3

    def test_array_in_hash(self):
        e = env_of('let {coords: [x, y]} = {coords: [10, 20]};')
        assert e['x'] == 10
        assert e['y'] == 20

    def test_hash_in_hash(self):
        e = env_of('let {inner: {a, b}} = {inner: {a: 1, b: 2}};')
        assert e['a'] == 1
        assert e['b'] == 2

    def test_nested_with_defaults(self):
        e = env_of('let [a, [b, c = 99]] = [1, [2]];')
        assert e['a'] == 1
        assert e['b'] == 2
        assert e['c'] == 99

    def test_complex_nested(self):
        src = '''
        let data = {name: "Alice", scores: [90, 85, 95]};
        let {name, scores: [first, ...rest]} = data;
        '''
        e = env_of(src)
        assert e['name'] == "Alice"
        assert e['first'] == 90
        assert e['rest'] == [85, 95]


# ============================================================
# 7. Destructuring Assignment (no let)
# ============================================================

class TestDestructuringAssignment:
    def test_swap(self):
        e = env_of('let a = 1; let b = 2; [a, b] = [b, a];')
        assert e['a'] == 2
        assert e['b'] == 1

    def test_reassign_from_array(self):
        e = env_of('let x = 0; let y = 0; [x, y] = [10, 20];')
        assert e['x'] == 10
        assert e['y'] == 20

    def test_swap_three(self):
        e = env_of('let a = 1; let b = 2; let c = 3; [a, b, c] = [c, a, b];')
        assert e['a'] == 3
        assert e['b'] == 1
        assert e['c'] == 2

    def test_assign_from_function(self):
        src = '''
        let a = 0;
        let b = 0;
        fn pair() { return [5, 6]; }
        [a, b] = pair();
        '''
        e = env_of(src)
        assert e['a'] == 5
        assert e['b'] == 6

    def test_swap_preserves_value(self):
        assert output_of('let a = 10; let b = 20; [a, b] = [b, a]; print(a); print(b);') == ['20', '10']


# ============================================================
# 8. Function Parameter Destructuring
# ============================================================

class TestFunctionParamDestructuring:
    def test_array_param(self):
        assert output_of('fn sum([a, b]) { print(a + b); } sum([3, 4]);') == ['7']

    def test_hash_param(self):
        assert output_of('fn greet({name}) { print(name); } greet({name: "Alice"});') == ['Alice']

    def test_nested_param(self):
        assert output_of('fn f([a, [b, c]]) { print(a + b + c); } f([1, [2, 3]]);') == ['6']

    def test_mixed_params(self):
        src = '''
        fn f(x, [a, b], y) {
            print(x);
            print(a);
            print(b);
            print(y);
        }
        f(1, [2, 3], 4);
        '''
        assert output_of(src) == ['1', '2', '3', '4']

    def test_hash_param_with_alias(self):
        assert output_of('fn f({x: a}) { print(a); } f({x: 42});') == ['42']

    def test_return_with_destructured_param(self):
        assert result_of('fn first([a, b]) { return a; } first([10, 20]);') == 10

    def test_lambda_with_destructured_param(self):
        assert output_of('let f = fn([a, b]) { print(a + b); }; f([3, 4]);') == ['7']

    def test_multiple_destructured_params(self):
        src = '''
        fn add_points([x1, y1], [x2, y2]) {
            return [x1 + x2, y1 + y2];
        }
        let [rx, ry] = add_points([1, 2], [3, 4]);
        print(rx);
        print(ry);
        '''
        assert output_of(src) == ['4', '6']


# ============================================================
# 9. For-In Destructuring
# ============================================================

class TestForInDestructuring:
    def test_array_of_pairs(self):
        src = '''
        let pairs = [[1, 2], [3, 4], [5, 6]];
        for ([a, b] in pairs) {
            print(a + b);
        }
        '''
        assert output_of(src) == ['3', '7', '11']

    def test_array_of_triples(self):
        src = '''
        let data = [[1, 2, 3], [4, 5, 6]];
        for ([a, b, c] in data) {
            print(a + b + c);
        }
        '''
        assert output_of(src) == ['6', '15']

    def test_array_of_nested(self):
        src = '''
        let data = [[1, [2, 3]], [4, [5, 6]]];
        for ([a, [b, c]] in data) {
            print(a + b + c);
        }
        '''
        assert output_of(src) == ['6', '15']

    def test_break_in_destructured_for_in(self):
        src = '''
        let pairs = [[1, 2], [3, 4], [5, 6]];
        for ([a, b] in pairs) {
            if (a == 3) { break; }
            print(a + b);
        }
        '''
        assert output_of(src) == ['3']

    def test_continue_in_destructured_for_in(self):
        src = '''
        let pairs = [[1, 2], [3, 4], [5, 6]];
        for ([a, b] in pairs) {
            if (a == 3) { continue; }
            print(a + b);
        }
        '''
        assert output_of(src) == ['3', '11']

    def test_hash_pattern_for_in(self):
        src = '''
        let items = [{name: "a", val: 1}, {name: "b", val: 2}];
        for ({name, val} in items) {
            print(name);
            print(val);
        }
        '''
        assert output_of(src) == ['a', '1', 'b', '2']


# ============================================================
# 10. Composition with Existing Features
# ============================================================

class TestCompositionWithExisting:
    def test_destructure_with_closures(self):
        src = '''
        fn make_pair(x, y) { return [x, y]; }
        let [a, b] = make_pair(10, 20);
        print(a + b);
        '''
        assert output_of(src) == ['30']

    def test_destructure_in_while(self):
        src = '''
        let pairs = [[1, 2], [3, 4]];
        let i = 0;
        while (i < len(pairs)) {
            let [a, b] = pairs[i];
            print(a + b);
            i = i + 1;
        }
        '''
        assert output_of(src) == ['3', '7']

    def test_destructure_with_map(self):
        src = '''
        let pairs = [[1, 2], [3, 4]];
        let sums = map(pairs, fn([a, b]) { return a + b; });
        print(sums);
        '''
        assert output_of(src) == ['[3, 7]']

    def test_destructure_with_filter(self):
        src = '''
        let pairs = [[1, 2], [3, 4], [5, 6]];
        let big = filter(pairs, fn([a, b]) { return a + b > 5; });
        print(len(big));
        '''
        assert output_of(src) == ['2']

    def test_destructure_with_try_catch(self):
        src = '''
        try {
            let [a, b] = [1, 2];
            print(a + b);
        } catch (e) {
            print("error");
        }
        '''
        assert output_of(src) == ['3']

    def test_destructure_with_error_handling(self):
        src = '''
        try {
            let [a] = 42;
        } catch (e) {
            print("caught");
        }
        '''
        assert output_of(src) == ['caught']

    def test_destructure_in_if(self):
        src = '''
        let data = [1, 2];
        let [a, b] = data;
        if (a + b == 3) {
            print("yes");
        }
        '''
        assert output_of(src) == ['yes']

    def test_nested_function_with_destructure(self):
        src = '''
        fn outer() {
            let [a, b] = [1, 2];
            fn inner() {
                return a + b;
            }
            return inner();
        }
        print(outer());
        '''
        assert output_of(src) == ['3']


# ============================================================
# 11. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_array_pattern_not_useful(self):
        """Empty destructuring -- just evaluates the expression."""
        e = env_of('let [] = [1, 2, 3];')
        # No bindings created, but should not crash
        assert True

    def test_single_rest(self):
        e = env_of('let [...all] = [1, 2, 3];')
        assert e['all'] == [1, 2, 3]

    def test_rest_from_empty(self):
        e = env_of('let [...rest] = [];')
        assert e['rest'] == []

    def test_float_values(self):
        e = env_of('let [a, b] = [1.5, 2.5];')
        assert e['a'] == 1.5
        assert e['b'] == 2.5

    def test_nested_hash_alias(self):
        e = env_of('let {pos: {x: px, y: py}} = {pos: {x: 1, y: 2}};')
        assert e['px'] == 1
        assert e['py'] == 2

    def test_destructure_result_is_not_on_stack(self):
        """let destructure should not leave value on stack."""
        r, o = run('let [a, b] = [1, 2]; print(a);')
        assert o == ['1']

    def test_large_array(self):
        src = 'let arr = range(10); let [a, b, c, ...rest] = arr;'
        e = env_of(src)
        assert e['a'] == 0
        assert e['b'] == 1
        assert e['c'] == 2
        assert e['rest'] == [3, 4, 5, 6, 7, 8, 9]

    def test_destructure_zero_values(self):
        """Arrays with zero/false values."""
        e = env_of('let [a, b] = [0, false];')
        assert e['a'] == 0
        assert e['b'] is False


# ============================================================
# 12. Module System Integration
# ============================================================

class TestModuleIntegration:
    def test_export_destructured(self):
        from destructuring import ModuleRegistry
        reg = ModuleRegistry()
        reg.register('math', 'export let PI = 3; export fn add(a, b) { return a + b; }')
        r = execute('import { PI, add } from "math"; let [a, b] = [PI, add(1, 2)];', registry=reg)
        assert r['env']['a'] == 3
        assert r['env']['b'] == 3


# ============================================================
# 13. Generator Integration
# ============================================================

class TestGeneratorIntegration:
    def test_destructure_from_generator_collect(self):
        src = '''
        fn gen() {
            yield 1;
            yield 2;
            yield 3;
        }
        let g = gen();
        let a = next(g);
        let b = next(g);
        let c = next(g);
        let [x, y, z] = [a, b, c];
        print(x + y + z);
        '''
        assert output_of(src) == ['6']


# ============================================================
# 14. Real-World Patterns
# ============================================================

class TestRealWorldPatterns:
    def test_coordinate_extraction(self):
        src = '''
        fn distance({x: x1, y: y1}, {x: x2, y: y2}) {
            let dx = x2 - x1;
            let dy = y2 - y1;
            return dx * dx + dy * dy;
        }
        print(distance({x: 0, y: 0}, {x: 3, y: 4}));
        '''
        assert output_of(src) == ['25']

    def test_options_with_defaults(self):
        src = '''
        fn create_user({name, age = 0, active = true}) {
            print(name);
            print(age);
            print(active);
        }
        create_user({name: "Bob"});
        '''
        assert output_of(src) == ['Bob', '0', 'true']

    def test_head_tail_pattern(self):
        src = '''
        let [head, ...tail] = [1, 2, 3, 4, 5];
        print(head);
        print(len(tail));
        '''
        assert output_of(src) == ['1', '4']

    def test_first_rest_processing(self):
        src = '''
        fn process_first([first, ...rest]) {
            print(first * 2);
            return rest;
        }
        let remaining = process_first([10, 20, 30]);
        print(len(remaining));
        '''
        assert output_of(src) == ['20', '2']

    def test_config_extraction(self):
        src = '''
        let config = {host: "localhost", port: 8080, debug: true};
        let {host, port, debug} = config;
        print(host);
        print(port);
        print(debug);
        '''
        assert output_of(src) == ['localhost', '8080', 'true']

    def test_matrix_row_processing(self):
        src = '''
        let matrix = [[1, 2], [3, 4], [5, 6]];
        let total = 0;
        for ([a, b] in matrix) {
            total = total + a + b;
        }
        print(total);
        '''
        assert output_of(src) == ['21']

    def test_named_return_destructure(self):
        src = '''
        fn divide(a, b) {
            if (b == 0) {
                return {result: 0, error: "division by zero"};
            }
            return {result: a / b, error: ""};
        }
        let {result, error} = divide(10, 2);
        print(result);
        '''
        assert output_of(src) == ['5']

    def test_nested_data_extraction(self):
        src = '''
        let user = {
            name: "Alice",
            address: {
                city: "NYC",
                zip: "10001"
            }
        };
        let {name, address: {city}} = user;
        print(name);
        print(city);
        '''
        assert output_of(src) == ['Alice', 'NYC']

    def test_array_of_records(self):
        src = '''
        let people = [
            {name: "Alice", age: 30},
            {name: "Bob", age: 25}
        ];
        for ({name, age} in people) {
            print(name);
        }
        '''
        assert output_of(src) == ['Alice', 'Bob']

    def test_reduce_with_destructuring(self):
        src = '''
        let pairs = [[1, 2], [3, 4], [5, 6]];
        let total = reduce(pairs, fn(acc, pair) {
            let [a, b] = pair;
            return acc + a + b;
        }, 0);
        print(total);
        '''
        assert output_of(src) == ['21']


# ============================================================
# 15. Lexer Tests
# ============================================================

class TestLexer:
    def test_dotdotdot_token(self):
        tokens = lex('...rest')
        assert tokens[0].value == '...'

    def test_dotdotdot_in_context(self):
        tokens = lex('[a, ...rest]')
        types = [t.value for t in tokens if t.value != '']
        assert '...' in types

    def test_dot_still_works(self):
        tokens = lex('a.b')
        assert tokens[1].value == '.'


# ============================================================
# 16. Parser Tests
# ============================================================

class TestParser:
    def test_parse_array_pattern(self):
        ast = parse('let [a, b] = [1, 2];')
        stmt = ast.stmts[0]
        from destructuring import LetDestructure, ArrayPattern
        assert isinstance(stmt, LetDestructure)
        assert isinstance(stmt.pattern, ArrayPattern)
        assert len(stmt.pattern.elements) == 2

    def test_parse_hash_pattern(self):
        ast = parse('let {x, y} = h;')
        stmt = ast.stmts[0]
        from destructuring import LetDestructure, HashPattern
        assert isinstance(stmt, LetDestructure)
        assert isinstance(stmt.pattern, HashPattern)

    def test_parse_rest_element(self):
        ast = parse('let [a, ...rest] = arr;')
        from destructuring import LetDestructure, ArrayPattern
        stmt = ast.stmts[0]
        assert isinstance(stmt, LetDestructure)
        assert stmt.pattern.elements[1].rest is True

    def test_parse_default(self):
        ast = parse('let [a = 1] = arr;')
        from destructuring import LetDestructure
        stmt = ast.stmts[0]
        assert stmt.pattern.elements[0].default is not None

    def test_parse_alias(self):
        ast = parse('let {x: a} = h;')
        from destructuring import LetDestructure, HashPattern
        stmt = ast.stmts[0]
        assert stmt.pattern.entries[0].key == 'x'
        assert stmt.pattern.entries[0].alias == 'a'


# ============================================================
# 17. Error Cases
# ============================================================

class TestErrors:
    def test_destructure_non_array(self):
        with pytest.raises(VMError):
            run('let [a, b] = 42;')

    def test_destructure_non_hash(self):
        with pytest.raises(VMError):
            run('let {x} = 42;')

    def test_missing_key_no_default(self):
        with pytest.raises(VMError):
            run('let {x} = {};')

    def test_index_out_of_bounds_no_default(self):
        with pytest.raises(VMError):
            run('let [a, b, c] = [1, 2];')


# ============================================================
# 18. Backward Compatibility
# ============================================================

class TestBackwardCompatibility:
    def test_normal_let_still_works(self):
        e = env_of('let x = 42;')
        assert e['x'] == 42

    def test_normal_assignment_still_works(self):
        e = env_of('let x = 1; x = 2;')
        assert e['x'] == 2

    def test_normal_for_in_still_works(self):
        assert output_of('for (x in [1, 2, 3]) { print(x); }') == ['1', '2', '3']

    def test_normal_for_in_kv_still_works(self):
        assert output_of('for (k, v in {a: 1}) { print(k); print(v); }') == ['a', '1']

    def test_normal_fn_params_still_work(self):
        assert output_of('fn add(a, b) { print(a + b); } add(3, 4);') == ['7']

    def test_closures_still_work(self):
        assert output_of('let f = fn(x) { return fn(y) { return x + y; }; }; print(f(3)(4));') == ['7']

    def test_generators_still_work(self):
        src = '''
        fn gen() { yield 1; yield 2; }
        let g = gen();
        print(next(g));
        print(next(g));
        '''
        assert output_of(src) == ['1', '2']

    def test_hash_literal_still_works(self):
        e = env_of('let h = {x: 1, y: 2};')
        assert e['h'] == {'x': 1, 'y': 2}

    def test_array_literal_still_works(self):
        e = env_of('let a = [1, 2, 3];')
        assert e['a'] == [1, 2, 3]

    def test_try_catch_still_works(self):
        assert output_of('try { throw "err"; } catch (e) { print(e); }') == ['err']

    def test_modules_still_work(self):
        from destructuring import ModuleRegistry
        reg = ModuleRegistry()
        reg.register('m', 'export let x = 42;')
        r = execute('import { x } from "m"; print(x);', registry=reg)
        assert r['output'] == ['42']

    def test_pattern_matching_not_broken(self):
        """Existing C040-C047 features should still work."""
        assert output_of('let arr = [1, 2, 3]; print(arr[0]); print(len(arr));') == ['1', '3']

    def test_for_in_with_break(self):
        src = '''
        for (x in [1, 2, 3, 4]) {
            if (x == 3) { break; }
            print(x);
        }
        '''
        assert output_of(src) == ['1', '2']


# ============================================================
# 19. Complex Patterns
# ============================================================

class TestComplexPatterns:
    def test_deeply_nested_mixed(self):
        src = '''
        let data = [1, {inner: [2, 3]}, 4];
        let [a, {inner: [b, c]}, d] = data;
        print(a + b + c + d);
        '''
        assert output_of(src) == ['10']

    def test_rest_after_nested(self):
        src = '''
        let [[a, b], ...rest] = [[1, 2], [3, 4], [5, 6]];
        print(a);
        print(b);
        print(len(rest));
        '''
        assert output_of(src) == ['1', '2', '2']

    def test_chained_destructure(self):
        src = '''
        let [a, b] = [1, 2];
        let [c, d] = [a + 1, b + 1];
        print(c);
        print(d);
        '''
        assert output_of(src) == ['2', '3']

    def test_destructure_in_loop(self):
        src = '''
        let result = 0;
        let i = 0;
        let pairs = [[1, 2], [3, 4], [5, 6]];
        while (i < len(pairs)) {
            let [a, b] = pairs[i];
            result = result + a * b;
            i = i + 1;
        }
        print(result);
        '''
        assert output_of(src) == ['44']

    def test_multiple_hash_destructure(self):
        src = '''
        let {x} = {x: 1};
        let {y} = {y: 2};
        print(x + y);
        '''
        assert output_of(src) == ['3']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
