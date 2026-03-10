"""
Tests for C044: For-In Loops
Challenge C044 -- AgentZero Session 045
"""

import pytest
from for_in_loops import (
    run, execute, parse, compile_source, lex,
    ParseError, CompileError, VMError, LexError,
)


# ============================================================
# Basic for-in over arrays
# ============================================================

class TestForInArray:
    def test_iterate_array_literal(self):
        result, output = run("""
            for (x in [1, 2, 3]) {
                print(x);
            }
        """)
        assert output == ['1', '2', '3']

    def test_iterate_array_variable(self):
        result, output = run("""
            let arr = [10, 20, 30];
            for (x in arr) {
                print(x);
            }
        """)
        assert output == ['10', '20', '30']

    def test_iterate_empty_array(self):
        result, output = run("""
            for (x in []) {
                print(x);
            }
            print("done");
        """)
        assert output == ['done']

    def test_iterate_single_element(self):
        result, output = run("""
            for (x in [42]) {
                print(x);
            }
        """)
        assert output == ['42']

    def test_accumulate_sum(self):
        result, output = run("""
            let sum = 0;
            for (x in [1, 2, 3, 4, 5]) {
                sum = sum + x;
            }
            print(sum);
        """)
        assert output == ['15']

    def test_iterate_string_array(self):
        result, output = run("""
            for (s in ["hello", "world"]) {
                print(s);
            }
        """)
        assert output == ['hello', 'world']

    def test_iterate_mixed_types(self):
        result, output = run("""
            for (x in [1, "two", true, false]) {
                print(x);
            }
        """)
        assert output == ['1', 'two', 'true', 'false']

    def test_iterate_nested_arrays(self):
        result, output = run("""
            for (arr in [[1, 2], [3, 4]]) {
                print(len(arr));
            }
        """)
        assert output == ['2', '2']

    def test_iterate_range(self):
        result, output = run("""
            for (i in range(5)) {
                print(i);
            }
        """)
        assert output == ['0', '1', '2', '3', '4']

    def test_iterate_range_start_end(self):
        result, output = run("""
            for (i in range(2, 5)) {
                print(i);
            }
        """)
        assert output == ['2', '3', '4']

    def test_index_during_iteration(self):
        """Access the iteration variable as an index."""
        result, output = run("""
            let arr = [10, 20, 30];
            for (i in range(3)) {
                print(arr[i]);
            }
        """)
        assert output == ['10', '20', '30']

    def test_modify_external_variable(self):
        result, output = run("""
            let count = 0;
            for (x in [1, 2, 3]) {
                count = count + 1;
            }
            print(count);
        """)
        assert output == ['3']

    def test_iterate_array_of_hashes(self):
        result, output = run("""
            let items = [{name: "a"}, {name: "b"}];
            for (item in items) {
                print(item.name);
            }
        """)
        assert output == ['a', 'b']


# ============================================================
# For-in over hash maps (single variable = keys)
# ============================================================

class TestForInHash:
    def test_iterate_hash_keys(self):
        result, output = run("""
            let m = {a: 1, b: 2, c: 3};
            let result = [];
            for (k in m) {
                push(result, k);
            }
            print(len(result));
        """)
        assert output == ['3']

    def test_iterate_empty_hash(self):
        result, output = run("""
            for (k in {}) {
                print(k);
            }
            print("done");
        """)
        assert output == ['done']

    def test_hash_key_access_value(self):
        result, output = run("""
            let m = {x: 10, y: 20};
            let sum = 0;
            for (k in m) {
                sum = sum + m[k];
            }
            print(sum);
        """)
        assert output == ['30']


# ============================================================
# Destructured for-in (k, v in hash)
# ============================================================

class TestForInDestructured:
    def test_kv_iteration(self):
        result, output = run("""
            let m = {a: 1, b: 2};
            for (k, v in m) {
                print(k);
                print(v);
            }
        """)
        # Dict order is preserved in Python 3.7+
        assert output == ['a', '1', 'b', '2']

    def test_kv_sum_values(self):
        result, output = run("""
            let scores = {math: 90, english: 85, science: 95};
            let total = 0;
            for (subject, score in scores) {
                total = total + score;
            }
            print(total);
        """)
        assert output == ['270']

    def test_kv_empty_hash(self):
        result, output = run("""
            for (k, v in {}) {
                print(k);
            }
            print("done");
        """)
        assert output == ['done']

    def test_kv_single_entry(self):
        result, output = run("""
            for (k, v in {name: "Alice"}) {
                print(k);
                print(v);
            }
        """)
        assert output == ['name', 'Alice']

    def test_kv_collect_entries(self):
        result, output = run("""
            let m = {x: 1, y: 2, z: 3};
            let keys_list = [];
            let vals_list = [];
            for (k, v in m) {
                push(keys_list, k);
                push(vals_list, v);
            }
            print(len(keys_list));
            print(len(vals_list));
        """)
        assert output == ['3', '3']


# ============================================================
# For-in over strings
# ============================================================

class TestForInString:
    def test_iterate_string_chars(self):
        result, output = run("""
            for (ch in "abc") {
                print(ch);
            }
        """)
        assert output == ['a', 'b', 'c']

    def test_iterate_empty_string(self):
        result, output = run("""
            for (ch in "") {
                print(ch);
            }
            print("done");
        """)
        assert output == ['done']

    def test_count_chars(self):
        result, output = run("""
            let count = 0;
            for (ch in "hello") {
                count = count + 1;
            }
            print(count);
        """)
        assert output == ['5']


# ============================================================
# Break statement
# ============================================================

class TestBreak:
    def test_break_in_for_in(self):
        result, output = run("""
            for (x in [1, 2, 3, 4, 5]) {
                if (x == 3) {
                    break;
                }
                print(x);
            }
        """)
        assert output == ['1', '2']

    def test_break_in_while(self):
        result, output = run("""
            let i = 0;
            while (true) {
                if (i == 3) {
                    break;
                }
                print(i);
                i = i + 1;
            }
        """)
        assert output == ['0', '1', '2']

    def test_break_first_iteration(self):
        result, output = run("""
            for (x in [1, 2, 3]) {
                break;
            }
            print("done");
        """)
        assert output == ['done']

    def test_break_in_nested_for(self):
        """Break only exits the innermost loop."""
        result, output = run("""
            for (i in [1, 2, 3]) {
                for (j in [10, 20, 30]) {
                    if (j == 20) {
                        break;
                    }
                    print(j);
                }
                print(i);
            }
        """)
        assert output == ['10', '1', '10', '2', '10', '3']

    def test_break_with_accumulation(self):
        result, output = run("""
            let sum = 0;
            for (x in range(100)) {
                if (x >= 5) {
                    break;
                }
                sum = sum + x;
            }
            print(sum);
        """)
        assert output == ['10']  # 0+1+2+3+4

    def test_break_outside_loop_error(self):
        with pytest.raises(CompileError, match="outside of loop"):
            run("break;")


# ============================================================
# Continue statement
# ============================================================

class TestContinue:
    def test_continue_in_for_in(self):
        result, output = run("""
            for (x in [1, 2, 3, 4, 5]) {
                if (x == 3) {
                    continue;
                }
                print(x);
            }
        """)
        assert output == ['1', '2', '4', '5']

    def test_continue_in_while(self):
        result, output = run("""
            let i = 0;
            while (i < 5) {
                i = i + 1;
                if (i == 3) {
                    continue;
                }
                print(i);
            }
        """)
        assert output == ['1', '2', '4', '5']

    def test_continue_skip_all(self):
        result, output = run("""
            for (x in [1, 2, 3]) {
                continue;
            }
            print("done");
        """)
        assert output == ['done']

    def test_continue_in_nested_loops(self):
        result, output = run("""
            for (i in [1, 2]) {
                for (j in [10, 20, 30]) {
                    if (j == 20) {
                        continue;
                    }
                    print(i * 100 + j);
                }
            }
        """)
        assert output == ['110', '130', '210', '230']

    def test_continue_with_accumulation(self):
        result, output = run("""
            let sum = 0;
            for (x in range(10)) {
                if (x % 2 == 0) {
                    continue;
                }
                sum = sum + x;
            }
            print(sum);
        """)
        assert output == ['25']  # 1+3+5+7+9

    def test_continue_outside_loop_error(self):
        with pytest.raises(CompileError, match="outside of loop"):
            run("continue;")


# ============================================================
# Break + Continue combined
# ============================================================

class TestBreakContinue:
    def test_break_and_continue(self):
        result, output = run("""
            for (x in range(10)) {
                if (x % 2 == 0) {
                    continue;
                }
                if (x >= 7) {
                    break;
                }
                print(x);
            }
        """)
        assert output == ['1', '3', '5']

    def test_break_continue_in_while(self):
        result, output = run("""
            let i = 0;
            while (i < 20) {
                i = i + 1;
                if (i % 3 == 0) {
                    continue;
                }
                if (i > 10) {
                    break;
                }
                print(i);
            }
        """)
        assert output == ['1', '2', '4', '5', '7', '8', '10']


# ============================================================
# Nested for-in loops
# ============================================================

class TestNestedForIn:
    def test_nested_array_iteration(self):
        result, output = run("""
            for (i in [1, 2]) {
                for (j in [3, 4]) {
                    print(i * 10 + j);
                }
            }
        """)
        assert output == ['13', '14', '23', '24']

    def test_triple_nested(self):
        result, output = run("""
            let count = 0;
            for (i in range(3)) {
                for (j in range(3)) {
                    for (k in range(3)) {
                        count = count + 1;
                    }
                }
            }
            print(count);
        """)
        assert output == ['27']

    def test_nested_with_hash_and_array(self):
        result, output = run("""
            let data = {fruits: ["apple", "banana"], vegs: ["carrot"]};
            for (k, v in data) {
                for (item in v) {
                    print(item);
                }
            }
        """)
        assert output == ['apple', 'banana', 'carrot']

    def test_matrix_iteration(self):
        result, output = run("""
            let matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
            let flat = [];
            for (row in matrix) {
                for (val in row) {
                    push(flat, val);
                }
            }
            print(len(flat));
            print(flat[0]);
            print(flat[8]);
        """)
        assert output == ['9', '1', '9']


# ============================================================
# For-in with closures and functions
# ============================================================

class TestForInWithFunctions:
    def test_for_in_calls_function(self):
        result, output = run("""
            fn double(x) { return x * 2; }
            for (x in [1, 2, 3]) {
                print(double(x));
            }
        """)
        assert output == ['2', '4', '6']

    def test_for_in_with_lambda(self):
        result, output = run("""
            let transform = fn(x) { return x + 10; };
            for (x in [1, 2, 3]) {
                print(transform(x));
            }
        """)
        assert output == ['11', '12', '13']

    def test_for_in_builds_closures(self):
        result, output = run("""
            let fns = [];
            for (i in range(3)) {
                let val = i;
                push(fns, fn() { return val; });
            }
            for (f in fns) {
                print(f());
            }
        """)
        assert output == ['0', '1', '2']

    def test_for_in_inside_function(self):
        result, output = run("""
            fn sum_array(arr) {
                let total = 0;
                for (x in arr) {
                    total = total + x;
                }
                return total;
            }
            print(sum_array([1, 2, 3, 4]));
        """)
        assert output == ['10']

    def test_for_in_with_map_builtin(self):
        result, output = run("""
            let doubled = map([1, 2, 3], fn(x) { return x * 2; });
            for (x in doubled) {
                print(x);
            }
        """)
        assert output == ['2', '4', '6']


# ============================================================
# For-in with hash map operations
# ============================================================

class TestForInHashOps:
    def test_for_in_with_has(self):
        result, output = run("""
            let m = {a: 1, b: 2, c: 3};
            let target_keys = ["a", "c", "d"];
            for (k in target_keys) {
                if (has(m, k)) {
                    print(m[k]);
                }
            }
        """)
        assert output == ['1', '3']

    def test_for_in_build_hash(self):
        result, output = run("""
            let result = {};
            let keys_arr = ["a", "b", "c"];
            let vals = [1, 2, 3];
            for (i in range(3)) {
                result[keys_arr[i]] = vals[i];
            }
            print(result.a);
            print(result.b);
            print(result.c);
        """)
        assert output == ['1', '2', '3']

    def test_for_in_merge_hashes(self):
        result, output = run("""
            let hashes = [{a: 1}, {b: 2}, {c: 3}];
            let combined = {};
            for (h in hashes) {
                combined = merge(combined, h);
            }
            print(size(combined));
        """)
        assert output == ['3']


# ============================================================
# For-in with array mutation
# ============================================================

class TestForInMutation:
    def test_build_array_from_iteration(self):
        result, output = run("""
            let squares = [];
            for (i in range(5)) {
                push(squares, i * i);
            }
            for (x in squares) {
                print(x);
            }
        """)
        assert output == ['0', '1', '4', '9', '16']

    def test_filter_manually(self):
        result, output = run("""
            let evens = [];
            for (x in range(10)) {
                if (x % 2 == 0) {
                    push(evens, x);
                }
            }
            print(len(evens));
            print(evens[0]);
            print(evens[4]);
        """)
        assert output == ['5', '0', '8']


# ============================================================
# For-in with break/continue in while (mixed loops)
# ============================================================

class TestMixedLoops:
    def test_while_inside_for_in(self):
        result, output = run("""
            for (x in [1, 2, 3]) {
                let i = 0;
                while (i < x) {
                    print(i);
                    i = i + 1;
                }
            }
        """)
        assert output == ['0', '0', '1', '0', '1', '2']

    def test_for_in_inside_while(self):
        result, output = run("""
            let round = 0;
            while (round < 2) {
                for (x in [10, 20]) {
                    print(round * 100 + x);
                }
                round = round + 1;
            }
        """)
        assert output == ['10', '20', '110', '120']

    def test_break_from_while_inside_for(self):
        result, output = run("""
            for (x in [1, 2, 3]) {
                let i = 0;
                while (true) {
                    if (i >= 2) { break; }
                    print(x * 10 + i);
                    i = i + 1;
                }
            }
        """)
        assert output == ['10', '11', '20', '21', '30', '31']

    def test_continue_in_for_inside_while(self):
        result, output = run("""
            let n = 0;
            while (n < 2) {
                for (x in [1, 2, 3]) {
                    if (x == 2) { continue; }
                    print(n * 10 + x);
                }
                n = n + 1;
            }
        """)
        assert output == ['1', '3', '11', '13']


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_for_in_variable_shadowing(self):
        result, output = run("""
            let x = 99;
            for (x in [1, 2, 3]) {
                print(x);
            }
            // x keeps its last loop value
            print(x);
        """)
        assert output == ['1', '2', '3', '3']

    def test_for_in_large_range(self):
        result, output = run("""
            let sum = 0;
            for (i in range(100)) {
                sum = sum + i;
            }
            print(sum);
        """)
        assert output == ['4950']

    def test_for_in_single_char_string(self):
        result, output = run("""
            for (ch in "x") {
                print(ch);
            }
        """)
        assert output == ['x']

    def test_for_in_with_boolean_array(self):
        result, output = run("""
            for (b in [true, false, true]) {
                print(b);
            }
        """)
        assert output == ['true', 'false', 'true']

    def test_for_in_computed_iterable(self):
        result, output = run("""
            fn get_items() { return [5, 10, 15]; }
            for (x in get_items()) {
                print(x);
            }
        """)
        assert output == ['5', '10', '15']

    def test_for_in_concat_result(self):
        result, output = run("""
            for (x in concat([1, 2], [3, 4])) {
                print(x);
            }
        """)
        assert output == ['1', '2', '3', '4']

    def test_for_in_slice_result(self):
        result, output = run("""
            let arr = [10, 20, 30, 40, 50];
            for (x in slice(arr, 1, 4)) {
                print(x);
            }
        """)
        assert output == ['20', '30', '40']


# ============================================================
# Error cases
# ============================================================

class TestErrors:
    def test_iterate_non_iterable_int(self):
        with pytest.raises(VMError, match="Cannot iterate"):
            run("for (x in 42) { print(x); }")

    def test_iterate_non_iterable_bool(self):
        with pytest.raises(VMError, match="Cannot iterate"):
            run("for (x in true) { print(x); }")

    def test_break_outside_loop(self):
        with pytest.raises(CompileError, match="outside of loop"):
            run("break;")

    def test_continue_outside_loop(self):
        with pytest.raises(CompileError, match="outside of loop"):
            run("continue;")

    def test_missing_in_keyword(self):
        with pytest.raises(ParseError):
            run("for (x [1, 2, 3]) { print(x); }")

    def test_missing_parens(self):
        with pytest.raises(ParseError):
            run("for x in [1, 2, 3] { print(x); }")


# ============================================================
# All C043 features still work (regression)
# ============================================================

class TestRegression:
    def test_hash_literal(self):
        result, output = run("""
            let m = {name: "Alice", age: 30};
            print(m.name);
            print(m.age);
        """)
        assert output == ['Alice', '30']

    def test_array_operations(self):
        result, output = run("""
            let arr = [1, 2, 3];
            push(arr, 4);
            print(len(arr));
            print(arr[3]);
        """)
        assert output == ['4', '4']

    def test_closures(self):
        result, output = run("""
            fn make_adder(n) {
                return fn(x) { return x + n; };
            }
            let add5 = make_adder(5);
            print(add5(10));
        """)
        assert output == ['15']

    def test_while_loop(self):
        result, output = run("""
            let i = 0;
            while (i < 3) {
                print(i);
                i = i + 1;
            }
        """)
        assert output == ['0', '1', '2']

    def test_if_else(self):
        result, output = run("""
            let x = 10;
            if (x > 5) {
                print("big");
            } else {
                print("small");
            }
        """)
        assert output == ['big']

    def test_dot_assign(self):
        result, output = run("""
            let obj = {x: 1};
            obj.x = 42;
            print(obj.x);
        """)
        assert output == ['42']

    def test_hash_builtins(self):
        result, output = run("""
            let m = {a: 1, b: 2};
            print(has(m, "a"));
            print(size(m));
            let m2 = merge(m, {c: 3});
            print(size(m2));
        """)
        assert output == ['true', '2', '3']

    def test_array_builtins(self):
        result, output = run("""
            let arr = [3, 1, 2];
            let sorted_arr = sort(arr);
            for (x in sorted_arr) {
                print(x);
            }
        """)
        assert output == ['1', '2', '3']

    def test_recursive_function(self):
        result, output = run("""
            fn fib(n) {
                if (n <= 1) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
            print(fib(7));
        """)
        assert output == ['13']

    def test_higher_order_functions(self):
        result, output = run("""
            let doubled = map([1, 2, 3], fn(x) { return x * 2; });
            let sum = reduce(doubled, fn(acc, x) { return acc + x; }, 0);
            print(sum);
        """)
        assert output == ['12']

    def test_nested_hash_array(self):
        result, output = run("""
            let data = {items: [1, 2, 3], meta: {count: 3}};
            print(data.items[1]);
            print(data.meta.count);
        """)
        assert output == ['2', '3']


# ============================================================
# Complex integration tests
# ============================================================

class TestIntegration:
    def test_fizzbuzz_with_for_in(self):
        result, output = run("""
            for (i in range(1, 16)) {
                if (i % 15 == 0) {
                    print("fizzbuzz");
                } else {
                    if (i % 3 == 0) {
                        print("fizz");
                    } else {
                        if (i % 5 == 0) {
                            print("buzz");
                        } else {
                            print(i);
                        }
                    }
                }
            }
        """)
        expected = ['1', '2', 'fizz', '4', 'buzz', 'fizz', '7', '8', 'fizz', 'buzz',
                    '11', 'fizz', '13', '14', 'fizzbuzz']
        assert output == expected

    def test_collect_unique_values(self):
        result, output = run("""
            let seen = {};
            let unique = [];
            let items = [1, 2, 3, 2, 1, 4, 3, 5];
            for (x in items) {
                if (not has(seen, x)) {
                    seen[x] = true;
                    push(unique, x);
                }
            }
            print(len(unique));
            for (u in unique) {
                print(u);
            }
        """)
        assert output == ['5', '1', '2', '3', '4', '5']

    def test_transpose_matrix(self):
        result, output = run("""
            let matrix = [[1, 2, 3], [4, 5, 6]];
            let rows = len(matrix);
            let cols = len(matrix[0]);
            let transposed = [];
            for (j in range(cols)) {
                let row = [];
                for (i in range(rows)) {
                    push(row, matrix[i][j]);
                }
                push(transposed, row);
            }
            print(len(transposed));
            for (row in transposed) {
                print(row);
            }
        """)
        assert output == ['3', '[1, 4]', '[2, 5]', '[3, 6]']

    def test_word_frequency(self):
        result, output = run("""
            let words = ["the", "cat", "sat", "on", "the", "mat", "the"];
            let freq = {};
            for (w in words) {
                if (has(freq, w)) {
                    freq[w] = freq[w] + 1;
                } else {
                    freq[w] = 1;
                }
            }
            print(freq["the"]);
            print(freq["cat"]);
        """)
        assert output == ['3', '1']

    def test_group_by(self):
        result, output = run("""
            let items = [
                {type: "fruit", name: "apple"},
                {type: "veg", name: "carrot"},
                {type: "fruit", name: "banana"},
                {type: "veg", name: "pea"}
            ];
            let groups = {};
            for (item in items) {
                let t = item.type;
                if (not has(groups, t)) {
                    groups[t] = [];
                }
                push(groups[t], item.name);
            }
            print(len(groups["fruit"]));
            print(len(groups["veg"]));
        """)
        assert output == ['2', '2']

    def test_flatten_nested(self):
        result, output = run("""
            fn flatten(arr) {
                let result = [];
                for (item in arr) {
                    for (x in item) {
                        push(result, x);
                    }
                }
                return result;
            }
            let nested = [[1, 2], [3], [4, 5, 6]];
            let flat = flatten(nested);
            print(len(flat));
            for (x in flat) {
                print(x);
            }
        """)
        assert output == ['6', '1', '2', '3', '4', '5', '6']

    def test_find_with_for_in(self):
        result, output = run("""
            let items = [
                {id: 1, name: "Alice"},
                {id: 2, name: "Bob"},
                {id: 3, name: "Charlie"}
            ];
            let found = "none";
            for (item in items) {
                if (item.id == 2) {
                    found = item.name;
                    break;
                }
            }
            print(found);
        """)
        assert output == ['Bob']

    def test_enumerate_pattern(self):
        """Simulate enumerate by iterating range and indexing."""
        result, output = run("""
            let items = ["a", "b", "c"];
            for (i in range(len(items))) {
                print(i);
                print(items[i]);
            }
        """)
        assert output == ['0', 'a', '1', 'b', '2', 'c']

    def test_zip_pattern(self):
        """Simulate zip by iterating indices."""
        result, output = run("""
            let names = ["Alice", "Bob", "Charlie"];
            let ages = [30, 25, 35];
            for (i in range(len(names))) {
                print(names[i]);
                print(ages[i]);
            }
        """)
        assert output == ['Alice', '30', 'Bob', '25', 'Charlie', '35']

    def test_bubble_sort(self):
        result, output = run("""
            let arr = [5, 3, 8, 1, 2];
            let n = len(arr);
            for (i in range(n)) {
                for (j in range(n - 1)) {
                    if (arr[j] > arr[j + 1]) {
                        let temp = arr[j];
                        arr[j] = arr[j + 1];
                        arr[j + 1] = temp;
                    }
                }
            }
            for (x in arr) {
                print(x);
            }
        """)
        assert output == ['1', '2', '3', '5', '8']
