"""
Tests for C042: Arrays
Challenge C042 -- AgentZero Session 043
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from arrays import (
    run, execute, parse, compile_source, lex, disassemble,
    VMError, ParseError, LexError, CompileError,
    ArrayLit, IndexExpr, IndexAssign, TokenType,
)


# ============================================================
# 1. Array Literals
# ============================================================

class TestArrayLiterals:
    def test_empty_array(self):
        result, output = run("let a = []; a;")
        assert result == []

    def test_single_element(self):
        result, output = run("let a = [42]; a;")
        assert result == [42]

    def test_multiple_elements(self):
        result, output = run("let a = [1, 2, 3]; a;")
        assert result == [1, 2, 3]

    def test_mixed_types(self):
        result, output = run('let a = [1, "hello", true, 3.14]; a;')
        assert result == [1, "hello", True, 3.14]

    def test_nested_arrays(self):
        result, output = run("let a = [[1, 2], [3, 4]]; a;")
        assert result == [[1, 2], [3, 4]]

    def test_deeply_nested(self):
        result, output = run("let a = [[[1]]]; a;")
        assert result == [[[1]]]

    def test_trailing_comma(self):
        result, output = run("let a = [1, 2, 3,]; a;")
        assert result == [1, 2, 3]

    def test_expression_elements(self):
        result, output = run("let a = [1 + 2, 3 * 4, 10 - 5]; a;")
        assert result == [3, 12, 5]

    def test_variable_elements(self):
        result, output = run("let x = 10; let a = [x, x + 1, x + 2]; a;")
        assert result == [10, 11, 12]

    def test_array_with_booleans(self):
        result, output = run("let a = [true, false, true]; a;")
        assert result == [True, False, True]

    def test_array_with_strings(self):
        result, output = run('let a = ["a", "b", "c"]; a;')
        assert result == ["a", "b", "c"]


# ============================================================
# 2. Array Indexing
# ============================================================

class TestArrayIndexing:
    def test_index_first(self):
        result, output = run("let a = [10, 20, 30]; a[0];")
        assert result == 10

    def test_index_middle(self):
        result, output = run("let a = [10, 20, 30]; a[1];")
        assert result == 20

    def test_index_last(self):
        result, output = run("let a = [10, 20, 30]; a[2];")
        assert result == 30

    def test_index_with_variable(self):
        result, output = run("let a = [10, 20, 30]; let i = 1; a[i];")
        assert result == 20

    def test_index_with_expression(self):
        result, output = run("let a = [10, 20, 30]; a[1 + 1];")
        assert result == 30

    def test_nested_index(self):
        result, output = run("let a = [[1, 2], [3, 4]]; a[0][1];")
        assert result == 2

    def test_nested_index_deep(self):
        result, output = run("let a = [[10, 20], [30, 40]]; a[1][0];")
        assert result == 30

    def test_index_out_of_bounds(self):
        with pytest.raises(VMError, match="out of bounds"):
            run("let a = [1, 2, 3]; a[5];")

    def test_index_negative(self):
        with pytest.raises(VMError, match="out of bounds"):
            run("let a = [1, 2, 3]; a[-1];")

    def test_index_non_integer(self):
        with pytest.raises(VMError, match="must be integer"):
            run('let a = [1, 2, 3]; a["x"];')

    def test_index_on_non_array(self):
        with pytest.raises(VMError, match="Cannot index"):
            run("let a = 42; a[0];")


# ============================================================
# 3. Index Assignment
# ============================================================

class TestIndexAssignment:
    def test_set_first(self):
        result, output = run("let a = [1, 2, 3]; a[0] = 99; a;")
        assert result == [99, 2, 3]

    def test_set_last(self):
        result, output = run("let a = [1, 2, 3]; a[2] = 99; a;")
        assert result == [1, 2, 99]

    def test_set_returns_value(self):
        result, output = run("let a = [1, 2, 3]; a[1] = 42;")
        assert result == 42

    def test_set_with_expression(self):
        result, output = run("let a = [1, 2, 3]; let i = 0; a[i] = a[i] + 10; a;")
        assert result == [11, 2, 3]

    def test_set_nested(self):
        result, output = run("let a = [[1, 2], [3, 4]]; a[0][1] = 99; a;")
        assert result == [[1, 99], [3, 4]]

    def test_set_out_of_bounds(self):
        with pytest.raises(VMError, match="out of bounds"):
            run("let a = [1, 2, 3]; a[5] = 99;")

    def test_set_on_non_array(self):
        with pytest.raises(VMError, match="Cannot assign to index"):
            run("let x = 42; x[0] = 1;")

    def test_mutation_visible_elsewhere(self):
        """Arrays are mutable references."""
        result, output = run("""
            let a = [1, 2, 3];
            let b = a;
            b[0] = 99;
            a[0];
        """)
        assert result == 99


# ============================================================
# 4. Array Concatenation
# ============================================================

class TestArrayConcat:
    def test_concat_plus(self):
        result, output = run("let a = [1, 2] + [3, 4]; a;")
        assert result == [1, 2, 3, 4]

    def test_concat_empty(self):
        result, output = run("let a = [] + [1, 2]; a;")
        assert result == [1, 2]

    def test_concat_both_empty(self):
        result, output = run("let a = [] + []; a;")
        assert result == []

    def test_concat_preserves_originals(self):
        result, output = run("""
            let a = [1, 2];
            let b = [3, 4];
            let c = a + b;
            a[0] = 99;
            c[0];
        """)
        assert result == 1  # c is a new array, not affected by mutating a

    def test_array_repeat(self):
        result, output = run("let a = [1, 2] * 3; a;")
        assert result == [1, 2, 1, 2, 1, 2]


# ============================================================
# 5. Array Equality
# ============================================================

class TestArrayEquality:
    def test_equal_arrays(self):
        result, output = run("[1, 2, 3] == [1, 2, 3];")
        assert result == True

    def test_unequal_arrays(self):
        result, output = run("[1, 2, 3] == [1, 2, 4];")
        assert result == False

    def test_different_lengths(self):
        result, output = run("[1, 2] == [1, 2, 3];")
        assert result == False

    def test_empty_equal(self):
        result, output = run("[] == [];")
        assert result == True

    def test_not_equal(self):
        result, output = run("[1] != [2];")
        assert result == True

    def test_nested_equality(self):
        result, output = run("[[1, 2], [3]] == [[1, 2], [3]];")
        assert result == True


# ============================================================
# 6. Builtin: len()
# ============================================================

class TestLen:
    def test_len_array(self):
        result, output = run("len([1, 2, 3]);")
        assert result == 3

    def test_len_empty(self):
        result, output = run("len([]);")
        assert result == 0

    def test_len_string(self):
        result, output = run('len("hello");')
        assert result == 5

    def test_len_nested(self):
        result, output = run("len([[1], [2], [3]]);")
        assert result == 3

    def test_len_wrong_type(self):
        with pytest.raises(VMError, match="requires array or string"):
            run("len(42);")

    def test_len_wrong_args(self):
        with pytest.raises(VMError, match="takes 1 argument"):
            run("len([1], [2]);")


# ============================================================
# 7. Builtin: push() and pop()
# ============================================================

class TestPushPop:
    def test_push(self):
        result, output = run("let a = [1, 2]; push(a, 3); a;")
        assert result == [1, 2, 3]

    def test_push_returns_array(self):
        result, output = run("push([1, 2], 3);")
        assert result == [1, 2, 3]

    def test_push_to_empty(self):
        result, output = run("let a = []; push(a, 1); a;")
        assert result == [1]

    def test_pop(self):
        result, output = run("let a = [1, 2, 3]; pop(a);")
        assert result == 3

    def test_pop_modifies(self):
        result, output = run("let a = [1, 2, 3]; pop(a); a;")
        assert result == [1, 2]

    def test_pop_empty(self):
        with pytest.raises(VMError, match="empty array"):
            run("pop([]);")

    def test_push_non_array(self):
        with pytest.raises(VMError, match="requires array"):
            run("push(42, 1);")


# ============================================================
# 8. Builtin: range()
# ============================================================

class TestRange:
    def test_range_single(self):
        result, output = run("range(5);")
        assert result == [0, 1, 2, 3, 4]

    def test_range_start_end(self):
        result, output = run("range(2, 6);")
        assert result == [2, 3, 4, 5]

    def test_range_step(self):
        result, output = run("range(0, 10, 2);")
        assert result == [0, 2, 4, 6, 8]

    def test_range_empty(self):
        result, output = run("range(0);")
        assert result == []

    def test_range_negative_step(self):
        result, output = run("range(5, 0, -1);")
        assert result == [5, 4, 3, 2, 1]


# ============================================================
# 9. Builtin: map()
# ============================================================

class TestMap:
    def test_map_double(self):
        result, output = run("""
            let arr = [1, 2, 3];
            map(arr, fn(x) { return x * 2; });
        """)
        assert result == [2, 4, 6]

    def test_map_empty(self):
        result, output = run("map([], fn(x) { return x; });")
        assert result == []

    def test_map_preserves_original(self):
        result, output = run("""
            let arr = [1, 2, 3];
            let doubled = map(arr, fn(x) { return x * 2; });
            arr;
        """)
        assert result == [1, 2, 3]

    def test_map_with_named_fn(self):
        result, output = run("""
            fn square(x) { return x * x; }
            map([1, 2, 3, 4], square);
        """)
        assert result == [1, 4, 9, 16]

    def test_map_strings(self):
        result, output = run("""
            map([1, 2, 3], fn(x) { return x + 10; });
        """)
        assert result == [11, 12, 13]


# ============================================================
# 10. Builtin: filter()
# ============================================================

class TestFilter:
    def test_filter_even(self):
        result, output = run("""
            filter([1, 2, 3, 4, 5, 6], fn(x) { return x % 2 == 0; });
        """)
        assert result == [2, 4, 6]

    def test_filter_empty_result(self):
        result, output = run("""
            filter([1, 3, 5], fn(x) { return x % 2 == 0; });
        """)
        assert result == []

    def test_filter_all_pass(self):
        result, output = run("""
            filter([2, 4, 6], fn(x) { return x % 2 == 0; });
        """)
        assert result == [2, 4, 6]

    def test_filter_preserves_original(self):
        result, output = run("""
            let arr = [1, 2, 3, 4, 5];
            let evens = filter(arr, fn(x) { return x % 2 == 0; });
            arr;
        """)
        assert result == [1, 2, 3, 4, 5]


# ============================================================
# 11. Builtin: reduce()
# ============================================================

class TestReduce:
    def test_reduce_sum(self):
        result, output = run("""
            reduce([1, 2, 3, 4], fn(acc, x) { return acc + x; }, 0);
        """)
        assert result == 10

    def test_reduce_product(self):
        result, output = run("""
            reduce([1, 2, 3, 4], fn(acc, x) { return acc * x; }, 1);
        """)
        assert result == 24

    def test_reduce_empty(self):
        result, output = run("""
            reduce([], fn(acc, x) { return acc + x; }, 42);
        """)
        assert result == 42

    def test_reduce_single_element(self):
        result, output = run("""
            reduce([5], fn(acc, x) { return acc + x; }, 0);
        """)
        assert result == 5

    def test_reduce_build_string(self):
        # Concatenate with strings
        result, output = run("""
            reduce(["a", "b", "c"], fn(acc, x) { return acc + x; }, "");
        """)
        assert result == "abc"


# ============================================================
# 12. Builtin: slice()
# ============================================================

class TestSlice:
    def test_slice_middle(self):
        result, output = run("slice([1, 2, 3, 4, 5], 1, 3);")
        assert result == [2, 3]

    def test_slice_from_start(self):
        result, output = run("slice([1, 2, 3, 4], 0, 2);")
        assert result == [1, 2]

    def test_slice_to_end(self):
        result, output = run("slice([1, 2, 3, 4], 2);")
        assert result == [3, 4]

    def test_slice_empty(self):
        result, output = run("slice([1, 2, 3], 1, 1);")
        assert result == []

    def test_slice_full(self):
        result, output = run("slice([1, 2, 3], 0, 3);")
        assert result == [1, 2, 3]


# ============================================================
# 13. Builtin: sort(), reverse()
# ============================================================

class TestSortReverse:
    def test_sort(self):
        result, output = run("sort([3, 1, 2]);")
        assert result == [1, 2, 3]

    def test_sort_already_sorted(self):
        result, output = run("sort([1, 2, 3]);")
        assert result == [1, 2, 3]

    def test_sort_empty(self):
        result, output = run("sort([]);")
        assert result == []

    def test_sort_preserves_original(self):
        result, output = run("""
            let a = [3, 1, 2];
            let b = sort(a);
            a;
        """)
        assert result == [3, 1, 2]

    def test_reverse(self):
        result, output = run("reverse([1, 2, 3]);")
        assert result == [3, 2, 1]

    def test_reverse_empty(self):
        result, output = run("reverse([]);")
        assert result == []

    def test_reverse_single(self):
        result, output = run("reverse([42]);")
        assert result == [42]


# ============================================================
# 14. Builtin: find(), each(), concat()
# ============================================================

class TestFindEachConcat:
    def test_find_present(self):
        result, output = run("""
            find([1, 2, 3, 4, 5], fn(x) { return x > 3; });
        """)
        assert result == 4

    def test_find_absent(self):
        result, output = run("""
            find([1, 2, 3], fn(x) { return x > 10; });
        """)
        assert result is None

    def test_each_side_effect(self):
        result, output = run("""
            each([1, 2, 3], fn(x) { print(x); });
        """)
        assert output == ["1", "2", "3"]

    def test_concat_builtin(self):
        result, output = run("concat([1, 2], [3, 4]);")
        assert result == [1, 2, 3, 4]


# ============================================================
# 15. Printing Arrays
# ============================================================

class TestPrintArrays:
    def test_print_array(self):
        result, output = run("print([1, 2, 3]);")
        assert output == ["[1, 2, 3]"]

    def test_print_empty(self):
        result, output = run("print([]);")
        assert output == ["[]"]

    def test_print_nested(self):
        result, output = run("print([[1, 2], [3, 4]]);")
        assert output == ["[[1, 2], [3, 4]]"]

    def test_print_mixed(self):
        result, output = run('print([1, "hi", true]);')
        assert output == ['[1, hi, true]']

    def test_print_with_bools(self):
        result, output = run("print([true, false]);")
        assert output == ["[true, false]"]


# ============================================================
# 16. Arrays with Closures
# ============================================================

class TestArraysWithClosures:
    def test_closure_captures_array(self):
        result, output = run("""
            let arr = [1, 2, 3];
            fn get_first() { return arr[0]; }
            arr[0] = 99;
            get_first();
        """)
        assert result == 99

    def test_array_of_closures(self):
        result, output = run("""
            let fns = [];
            push(fns, fn(x) { return x + 1; });
            push(fns, fn(x) { return x * 2; });
            fns[0](5) + fns[1](5);
        """)
        assert result == 16  # 6 + 10

    def test_closure_modifies_array(self):
        result, output = run("""
            let arr = [0];
            fn inc() { arr[0] = arr[0] + 1; }
            inc();
            inc();
            inc();
            arr[0];
        """)
        assert result == 3

    def test_map_with_closure_capture(self):
        result, output = run("""
            let factor = 10;
            map([1, 2, 3], fn(x) { return x * factor; });
        """)
        assert result == [10, 20, 30]

    def test_filter_with_closure_capture(self):
        result, output = run("""
            let threshold = 3;
            filter([1, 2, 3, 4, 5], fn(x) { return x > threshold; });
        """)
        assert result == [4, 5]

    def test_make_adder_over_array(self):
        result, output = run("""
            fn make_adder(n) { return fn(x) { return x + n; }; }
            map([1, 2, 3], make_adder(10));
        """)
        assert result == [11, 12, 13]


# ============================================================
# 17. Arrays with Loops
# ============================================================

class TestArraysWithLoops:
    def test_build_array_in_loop(self):
        result, output = run("""
            let arr = [];
            let i = 0;
            while (i < 5) {
                push(arr, i);
                i = i + 1;
            }
            arr;
        """)
        assert result == [0, 1, 2, 3, 4]

    def test_sum_array_in_loop(self):
        result, output = run("""
            let arr = [10, 20, 30];
            let sum = 0;
            let i = 0;
            while (i < len(arr)) {
                sum = sum + arr[i];
                i = i + 1;
            }
            sum;
        """)
        assert result == 60

    def test_reverse_in_place(self):
        result, output = run("""
            let arr = [1, 2, 3, 4, 5];
            let i = 0;
            let j = len(arr) - 1;
            while (i < j) {
                let tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
                i = i + 1;
                j = j - 1;
            }
            arr;
        """)
        assert result == [5, 4, 3, 2, 1]


# ============================================================
# 18. Arrays with Conditionals
# ============================================================

class TestArraysWithConditionals:
    def test_conditional_array(self):
        result, output = run("""
            let x = 5;
            let arr = [];
            if (x > 3) {
                arr = [1, 2, 3];
            } else {
                arr = [4, 5, 6];
            }
            arr;
        """)
        assert result == [1, 2, 3]

    def test_index_in_condition(self):
        result, output = run("""
            let arr = [10, 20, 30];
            if (arr[0] > 5) {
                arr[0];
            } else {
                0;
            }
        """)
        assert result == 10


# ============================================================
# 19. String Indexing (bonus -- reuse INDEX_GET)
# ============================================================

class TestStringIndexing:
    def test_string_index(self):
        result, output = run('"hello"[0];')
        assert result == "h"

    def test_string_index_last(self):
        result, output = run('"hello"[4];')
        assert result == "o"

    def test_string_index_out_of_bounds(self):
        with pytest.raises(VMError, match="out of bounds"):
            run('"hi"[5];')

    def test_string_len(self):
        result, output = run('len("hello");')
        assert result == 5


# ============================================================
# 20. Complex Compositions
# ============================================================

class TestComplexCompositions:
    def test_map_filter_reduce(self):
        """Compose map, filter, reduce."""
        result, output = run("""
            let nums = range(1, 11);
            let evens = filter(nums, fn(x) { return x % 2 == 0; });
            let doubled = map(evens, fn(x) { return x * 2; });
            reduce(doubled, fn(acc, x) { return acc + x; }, 0);
        """)
        # evens: [2, 4, 6, 8, 10], doubled: [4, 8, 12, 16, 20], sum: 60
        assert result == 60

    def test_matrix_access(self):
        result, output = run("""
            let matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
            matrix[1][2];
        """)
        assert result == 6

    def test_matrix_sum(self):
        result, output = run("""
            let matrix = [[1, 2], [3, 4]];
            let sum = 0;
            let i = 0;
            while (i < len(matrix)) {
                let j = 0;
                while (j < len(matrix[i])) {
                    sum = sum + matrix[i][j];
                    j = j + 1;
                }
                i = i + 1;
            }
            sum;
        """)
        assert result == 10

    def test_fibonacci_array(self):
        result, output = run("""
            let fib = [0, 1];
            let i = 2;
            while (i < 10) {
                push(fib, fib[i-1] + fib[i-2]);
                i = i + 1;
            }
            fib;
        """)
        assert result == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_counter_with_array(self):
        """Use an array as a mutable cell in a closure."""
        result, output = run("""
            fn make_counter() {
                let state = [0];
                return fn() {
                    state[0] = state[0] + 1;
                    return state[0];
                };
            }
            let counter = make_counter();
            counter();
            counter();
            counter();
        """)
        assert result == 3

    def test_accumulate_with_each(self):
        result, output = run("""
            let result = [];
            each(range(5), fn(x) { push(result, x * x); });
            result;
        """)
        assert result == [0, 1, 4, 9, 16]

    def test_sort_then_reverse(self):
        result, output = run("""
            let a = [3, 1, 4, 1, 5, 9, 2, 6];
            reverse(sort(a));
        """)
        assert result == [9, 6, 5, 4, 3, 2, 1, 1]

    def test_chained_operations(self):
        result, output = run("""
            let nums = range(1, 8);
            let result = reduce(
                map(
                    filter(nums, fn(x) { return x % 2 != 0; }),
                    fn(x) { return x * x; }
                ),
                fn(acc, x) { return acc + x; },
                0
            );
            result;
        """)
        # odds: [1, 3, 5, 7], squares: [1, 9, 25, 49], sum: 84
        assert result == 84


# ============================================================
# 21. Parsing Tests
# ============================================================

class TestParsing:
    def test_parse_array_literal(self):
        ast = parse("[1, 2, 3];")
        stmt = ast.stmts[0]
        assert isinstance(stmt, ArrayLit)
        assert len(stmt.elements) == 3

    def test_parse_index_expr(self):
        ast = parse("a[0];")
        stmt = ast.stmts[0]
        assert isinstance(stmt, IndexExpr)

    def test_parse_index_assign(self):
        ast = parse("a[0] = 5;")
        stmt = ast.stmts[0]
        assert isinstance(stmt, IndexAssign)

    def test_parse_chained_index(self):
        ast = parse("a[0][1];")
        stmt = ast.stmts[0]
        assert isinstance(stmt, IndexExpr)
        assert isinstance(stmt.obj, IndexExpr)

    def test_parse_array_in_call(self):
        ast = parse("len([1, 2, 3]);")
        # Should parse without error

    def test_lex_brackets(self):
        tokens = lex("[1]")
        types = [t.type for t in tokens]
        assert TokenType.LBRACKET in types
        assert TokenType.RBRACKET in types


# ============================================================
# 22. Error Handling
# ============================================================

class TestErrors:
    def test_unclosed_bracket(self):
        with pytest.raises(ParseError):
            run("[1, 2, 3;")

    def test_index_missing_bracket(self):
        with pytest.raises(ParseError):
            run("a[0;")

    def test_len_no_args(self):
        with pytest.raises(VMError, match="takes 1 argument"):
            run("len();")

    def test_push_one_arg(self):
        with pytest.raises(VMError, match="takes 2 arguments"):
            run("push([1]);")

    def test_reduce_two_args(self):
        with pytest.raises(VMError, match="takes 3 arguments"):
            run("reduce([1, 2], fn(a, b) { return a + b; });")


# ============================================================
# 23. Disassembly
# ============================================================

class TestDisassembly:
    def test_disassemble_array(self):
        chunk, _ = compile_source("[1, 2, 3];")
        text = disassemble(chunk)
        assert "MAKE_ARRAY" in text

    def test_disassemble_index(self):
        chunk, _ = compile_source("let a = [1]; a[0];")
        text = disassemble(chunk)
        assert "INDEX_GET" in text


# ============================================================
# 24. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_array_as_function_arg(self):
        result, output = run("""
            fn first(arr) { return arr[0]; }
            first([99, 100]);
        """)
        assert result == 99

    def test_array_as_return_value(self):
        result, output = run("""
            fn make_pair(a, b) { return [a, b]; }
            let p = make_pair(1, 2);
            p[0] + p[1];
        """)
        assert result == 3

    def test_empty_array_len(self):
        result, output = run("len([]);")
        assert result == 0

    def test_single_element_array(self):
        result, output = run("[42][0];")
        assert result == 42

    def test_array_literal_indexing(self):
        """Direct indexing on a literal."""
        result, output = run("[10, 20, 30][1];")
        assert result == 20

    def test_range_used_in_loop(self):
        result, output = run("""
            let sum = 0;
            let nums = range(1, 6);
            let i = 0;
            while (i < len(nums)) {
                sum = sum + nums[i];
                i = i + 1;
            }
            sum;
        """)
        assert result == 15

    def test_builtin_as_variable(self):
        """Builtins can be used as first-class values (passed around)."""
        # This tests that builtins work when loaded via LOAD
        result, output = run("len([1, 2, 3]);")
        assert result == 3

    def test_large_array(self):
        result, output = run("len(range(100));")
        assert result == 100

    def test_nested_map(self):
        result, output = run("""
            let matrix = [[1, 2], [3, 4]];
            map(matrix, fn(row) { return map(row, fn(x) { return x * 10; }); });
        """)
        assert result == [[10, 20], [30, 40]]

    def test_closure_counter_via_array(self):
        """Classic closure pattern using array as mutable cell."""
        result, output = run("""
            fn counter() {
                let c = [0];
                return [
                    fn() { c[0] = c[0] + 1; return c[0]; },
                    fn() { return c[0]; }
                ];
            }
            let ctr = counter();
            ctr[0]();
            ctr[0]();
            ctr[0]();
            ctr[1]();
        """)
        assert result == 3

    def test_array_contains_none(self):
        result, output = run("""
            fn nothing() {}
            let arr = [1, nothing(), 3];
            arr;
        """)
        assert result == [1, None, 3]


# ============================================================
# 25. Integration: All Features Together
# ============================================================

class TestIntegration:
    def test_bubble_sort(self):
        result, output = run("""
            let arr = [5, 3, 8, 1, 9, 2, 7, 4, 6];
            let n = len(arr);
            let i = 0;
            while (i < n) {
                let j = 0;
                while (j < n - 1 - i) {
                    if (arr[j] > arr[j + 1]) {
                        let tmp = arr[j];
                        arr[j] = arr[j + 1];
                        arr[j + 1] = tmp;
                    }
                    j = j + 1;
                }
                i = i + 1;
            }
            arr;
        """)
        assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_functional_pipeline(self):
        """A full functional pipeline with closures, arrays, and builtins."""
        result, output = run("""
            fn compose(f, g) {
                return fn(x) { return f(g(x)); };
            }
            fn double(x) { return x * 2; }
            fn inc(x) { return x + 1; }
            let double_inc = compose(double, inc);
            map(range(5), double_inc);
        """)
        # inc then double: (0+1)*2=2, (1+1)*2=4, (2+1)*2=6, (3+1)*2=8, (4+1)*2=10
        assert result == [2, 4, 6, 8, 10]

    def test_group_by(self):
        """Group elements using arrays."""
        result, output = run("""
            let evens = [];
            let odds = [];
            each(range(1, 11), fn(x) {
                if (x % 2 == 0) {
                    push(evens, x);
                } else {
                    push(odds, x);
                }
            });
            print(evens);
            print(odds);
        """)
        assert output == ["[2, 4, 6, 8, 10]", "[1, 3, 5, 7, 9]"]

    def test_flatten(self):
        """Flatten a 2D array using reduce + concat."""
        result, output = run("""
            let matrix = [[1, 2], [3, 4], [5, 6]];
            reduce(matrix, fn(acc, row) { return concat(acc, row); }, []);
        """)
        assert result == [1, 2, 3, 4, 5, 6]

    def test_zip_manual(self):
        """Zip two arrays together."""
        result, output = run("""
            let a = [1, 2, 3];
            let b = [10, 20, 30];
            let zipped = [];
            let i = 0;
            while (i < len(a)) {
                push(zipped, [a[i], b[i]]);
                i = i + 1;
            }
            zipped;
        """)
        assert result == [[1, 10], [2, 20], [3, 30]]

    def test_selection_sort(self):
        result, output = run("""
            let arr = [64, 25, 12, 22, 11];
            let n = len(arr);
            let i = 0;
            while (i < n - 1) {
                let min_idx = i;
                let j = i + 1;
                while (j < n) {
                    if (arr[j] < arr[min_idx]) {
                        min_idx = j;
                    }
                    j = j + 1;
                }
                let tmp = arr[i];
                arr[i] = arr[min_idx];
                arr[min_idx] = tmp;
                i = i + 1;
            }
            arr;
        """)
        assert result == [11, 12, 22, 25, 64]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
