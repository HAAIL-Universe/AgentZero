"""Tests for C057: String Methods, Array Methods, and Hash Methods.

Extends C056 (Async/Await) with built-in methods on strings, arrays, and hashes.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from string_methods import Parser, Compiler, VM, run, VMError


def run_code(source):
    """Helper: run source, return (result, output)."""
    return run(source)


def run_val(source):
    """Helper: run source, return just the result value."""
    result, _ = run(source)
    return result


def run_output(source):
    """Helper: run source, return just the printed output."""
    _, output = run(source)
    if isinstance(output, list):
        return "\n".join(output)
    return output.strip()


# ============================================================
# String Methods
# ============================================================

class TestStringLength:
    def test_length_property(self):
        assert run_val('"hello".length;') == 5

    def test_empty_length(self):
        assert run_val('"".length;') == 0

    def test_length_via_variable(self):
        assert run_val('let s = "world"; s.length;') == 5


class TestStringSplit:
    def test_split_basic(self):
        assert run_val('"a,b,c".split(",");') == ["a", "b", "c"]

    def test_split_space(self):
        assert run_val('"hello world".split(" ");') == ["hello", "world"]

    def test_split_with_limit(self):
        assert run_val('"a,b,c,d".split(",", 2);') == ["a", "b", "c,d"]

    def test_split_no_match(self):
        assert run_val('"hello".split(",");') == ["hello"]

    def test_split_empty_sep(self):
        assert run_val('"abc".split("");') == ["a", "b", "c"]


class TestStringTrim:
    def test_trim(self):
        assert run_val('"  hello  ".trim();') == "hello"

    def test_trim_start(self):
        assert run_val('"  hello  ".trimStart();') == "hello  "

    def test_trim_end(self):
        assert run_val('"  hello  ".trimEnd();') == "  hello"

    def test_trim_tabs_newlines(self):
        assert run_val(r'"\t\nhello\n".trim();') == "hello"


class TestStringStartsEndsWith:
    def test_starts_with_true(self):
        assert run_val('"hello world".startsWith("hello");') == True

    def test_starts_with_false(self):
        assert run_val('"hello world".startsWith("world");') == False

    def test_ends_with_true(self):
        assert run_val('"hello world".endsWith("world");') == True

    def test_ends_with_false(self):
        assert run_val('"hello world".endsWith("hello");') == False


class TestStringIncludes:
    def test_includes_true(self):
        assert run_val('"hello world".includes("lo wo");') == True

    def test_includes_false(self):
        assert run_val('"hello world".includes("xyz");') == False

    def test_includes_empty(self):
        assert run_val('"hello".includes("");') == True


class TestStringIndexOf:
    def test_index_of_found(self):
        assert run_val('"hello".indexOf("ll");') == 2

    def test_index_of_not_found(self):
        assert run_val('"hello".indexOf("xyz");') == -1

    def test_index_of_with_start(self):
        assert run_val('"hello hello".indexOf("hello", 1);') == 6

    def test_last_index_of(self):
        assert run_val('"hello hello".lastIndexOf("hello");') == 6

    def test_last_index_of_with_end(self):
        assert run_val('"hello hello".lastIndexOf("hello", 4);') == 0


class TestStringReplace:
    def test_replace_first(self):
        assert run_val('"aaa".replace("a", "b");') == "baa"

    def test_replace_all(self):
        assert run_val('"aaa".replaceAll("a", "b");') == "bbb"

    def test_replace_no_match(self):
        assert run_val('"hello".replace("x", "y");') == "hello"


class TestStringCase:
    def test_to_upper(self):
        assert run_val('"hello".toUpperCase();') == "HELLO"

    def test_to_lower(self):
        assert run_val('"HELLO".toLowerCase();') == "hello"

    def test_mixed_case(self):
        assert run_val('"Hello World".toUpperCase();') == "HELLO WORLD"


class TestStringSlice:
    def test_slice_from(self):
        assert run_val('"hello".slice(1);') == "ello"

    def test_slice_range(self):
        assert run_val('"hello".slice(1, 3);') == "el"

    def test_slice_negative(self):
        assert run_val('"hello".slice(-3);') == "llo"

    def test_slice_negative_end(self):
        assert run_val('"hello".slice(1, -1);') == "ell"


class TestStringSubstring:
    def test_substring_basic(self):
        assert run_val('"hello".substring(1, 3);') == "el"

    def test_substring_swapped(self):
        # substring swaps args if start > end
        assert run_val('"hello".substring(3, 1);') == "el"

    def test_substring_from(self):
        assert run_val('"hello".substring(2);') == "llo"


class TestStringCharAt:
    def test_char_at(self):
        assert run_val('"hello".charAt(1);') == "e"

    def test_char_at_out_of_bounds(self):
        assert run_val('"hello".charAt(10);') == ""

    def test_char_code_at(self):
        assert run_val('"A".charCodeAt(0);') == 65

    def test_char_code_at_out_of_bounds(self):
        assert run_val('"".charCodeAt(0);') is None


class TestStringRepeat:
    def test_repeat(self):
        assert run_val('"ab".repeat(3);') == "ababab"

    def test_repeat_zero(self):
        assert run_val('"hello".repeat(0);') == ""


class TestStringPad:
    def test_pad_start(self):
        assert run_val('"5".padStart(3, "0");') == "005"

    def test_pad_start_default(self):
        assert run_val('"hi".padStart(5);') == "   hi"

    def test_pad_end(self):
        assert run_val('"5".padEnd(3, "0");') == "500"

    def test_pad_end_default(self):
        assert run_val('"hi".padEnd(5);') == "hi   "

    def test_pad_start_no_change(self):
        assert run_val('"hello".padStart(3, "0");') == "hello"

    def test_pad_start_multi_char(self):
        assert run_val('"1".padStart(5, "ab");') == "abab1"


class TestStringConcat:
    def test_concat(self):
        assert run_val('"hello".concat(" ", "world");') == "hello world"

    def test_concat_single(self):
        assert run_val('"a".concat("b");') == "ab"


class TestStringMatch:
    def test_match_found(self):
        assert run_val(r'"hello123".match("[0-9]+");') == "123"

    def test_match_not_found(self):
        assert run_val(r'"hello".match("[0-9]+");') is None

    def test_search_found(self):
        assert run_val(r'"hello123".search("[0-9]+");') == 5

    def test_search_not_found(self):
        assert run_val(r'"hello".search("[0-9]+");') == -1


class TestStringChaining:
    def test_chain_trim_split(self):
        assert run_val('"  a,b,c  ".trim().split(",");') == ["a", "b", "c"]

    def test_chain_upper_slice(self):
        assert run_val('"hello".toUpperCase().slice(0, 3);') == "HEL"

    def test_chain_replace_lower(self):
        assert run_val('"Hello World".replace("World", "Earth").toLowerCase();') == "hello earth"

    def test_method_on_literal(self):
        assert run_val('"HELLO".toLowerCase();') == "hello"


class TestStringMethodErrors:
    def test_invalid_method(self):
        with pytest.raises(VMError, match="no method"):
            run_val('"hello".foo();')

    def test_starts_with_wrong_type(self):
        with pytest.raises(VMError, match="must be string"):
            run_val('"hello".startsWith(5);')


# ============================================================
# Array Methods
# ============================================================

class TestArrayLength:
    def test_length(self):
        assert run_val('[1, 2, 3].length;') == 3

    def test_empty_length(self):
        assert run_val('[].length;') == 0


class TestArrayPushPop:
    def test_push(self):
        assert run_output('let a = [1, 2]; a.push(3); print a;') == "[1, 2, 3]"

    def test_push_returns_length(self):
        assert run_val('let a = [1]; a.push(2);') == 2

    def test_pop(self):
        assert run_val('let a = [1, 2, 3]; a.pop();') == 3

    def test_pop_mutates(self):
        assert run_output('let a = [1, 2, 3]; a.pop(); print a;') == "[1, 2]"

    def test_shift(self):
        assert run_val('let a = [1, 2, 3]; a.shift();') == 1

    def test_shift_mutates(self):
        assert run_output('let a = [1, 2, 3]; a.shift(); print a;') == "[2, 3]"

    def test_unshift(self):
        assert run_output('let a = [2, 3]; a.unshift(1); print a;') == "[1, 2, 3]"

    def test_unshift_returns_length(self):
        assert run_val('let a = [2, 3]; a.unshift(0, 1);') == 4


class TestArrayIndexOf:
    def test_index_of_found(self):
        assert run_val('[1, 2, 3, 2].indexOf(2);') == 1

    def test_index_of_not_found(self):
        assert run_val('[1, 2, 3].indexOf(5);') == -1

    def test_index_of_type_strict(self):
        # Should not match 1 == true due to type-aware comparison
        assert run_val('[1, 2, 3].indexOf(true);') == -1


class TestArrayIncludes:
    def test_includes_true(self):
        assert run_val('[1, 2, 3].includes(2);') == True

    def test_includes_false(self):
        assert run_val('[1, 2, 3].includes(5);') == False

    def test_includes_string(self):
        assert run_val('["a", "b"].includes("a");') == True


class TestArrayJoin:
    def test_join_default(self):
        assert run_val('[1, 2, 3].join(",");') == "1,2,3"

    def test_join_custom(self):
        assert run_val('["a", "b", "c"].join(" - ");') == "a - b - c"

    def test_join_empty(self):
        assert run_val('[1, 2, 3].join("");') == "123"


class TestArrayReverse:
    def test_reverse(self):
        assert run_output('let a = [1, 2, 3]; a.reverse(); print a;') == "[3, 2, 1]"

    def test_reverse_returns_array(self):
        assert run_val('[1, 2, 3].reverse();') == [3, 2, 1]


class TestArraySlice:
    def test_slice_from(self):
        assert run_val('[1, 2, 3, 4].slice(1);') == [2, 3, 4]

    def test_slice_range(self):
        assert run_val('[1, 2, 3, 4].slice(1, 3);') == [2, 3]

    def test_slice_negative(self):
        assert run_val('[1, 2, 3, 4].slice(-2);') == [3, 4]

    def test_slice_no_mutate(self):
        assert run_output('let a = [1, 2, 3]; let b = a.slice(1); print a;') == "[1, 2, 3]"


class TestArrayConcat:
    def test_concat_arrays(self):
        assert run_val('[1, 2].concat([3, 4]);') == [1, 2, 3, 4]

    def test_concat_values(self):
        assert run_val('[1].concat(2, 3);') == [1, 2, 3]

    def test_concat_mixed(self):
        assert run_val('[1].concat([2, 3], 4);') == [1, 2, 3, 4]


class TestArrayFlat:
    def test_flat_default(self):
        assert run_val('[[1, 2], [3, 4]].flat();') == [1, 2, 3, 4]

    def test_flat_nested(self):
        assert run_val('[1, [2, [3, [4]]]].flat(2);') == [1, 2, 3, [4]]

    def test_flat_deep(self):
        assert run_val('[1, [2, [3, [4]]]].flat(3);') == [1, 2, 3, 4]


class TestArrayFill:
    def test_fill_all(self):
        assert run_val('[1, 2, 3].fill(0);') == [0, 0, 0]

    def test_fill_range(self):
        assert run_val('[1, 2, 3, 4].fill(0, 1, 3);') == [1, 0, 0, 4]


class TestArrayMap:
    def test_map_basic(self):
        assert run_val('let a = [1, 2, 3]; a.map(fn(x) { return x * 2; });') == [2, 4, 6]

    def test_map_with_index(self):
        assert run_val('[10, 20, 30].map(fn(x, i) { return i; });') == [0, 1, 2]

    def test_map_strings(self):
        assert run_val('["a", "b"].map(fn(s) { return s.toUpperCase(); });') == ["A", "B"]


class TestArrayFilter:
    def test_filter_basic(self):
        assert run_val('[1, 2, 3, 4, 5].filter(fn(x) { return x > 3; });') == [4, 5]

    def test_filter_empty_result(self):
        assert run_val('[1, 2, 3].filter(fn(x) { return x > 10; });') == []

    def test_filter_all(self):
        assert run_val('[1, 2, 3].filter(fn(x) { return true; });') == [1, 2, 3]


class TestArrayReduce:
    def test_reduce_sum(self):
        assert run_val('[1, 2, 3, 4].reduce(fn(acc, x) { return acc + x; }, 0);') == 10

    def test_reduce_no_initial(self):
        assert run_val('[1, 2, 3].reduce(fn(acc, x) { return acc + x; });') == 6

    def test_reduce_strings(self):
        assert run_val('["a", "b", "c"].reduce(fn(acc, x) { return acc.concat(x); }, "");') == "abc"


class TestArrayForEach:
    def test_foreach_output(self):
        out = run_output('[1, 2, 3].forEach(fn(x) { print x; });')
        assert out == "1\n2\n3"

    def test_foreach_returns_null(self):
        assert run_val('[1].forEach(fn(x) { return x; });') is None


class TestArrayFind:
    def test_find_found(self):
        assert run_val('[1, 2, 3, 4].find(fn(x) { return x > 2; });') == 3

    def test_find_not_found(self):
        assert run_val('[1, 2, 3].find(fn(x) { return x > 10; });') is None


class TestArrayEvery:
    def test_every_true(self):
        assert run_val('[2, 4, 6].every(fn(x) { return x > 0; });') == True

    def test_every_false(self):
        assert run_val('[2, 4, -1].every(fn(x) { return x > 0; });') == False


class TestArraySome:
    def test_some_true(self):
        assert run_val('[1, -1, 3].some(fn(x) { return x < 0; });') == True

    def test_some_false(self):
        assert run_val('[1, 2, 3].some(fn(x) { return x < 0; });') == False


class TestArraySort:
    def test_sort_default(self):
        r = run_val('[3, 1, 2].sort();')
        assert r == [1, 2, 3]

    def test_sort_with_comparator(self):
        # Sort descending
        assert run_val('[1, 3, 2].sort(fn(a, b) { return b - a; });') == [3, 2, 1]

    def test_sort_mutates(self):
        assert run_output('let a = [3, 1, 2]; a.sort(); print a;') == "[1, 2, 3]"


class TestArrayChaining:
    def test_filter_map(self):
        assert run_val('[1, 2, 3, 4, 5].filter(fn(x) { return x > 2; }).map(fn(x) { return x * 10; });') == [30, 40, 50]

    def test_map_join(self):
        assert run_val('[1, 2, 3].map(fn(x) { return x * 2; }).join(", ");') == "2, 4, 6"

    def test_slice_reverse(self):
        assert run_val('[1, 2, 3, 4].slice(1, 3).reverse();') == [3, 2]


# ============================================================
# Hash Methods
# ============================================================

class TestHashKeys:
    def test_keys(self):
        r = run_val('let h = {a: 1, b: 2}; h.keys();')
        assert sorted(r) == ["a", "b"]

    def test_values(self):
        r = run_val('let h = {a: 1, b: 2}; h.values();')
        assert sorted(r) == [1, 2]

    def test_entries(self):
        r = run_val('let h = {x: 10}; h.entries();')
        assert r == [["x", 10]]


class TestHashHas:
    def test_has_true(self):
        assert run_val('let h = {a: 1}; h.has("a");') == True

    def test_has_false(self):
        assert run_val('let h = {a: 1}; h.has("b");') == False


class TestHashDelete:
    def test_delete_existing(self):
        assert run_val('let h = {a: 1, b: 2}; h.delete("a");') == True

    def test_delete_mutates(self):
        assert run_val('let h = {a: 1, b: 2}; h.delete("a"); h.has("a");') == False

    def test_delete_missing(self):
        assert run_val('let h = {a: 1}; h.delete("b");') == False


class TestHashSize:
    def test_size(self):
        assert run_val('{a: 1, b: 2, c: 3}.size;') == 3

    def test_empty_size(self):
        assert run_val('{}.size;') == 0


class TestHashMerge:
    def test_merge(self):
        r = run_val('let h = {a: 1}; h.merge({b: 2});')
        assert r == {"a": 1, "b": 2}

    def test_merge_override(self):
        r = run_val('let h = {a: 1, b: 2}; h.merge({b: 3});')
        assert r == {"a": 1, "b": 3}

    def test_merge_no_mutate(self):
        assert run_val('let h = {a: 1}; h.merge({b: 2}); h.has("b");') == False


class TestHashIsEmpty:
    def test_empty(self):
        assert run_val('{}.isEmpty();') == True

    def test_not_empty(self):
        assert run_val('{a: 1}.isEmpty();') == False


class TestHashPropertyPrecedence:
    def test_property_over_method(self):
        # If a hash has a key named "keys", property wins
        assert run_val('let h = {keys: 42}; h.keys;') == 42

    def test_method_when_no_property(self):
        r = run_val('let h = {a: 1}; h.keys();')
        assert r == ["a"]


# ============================================================
# Integration: Methods with other language features
# ============================================================

class TestMethodsWithClosures:
    def test_map_with_closure(self):
        assert run_val("""
            let factor = 10;
            [1, 2, 3].map(fn(x) { return x * factor; });
        """) == [10, 20, 30]

    def test_filter_with_closure(self):
        assert run_val("""
            let threshold = 3;
            [1, 2, 3, 4, 5].filter(fn(x) { return x > threshold; });
        """) == [4, 5]


class TestMethodsWithDestructuring:
    def test_destructure_split_result(self):
        assert run_output("""
            let [first, ...rest] = "a,b,c".split(",");
            print first;
        """) == "a"

    def test_destructure_entries(self):
        assert run_output("""
            let h = {x: 10};
            let entries = h.entries();
            let [k, v] = entries[0];
            print k;
            print v;
        """) == "x\n10"


class TestMethodsWithPipe:
    def test_string_pipe(self):
        assert run_val('"  HELLO  " |> fn(s) { return s.trim(); } |> fn(s) { return s.toLowerCase(); };') == "hello"


class TestMethodsWithOptionalChaining:
    def test_optional_property_on_null(self):
        assert run_val('let s = null; s?.length;') is None

    def test_optional_not_null(self):
        assert run_val('let s = "hello"; s.toUpperCase();') == "HELLO"

    def test_optional_length_not_null(self):
        assert run_val('let s = "hello"; s?.length;') == 5


class TestMethodsWithForIn:
    def test_for_in_keys(self):
        assert run_output("""
            let h = {a: 1, b: 2};
            let k = h.keys();
            k.sort();
            for (x in k) {
                print x;
            }
        """) == "a\nb"


class TestMethodsWithClasses:
    def test_class_method_returns_string(self):
        assert run_val("""
            class Greeter {
                init(name) {
                    this.name = name;
                }
                greet() {
                    return "Hello, ".concat(this.name, "!");
                }
            }
            let g = Greeter("world");
            g.greet();
        """) == "Hello, world!"


class TestMethodsWithSpread:
    def test_spread_into_push(self):
        assert run_output("""
            let a = [1, 2];
            let b = [3, 4];
            a.push(...b);
            print a;
        """) == "[1, 2, 3, 4]"


class TestMethodsWithErrorHandling:
    def test_try_catch_method_error(self):
        assert run_output("""
            try {
                [].pop();
            } catch (e) {
                print "caught";
            }
        """) == "caught"


class TestMethodsWithStringInterpolation:
    def test_interpolation_with_method(self):
        assert run_output("""
            let name = "world";
            print f"Hello, ${name.toUpperCase()}!";
        """) == "Hello, WORLD!"


class TestMethodsWithAsync:
    def test_async_with_methods(self):
        assert run_val("""
            let items = [1, 2, 3];
            items.map(fn(x) { return x * 2; }).join(", ");
        """) == "2, 4, 6"


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_method_on_string_literal(self):
        assert run_val('"hello world".split(" ").length;') == 2

    def test_chained_array_methods(self):
        assert run_val('[1, 2, 3, 4, 5].filter(fn(x) { return x > 2; }).map(fn(x) { return x * x; }).reduce(fn(a, b) { return a + b; }, 0);') == 50

    def test_empty_array_methods(self):
        assert run_val('[].map(fn(x) { return x; });') == []
        assert run_val('[].filter(fn(x) { return true; });') == []
        assert run_val('[].every(fn(x) { return false; });') == True
        assert run_val('[].some(fn(x) { return true; });') == False

    def test_hash_empty_methods(self):
        assert run_val('{}.keys();') == []
        assert run_val('{}.values();') == []
        assert run_val('{}.entries();') == []

    def test_string_method_no_args(self):
        assert run_val('"  hello  ".trim();') == "hello"

    def test_array_method_error(self):
        with pytest.raises(VMError, match="no method"):
            run_val('[1, 2].foo();')

    def test_method_on_variable(self):
        assert run_val('let s = "hello"; let parts = s.split("l"); parts;') == ["he", "", "o"]
