"""Tests for C059: Computed Properties in Hash Literals.

Extends C058 with {[expr]: value} syntax for computed hash keys.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from computed_properties import run, execute, ModuleRegistry

# ── Helper ──────────────────────────────────────────────────────────

def val(src):
    """Run source, return result."""
    result, output = run(src)
    return result

def out(src):
    """Run source, return printed output."""
    result, output = run(src)
    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════
# BASIC COMPUTED PROPERTIES
# ═══════════════════════════════════════════════════════════════════

class TestBasicComputed:
    def test_string_variable_as_key(self):
        assert out('let k = "name"; let h = {[k]: "alice"}; print h.name;') == "alice"

    def test_string_literal_as_key(self):
        assert out('let h = {["greeting"]: "hello"}; print h.greeting;') == "hello"

    def test_integer_variable_as_key(self):
        assert out('let k = 42; let h = {[k]: "answer"}; print h[42];') == "answer"

    def test_expression_as_key(self):
        assert out('let h = {[1 + 2]: "three"}; print h[3];') == "three"

    def test_string_concat_as_key(self):
        assert out('let prefix = "user"; let h = {[prefix + "_name"]: "bob"}; print h.user_name;') == "bob"

    def test_function_call_as_key(self):
        src = '''
        fn getKey() { return "dynamic"; }
        let h = {[getKey()]: "value"};
        print h.dynamic;
        '''
        assert out(src) == "value"

    def test_ternary_as_key(self):
        src = '''
        let flag = true;
        let h = {[if (flag) "yes" else "no"]: 1};
        print h.yes;
        '''
        assert out(src) == "1"

    def test_method_call_as_key(self):
        src = '''
        let arr = ["a", "b", "c"];
        let h = {[arr.join("-")]: "joined"};
        print h["a-b-c"];
        '''
        assert out(src) == "joined"


# ═══════════════════════════════════════════════════════════════════
# MIXED KEYS (computed + static)
# ═══════════════════════════════════════════════════════════════════

class TestMixedKeys:
    def test_computed_and_static(self):
        src = '''
        let k = "b";
        let h = {a: 1, [k]: 2, c: 3};
        print h.a;
        print h.b;
        print h.c;
        '''
        assert out(src) == "1\n2\n3"

    def test_computed_first(self):
        assert out('let h = {["x"]: 10, y: 20}; print h.x; print h.y;') == "10\n20"

    def test_computed_last(self):
        assert out('let h = {x: 10, ["y"]: 20}; print h.x; print h.y;') == "10\n20"

    def test_multiple_computed(self):
        src = '''
        let a = "x";
        let b = "y";
        let h = {[a]: 1, [b]: 2};
        print h.x;
        print h.y;
        '''
        assert out(src) == "1\n2"

    def test_all_computed(self):
        src = '''
        let h = {["a"]: 1, ["b"]: 2, ["c"]: 3};
        print h.a;
        print h.b;
        print h.c;
        '''
        assert out(src) == "1\n2\n3"

    def test_computed_with_int_key(self):
        src = '''
        let h = {name: "test", [0]: "zero", [1]: "one"};
        print h.name;
        print h[0];
        print h[1];
        '''
        assert out(src) == "test\nzero\none"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED + SPREAD
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithSpread:
    def test_computed_then_spread(self):
        src = '''
        let base = {x: 1};
        let h = {["y"]: 2, ...base};
        print h.x;
        print h.y;
        '''
        assert out(src) == "1\n2"

    def test_spread_then_computed(self):
        src = '''
        let base = {x: 1};
        let h = {...base, ["y"]: 2};
        print h.x;
        print h.y;
        '''
        assert out(src) == "1\n2"

    def test_computed_overrides_spread(self):
        src = '''
        let base = {name: "old"};
        let h = {...base, ["name"]: "new"};
        print h.name;
        '''
        assert out(src) == "new"

    def test_spread_overrides_computed(self):
        src = '''
        let base = {name: "newer"};
        let h = {["name"]: "old", ...base};
        print h.name;
        '''
        assert out(src) == "newer"


# ═══════════════════════════════════════════════════════════════════
# DYNAMIC KEY PATTERNS
# ═══════════════════════════════════════════════════════════════════

class TestDynamicPatterns:
    def test_loop_building_hash(self):
        src = '''
        let h = {};
        let keys = ["a", "b", "c"];
        let i = 0;
        while (i < keys.length) {
            h[keys[i]] = i;
            i = i + 1;
        }
        print h.a;
        print h.b;
        print h.c;
        '''
        assert out(src) == "0\n1\n2"

    def test_computed_from_loop_variable(self):
        src = '''
        fn makeHash(keys) {
            let result = {};
            for (k in keys) {
                result[k] = k.length;
            }
            return result;
        }
        let h = makeHash(["hi", "hello", "hey"]);
        print h.hi;
        print h.hello;
        print h.hey;
        '''
        assert out(src) == "2\n5\n3"

    def test_symbol_like_key(self):
        src = '''
        let TYPE = "__type__";
        let h = {[TYPE]: "config", value: 42};
        print h.__type__;
        print h.value;
        '''
        assert out(src) == "config\n42"

    def test_nested_computed(self):
        src = '''
        let k1 = "outer";
        let k2 = "inner";
        let h = {[k1]: {[k2]: "deep"}};
        print h.outer.inner;
        '''
        assert out(src) == "deep"

    def test_computed_with_null_coalescing(self):
        src = '''
        let k = null ?? "fallback";
        let h = {[k]: true};
        print h.fallback;
        '''
        assert out(src) == "true"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH DESTRUCTURING
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithDestructuring:
    def test_computed_hash_then_destructure(self):
        src = '''
        let k = "name";
        let h = {[k]: "alice", age: 30};
        let {name, age} = h;
        print name;
        print age;
        '''
        assert out(src) == "alice\n30"

    def test_computed_key_from_destructured(self):
        src = '''
        let {x, y} = {x: "key1", y: "key2"};
        let h = {[x]: 10, [y]: 20};
        print h.key1;
        print h.key2;
        '''
        assert out(src) == "10\n20"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH CLASSES
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithClasses:
    def test_class_instance_property_as_key(self):
        src = '''
        class Config {
            init(key) { this.key = key; }
        }
        let cfg = Config("setting");
        let h = {[cfg.key]: "value"};
        print h.setting;
        '''
        assert out(src) == "value"

    def test_computed_in_method_return(self):
        src = '''
        class Builder {
            build(key, val) {
                return {[key]: val};
            }
        }
        let b = Builder();
        let h = b.build("name", "test");
        print h.name;
        '''
        assert out(src) == "test"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH ASYNC
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithAsync:
    def test_computed_in_async_fn(self):
        src = '''
        async fn makeHash(key) {
            return {[key]: "async_value"};
        }
        let p = makeHash("result");
        let h = await p;
        print h.result;
        '''
        assert out(src) == "async_value"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH GENERATORS
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithGenerators:
    def test_generator_yielding_computed_hash(self):
        src = '''
        fn* gen() {
            yield {["a"]: 1};
            yield {["b"]: 2};
        }
        let g = gen();
        let h1 = next(g);
        let h2 = next(g);
        print h1.a;
        print h2.b;
        '''
        assert out(src) == "1\n2"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH PIPE OPERATOR
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithPipe:
    def test_piped_value_as_key(self):
        src = '''
        fn upper(s) { return s.toUpperCase(); }
        let key = "hello" |> upper;
        let h = {[key]: "world"};
        print h.HELLO;
        '''
        assert out(src) == "world"


# ═══════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_string_key(self):
        assert out('let h = {[""]: "empty"}; print h[""];') == "empty"

    def test_boolean_key(self):
        assert out('let h = {[true]: "yes"}; print h[true];') == "yes"

    def test_computed_overwrites_earlier(self):
        src = '''
        let h = {["x"]: 1, ["x"]: 2};
        print h.x;
        '''
        assert out(src) == "2"

    def test_static_and_computed_same_key(self):
        src = '''
        let h = {x: 1, ["x"]: 2};
        print h.x;
        '''
        assert out(src) == "2"

    def test_computed_key_evaluation_order(self):
        src = '''
        let h = {["first"]: "a", ["second"]: "b"};
        print h.first;
        print h.second;
        '''
        assert out(src) == "a\nb"

    def test_trailing_comma_after_computed(self):
        assert out('let h = {["x"]: 1,}; print h.x;') == "1"

    def test_computed_only_hash(self):
        assert out('let k = "solo"; let h = {[k]: 42}; print h.solo;') == "42"

    def test_complex_expression_key(self):
        src = '''
        let items = ["foo", "bar"];
        let h = {[items[0] + "_" + items[1]]: true};
        print h.foo_bar;
        '''
        assert out(src) == "true"

    def test_computed_with_optional_chaining(self):
        src = '''
        let obj = {key: "dynamic"};
        let h = {[obj?.key]: "found"};
        print h.dynamic;
        '''
        assert out(src) == "found"

    def test_null_coalescing_key(self):
        src = '''
        let k = null;
        let h = {[k ?? "default"]: "val"};
        print h.default;
        '''
        assert out(src) == "val"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH STRING INTERPOLATION
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithInterpolation:
    def test_interpolated_key(self):
        src = '''
        let n = 5;
        let h = {[f"item_${n}"]: "five"};
        print h.item_5;
        '''
        assert out(src) == "five"

    def test_multiple_interpolated_keys(self):
        src = '''
        let h = {[f"k_${1}"]: "a", [f"k_${2}"]: "b"};
        print h.k_1;
        print h.k_2;
        '''
        assert out(src) == "a\nb"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH FOR-IN
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithForIn:
    def test_iterate_computed_hash(self):
        src = '''
        let h = {["a"]: 1, ["b"]: 2};
        let keys = [];
        for (k, v in h) {
            keys.push(k);
        }
        print keys.length;
        '''
        assert out(src) == "2"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH TRY/CATCH
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithErrorHandling:
    def test_computed_in_try(self):
        src = '''
        try {
            let h = {["key"]: 42};
            print h.key;
        } catch (e) {
            print "error";
        }
        '''
        assert out(src) == "42"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH CLOSURES
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithClosures:
    def test_closure_computed_key(self):
        src = '''
        fn maker(prefix) {
            return fn(name) {
                return {[prefix + "_" + name]: true};
            };
        }
        let m = maker("user");
        let h = m("alice");
        print h.user_alice;
        '''
        assert out(src) == "true"

    def test_computed_captures_variable(self):
        src = '''
        let keys = [];
        for (i in [0, 1, 2]) {
            keys.push(f"k${i}");
        }
        let h = {[keys[0]]: "a", [keys[1]]: "b", [keys[2]]: "c"};
        print h.k0;
        print h.k1;
        print h.k2;
        '''
        assert out(src) == "a\nb\nc"


# ═══════════════════════════════════════════════════════════════════
# REGRESSION: existing hash features still work
# ═══════════════════════════════════════════════════════════════════

class TestRegression:
    def test_static_keys_still_work(self):
        assert out('let h = {name: "test", age: 30}; print h.name; print h.age;') == "test\n30"

    def test_string_keys_still_work(self):
        assert out('let h = {"key": "val"}; print h.key;') == "val"

    def test_int_keys_still_work(self):
        assert out('let h = {1: "one", 2: "two"}; print h[1]; print h[2];') == "one\ntwo"

    def test_empty_hash_still_works(self):
        assert out('let h = {}; print type(h);') == "hash"

    def test_spread_still_works(self):
        src = '''
        let a = {x: 1};
        let b = {...a, y: 2};
        print b.x;
        print b.y;
        '''
        assert out(src) == "1\n2"

    def test_dot_access_still_works(self):
        assert out('let h = {name: "test"}; print h.name;') == "test"

    def test_bracket_access_still_works(self):
        assert out('let h = {name: "test"}; print h["name"];') == "test"

    def test_nested_hash_still_works(self):
        assert out('let h = {a: {b: "deep"}}; print h.a.b;') == "deep"

    def test_hash_assignment_still_works(self):
        assert out('let h = {x: 1}; h.x = 2; print h.x;') == "2"

    def test_trailing_comma_still_works(self):
        assert out('let h = {a: 1, b: 2,}; print h.a; print h.b;') == "1\n2"


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH MODULES
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithModules:
    def test_computed_in_module(self):
        src = '''
        import { KEY } from "config";
        let h = {[KEY]: "module_val"};
        print h.setting;
        '''
        reg = ModuleRegistry()
        reg.register("config", 'export let KEY = "setting";')
        result = execute(src, registry=reg)
        assert result["output"] == ["module_val"]


# ═══════════════════════════════════════════════════════════════════
# COMPUTED WITH FINALLY
# ═══════════════════════════════════════════════════════════════════

class TestComputedWithFinally:
    def test_computed_in_finally(self):
        src = '''
        let result = null;
        try {
            throw "err";
        } catch (e) {
            result = "caught";
        } finally {
            let h = {["status"]: result};
            print h.status;
        }
        '''
        assert out(src) == "caught"
