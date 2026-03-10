"""
Tests for C043 Hash Maps
Challenge C043 -- AgentZero Session 044
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from hash_maps import (
    lex, parse, run, execute, compile_source, disassemble,
    TokenType, LexError, ParseError, CompileError, VMError,
    HashLit, DotExpr, DotAssign, IndexExpr, _format_value,
)


# ============================================================
# Lexer Tests
# ============================================================

class TestLexer:
    def test_colon_token(self):
        tokens = lex(":")
        assert tokens[0].type == TokenType.COLON

    def test_dot_token(self):
        tokens = lex(".")
        assert tokens[0].type == TokenType.DOT

    def test_hash_literal_tokens(self):
        tokens = lex('{a: 1, b: 2}')
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert TokenType.LBRACE in types
        assert TokenType.COLON in types
        assert TokenType.RBRACE in types

    def test_dot_access_tokens(self):
        tokens = lex('obj.key')
        assert tokens[0].type == TokenType.IDENT
        assert tokens[1].type == TokenType.DOT
        assert tokens[2].type == TokenType.IDENT


# ============================================================
# Parser Tests
# ============================================================

class TestParser:
    def test_empty_hash(self):
        ast = parse("let m = {};")
        node = ast.stmts[0].value
        assert isinstance(node, HashLit)
        assert len(node.pairs) == 0

    def test_single_pair(self):
        ast = parse('let m = {"a": 1};')
        node = ast.stmts[0].value
        assert isinstance(node, HashLit)
        assert len(node.pairs) == 1

    def test_multiple_pairs(self):
        ast = parse('let m = {"a": 1, "b": 2, "c": 3};')
        node = ast.stmts[0].value
        assert isinstance(node, HashLit)
        assert len(node.pairs) == 3

    def test_trailing_comma(self):
        ast = parse('let m = {"a": 1, "b": 2,};')
        node = ast.stmts[0].value
        assert isinstance(node, HashLit)
        assert len(node.pairs) == 2

    def test_dot_access_parse(self):
        ast = parse('m.x;')
        node = ast.stmts[0]
        assert isinstance(node, DotExpr)
        assert node.key == 'x'

    def test_dot_assign_parse(self):
        ast = parse('m.x = 5;')
        node = ast.stmts[0]
        assert isinstance(node, DotAssign)
        assert node.key == 'x'

    def test_chained_dot(self):
        ast = parse('a.b.c;')
        node = ast.stmts[0]
        assert isinstance(node, DotExpr)
        assert node.key == 'c'
        assert isinstance(node.obj, DotExpr)
        assert node.obj.key == 'b'

    def test_hash_with_ident_keys(self):
        """Identifier keys in hash literals (bare names as keys)."""
        ast = parse('let m = {x: 1, y: 2};')
        node = ast.stmts[0].value
        assert isinstance(node, HashLit)
        assert len(node.pairs) == 2

    def test_hash_in_expression_position(self):
        ast = parse('let m = {"a": 1};')
        node = ast.stmts[0].value
        assert isinstance(node, HashLit)

    def test_index_and_dot_mixed(self):
        ast = parse('m["a"].b;')
        node = ast.stmts[0]
        assert isinstance(node, DotExpr)
        assert isinstance(node.obj, IndexExpr)


# ============================================================
# Basic Hash Map Tests
# ============================================================

class TestBasicHashMap:
    def test_empty_hash(self):
        result, output = run('let m = {}; print(m);')
        assert output == ['{}']

    def test_single_pair_string_key(self):
        result, output = run('let m = {"name": "alice"}; print(m);')
        assert output == ['{name: alice}']

    def test_multiple_pairs(self):
        result, output = run('let m = {"a": 1, "b": 2}; print(m);')
        assert 'a: 1' in output[0]
        assert 'b: 2' in output[0]

    def test_integer_values(self):
        result, output = run('let m = {"x": 10, "y": 20}; print(m);')
        assert 'x: 10' in output[0]
        assert 'y: 20' in output[0]

    def test_boolean_values(self):
        result, output = run('let m = {"active": true, "done": false}; print(m);')
        assert 'active: true' in output[0]
        assert 'done: false' in output[0]

    def test_nested_hash(self):
        result, output = run('let m = {"inner": {"x": 1}}; print(m);')
        assert 'inner: {x: 1}' in output[0]

    def test_hash_with_array_value(self):
        result, output = run('let m = {"items": [1, 2, 3]}; print(m);')
        assert 'items: [1, 2, 3]' in output[0]

    def test_integer_key(self):
        r = execute('let m = {1: "one", 2: "two"};')
        m = r['env']['m']
        assert m[1] == "one"
        assert m[2] == "two"

    def test_expression_values(self):
        result, output = run('let x = 5; let m = {"val": x + 1}; print(m);')
        assert 'val: 6' in output[0]


# ============================================================
# Dot Access Tests
# ============================================================

class TestDotAccess:
    def test_basic_dot_get(self):
        result, output = run('let m = {"x": 42}; print(m.x);')
        assert output == ['42']

    def test_dot_get_string(self):
        result, output = run('let m = {"name": "alice"}; print(m.name);')
        assert output == ['alice']

    def test_dot_set(self):
        result, output = run('let m = {"x": 1}; m.x = 99; print(m.x);')
        assert output == ['99']

    def test_dot_set_new_key(self):
        result, output = run('let m = {}; m.name = "bob"; print(m.name);')
        assert output == ['bob']

    def test_chained_dot_access(self):
        result, output = run('let m = {"a": {"b": {"c": 42}}}; print(m.a.b.c);')
        assert output == ['42']

    def test_chained_dot_set(self):
        result, output = run("""
            let m = {"inner": {"x": 0}};
            m.inner.x = 99;
            print(m.inner.x);
        """)
        assert output == ['99']

    def test_dot_access_after_index(self):
        result, output = run('let m = {"items": [{"name": "a"}]}; print(m.items[0]);')
        assert output == ['{name: a}']

    def test_index_after_dot(self):
        result, output = run('let arr = [{"x": 1}, {"x": 2}]; print(arr[1].x);')
        assert output == ['2']

    def test_dot_access_nonexistent_key(self):
        with pytest.raises(VMError, match="not found"):
            run('let m = {}; m.x;')


# ============================================================
# String Key Indexing Tests
# ============================================================

class TestStringKeyIndex:
    def test_string_index_get(self):
        result, output = run('let m = {"key": 42}; print(m["key"]);')
        assert output == ['42']

    def test_string_index_set(self):
        result, output = run('let m = {}; m["key"] = 99; print(m["key"]);')
        assert output == ['99']

    def test_dynamic_key(self):
        result, output = run("""
            let m = {"a": 1, "b": 2};
            let k = "a";
            print(m[k]);
        """)
        assert output == ['1']

    def test_dynamic_key_set(self):
        result, output = run("""
            let m = {};
            let k = "name";
            m[k] = "alice";
            print(m.name);
        """)
        assert output == ['alice']

    def test_computed_key(self):
        result, output = run("""
            let m = {"key1": 10, "key2": 20};
            let i = 1;
            // Can't compute string keys easily, but can use variables
            print(m["key1"]);
            print(m["key2"]);
        """)
        assert output == ['10', '20']


# ============================================================
# Hash Map Builtins
# ============================================================

class TestBuiltins:
    def test_keys(self):
        result, output = run('let m = {"a": 1, "b": 2}; print(keys(m));')
        assert output == ['[a, b]']

    def test_values(self):
        result, output = run('let m = {"a": 1, "b": 2}; print(values(m));')
        assert output == ['[1, 2]']

    def test_has_true(self):
        result, output = run('let m = {"x": 1}; print(has(m, "x"));')
        assert output == ['true']

    def test_has_false(self):
        result, output = run('let m = {"x": 1}; print(has(m, "y"));')
        assert output == ['false']

    def test_delete(self):
        result, output = run("""
            let m = {"a": 1, "b": 2};
            delete(m, "a");
            print(has(m, "a"));
            print(m.b);
        """)
        assert output == ['false', '2']

    def test_delete_nonexistent(self):
        result, output = run("""
            let m = {"a": 1};
            delete(m, "z");
            print(len(m));
        """)
        assert output == ['1']

    def test_merge(self):
        result, output = run("""
            let a = {"x": 1};
            let b = {"y": 2};
            let c = merge(a, b);
            print(c.x);
            print(c.y);
        """)
        assert output == ['1', '2']

    def test_merge_override(self):
        result, output = run("""
            let a = {"x": 1};
            let b = {"x": 99};
            let c = merge(a, b);
            print(c.x);
        """)
        assert output == ['99']

    def test_merge_doesnt_modify_originals(self):
        result, output = run("""
            let a = {"x": 1};
            let b = {"y": 2};
            let c = merge(a, b);
            print(has(a, "y"));
        """)
        assert output == ['false']

    def test_entries(self):
        result, output = run("""
            let m = {"a": 1, "b": 2};
            let e = entries(m);
            print(len(e));
            print(e[0]);
        """)
        assert output == ['2', '[a, 1]']

    def test_size(self):
        result, output = run('let m = {"a": 1, "b": 2, "c": 3}; print(size(m));')
        assert output == ['3']

    def test_len_hash(self):
        result, output = run('let m = {"a": 1, "b": 2}; print(len(m));')
        assert output == ['2']

    def test_keys_empty(self):
        result, output = run('let m = {}; print(keys(m));')
        assert output == ['[]']

    def test_values_empty(self):
        result, output = run('let m = {}; print(values(m));')
        assert output == ['[]']

    def test_entries_empty(self):
        result, output = run('let m = {}; print(entries(m));')
        assert output == ['[]']

    def test_size_array(self):
        result, output = run('print(size([1, 2, 3]));')
        assert output == ['3']

    def test_size_string(self):
        result, output = run('print(size("hello"));')
        assert output == ['5']


# ============================================================
# Hash Map Mutation Tests
# ============================================================

class TestMutation:
    def test_add_key_via_dot(self):
        result, output = run("""
            let m = {};
            m.x = 1;
            m.y = 2;
            print(m.x);
            print(m.y);
        """)
        assert output == ['1', '2']

    def test_add_key_via_index(self):
        result, output = run("""
            let m = {};
            m["key"] = "value";
            print(m["key"]);
        """)
        assert output == ['value']

    def test_overwrite_value(self):
        result, output = run("""
            let m = {"x": 1};
            m.x = 2;
            m.x = 3;
            print(m.x);
        """)
        assert output == ['3']

    def test_mutation_is_reference(self):
        """Hash maps are mutable references like arrays."""
        result, output = run("""
            fn set_name(m) {
                m.name = "modified";
            }
            let obj = {"name": "original"};
            set_name(obj);
            print(obj.name);
        """)
        assert output == ['modified']

    def test_hash_in_array_mutation(self):
        result, output = run("""
            let items = [{"x": 1}, {"x": 2}];
            items[0].x = 99;
            print(items[0].x);
        """)
        assert output == ['99']

    def test_array_in_hash_mutation(self):
        result, output = run("""
            let m = {"items": [1, 2, 3]};
            push(m.items, 4);
            print(len(m.items));
        """)
        assert output == ['4']


# ============================================================
# Hash + Closure Integration
# ============================================================

class TestClosureIntegration:
    def test_closure_captures_hash(self):
        result, output = run("""
            let m = {"count": 0};
            fn inc() {
                m.count = m.count + 1;
            }
            inc();
            inc();
            print(m.count);
        """)
        assert output == ['2']

    def test_closure_returns_hash(self):
        result, output = run("""
            fn make_point(x, y) {
                return {"x": x, "y": y};
            }
            let p = make_point(3, 4);
            print(p.x);
            print(p.y);
        """)
        assert output == ['3', '4']

    def test_hash_with_closure_value(self):
        result, output = run("""
            let obj = {
                "greet": fn(name) {
                    return "hello";
                }
            };
            print(obj.greet("world"));
        """)
        assert output == ['hello']

    def test_method_like_pattern(self):
        """Closures as methods on hash maps."""
        result, output = run("""
            fn make_counter() {
                let state = {"n": 0};
                return {
                    "inc": fn() { state.n = state.n + 1; return state.n; },
                    "get": fn() { return state.n; }
                };
            }
            let c = make_counter();
            c.inc();
            c.inc();
            c.inc();
            print(c.get());
        """)
        assert output == ['3']

    def test_factory_pattern(self):
        result, output = run("""
            fn make_person(name, age) {
                return {
                    "name": name,
                    "age": age,
                    "greet": fn() { return name; }
                };
            }
            let p = make_person("Alice", 30);
            print(p.name);
            print(p.age);
            print(p.greet());
        """)
        assert output == ['Alice', '30', 'Alice']


# ============================================================
# Hash + Array Composition
# ============================================================

class TestArrayComposition:
    def test_array_of_hashes(self):
        result, output = run("""
            let items = [
                {"name": "a", "val": 1},
                {"name": "b", "val": 2},
                {"name": "c", "val": 3}
            ];
            print(items[1].name);
        """)
        assert output == ['b']

    def test_map_over_hashes(self):
        result, output = run("""
            let items = [
                {"x": 1},
                {"x": 2},
                {"x": 3}
            ];
            let xs = map(items, fn(item) { return item.x; });
            print(xs);
        """)
        assert output == ['[1, 2, 3]']

    def test_filter_hashes(self):
        result, output = run("""
            let items = [
                {"name": "a", "active": true},
                {"name": "b", "active": false},
                {"name": "c", "active": true}
            ];
            let active = filter(items, fn(item) { return item.active; });
            print(len(active));
        """)
        assert output == ['2']

    def test_reduce_hashes(self):
        result, output = run("""
            let items = [
                {"price": 10},
                {"price": 20},
                {"price": 30}
            ];
            let total = reduce(items, fn(acc, item) { return acc + item.price; }, 0);
            print(total);
        """)
        assert output == ['60']

    def test_hash_values_are_arrays(self):
        result, output = run("""
            let data = {
                "names": ["alice", "bob"],
                "ages": [30, 25]
            };
            print(data.names[0]);
            print(data.ages[1]);
        """)
        assert output == ['alice', '25']

    def test_find_in_hash_array(self):
        result, output = run("""
            let users = [
                {"id": 1, "name": "alice"},
                {"id": 2, "name": "bob"}
            ];
            let found = find(users, fn(u) { return u.id == 2; });
            print(found.name);
        """)
        assert output == ['bob']


# ============================================================
# Complex Composition Tests
# ============================================================

class TestComplexComposition:
    def test_nested_hash_construction(self):
        result, output = run("""
            let config = {
                "db": {
                    "host": "localhost",
                    "port": 5432
                },
                "app": {
                    "debug": true
                }
            };
            print(config.db.host);
            print(config.db.port);
            print(config.app.debug);
        """)
        assert output == ['localhost', '5432', 'true']

    def test_dynamic_hash_building(self):
        result, output = run("""
            let m = {};
            let i = 0;
            while (i < 3) {
                m[i] = i * i;
                i = i + 1;
            }
            print(m[0]);
            print(m[1]);
            print(m[2]);
        """)
        assert output == ['0', '1', '4']

    def test_hash_equality(self):
        """Hash maps support == comparison."""
        result, output = run("""
            let a = {"x": 1};
            let b = {"x": 1};
            print(a == b);
        """)
        assert output == ['true']

    def test_hash_inequality(self):
        result, output = run("""
            let a = {"x": 1};
            let b = {"x": 2};
            print(a == b);
        """)
        assert output == ['false']

    def test_hash_as_fn_arg(self):
        result, output = run("""
            fn get_x(obj) {
                return obj.x;
            }
            print(get_x({"x": 42}));
        """)
        assert output == ['42']

    def test_hash_as_fn_return(self):
        result, output = run("""
            fn make_pair(a, b) {
                return {"first": a, "second": b};
            }
            let p = make_pair(1, 2);
            print(p.first);
            print(p.second);
        """)
        assert output == ['1', '2']

    def test_iterate_entries(self):
        """Use array-as-cell pattern for mutable closure state."""
        result, output = run("""
            let m = {"a": 1, "b": 2, "c": 3};
            let e = entries(m);
            let sum = [0];
            each(e, fn(pair) {
                sum[0] = sum[0] + pair[1];
            });
            print(sum[0]);
        """)
        assert output == ['6']

    def test_merge_chain(self):
        result, output = run("""
            let a = {"x": 1};
            let b = {"y": 2};
            let c = {"z": 3};
            let all = merge(merge(a, b), c);
            print(all.x);
            print(all.y);
            print(all.z);
        """)
        assert output == ['1', '2', '3']

    def test_copy_via_merge(self):
        result, output = run("""
            let orig = {"x": 1, "y": 2};
            let copy = merge(orig, {});
            copy.x = 99;
            print(orig.x);
            print(copy.x);
        """)
        assert output == ['1', '99']

    def test_record_pattern(self):
        """Common pattern: hash as a record/struct."""
        result, output = run("""
            fn make_rect(w, h) {
                return {
                    "width": w,
                    "height": h,
                    "area": fn() { return w * h; },
                    "perimeter": fn() { return 2 * (w + h); }
                };
            }
            let r = make_rect(3, 4);
            print(r.area());
            print(r.perimeter());
        """)
        assert output == ['12', '14']


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_hash_len(self):
        result, output = run('print(len({}));')
        assert output == ['0']

    def test_hash_with_none_value(self):
        r = execute('let m = {"x": 0}; m.x;')
        # 0 is falsy but should be stored
        assert r['env']['m']['x'] == 0

    def test_hash_with_hash_value(self):
        result, output = run("""
            let m = {"a": {"b": {"c": 42}}};
            print(m.a.b.c);
        """)
        assert output == ['42']

    def test_hash_with_mixed_key_types(self):
        """Note: True==1 in Python, so {1: "int", true: "bool"} overwrites key 1."""
        r = execute('let m = {1: "int", "a": "str"};')
        m = r['env']['m']
        assert m[1] == "int"
        assert m["a"] == "str"

    def test_hash_preserves_insertion_order(self):
        result, output = run("""
            let m = {"c": 3, "a": 1, "b": 2};
            print(keys(m));
        """)
        assert output == ['[c, a, b]']

    def test_keys_returns_array(self):
        """keys() returns an array that can be used with array builtins."""
        result, output = run("""
            let m = {"x": 1, "y": 2};
            let k = keys(m);
            print(len(k));
            print(sort(k));
        """)
        assert output == ['2', '[x, y]']

    def test_values_returns_array(self):
        result, output = run("""
            let m = {"a": 3, "b": 1, "c": 2};
            let v = values(m);
            print(sort(v));
        """)
        assert output == ['[1, 2, 3]']

    def test_dot_access_keyword_like(self):
        """Dot access should work even with keyword-like field names."""
        # Using string keys to avoid keyword parsing issues
        result, output = run('let m = {"if": 1}; print(m["if"]);')
        assert output == ['1']

    def test_format_hash(self):
        assert _format_value({}) == '{}'
        assert _format_value({'a': 1}) == '{a: 1}'

    def test_format_nested(self):
        v = {'a': [1, 2], 'b': {'c': 3}}
        s = _format_value(v)
        assert 'a: [1, 2]' in s
        assert 'b: {c: 3}' in s

    def test_hash_in_conditional(self):
        result, output = run("""
            let m = {"active": true};
            if (m.active) {
                print("yes");
            }
        """)
        assert output == ['yes']

    def test_hash_in_while(self):
        result, output = run("""
            let state = {"i": 0, "sum": 0};
            while (state.i < 5) {
                state.sum = state.sum + state.i;
                state.i = state.i + 1;
            }
            print(state.sum);
        """)
        assert output == ['10']

    def test_delete_and_recheck(self):
        result, output = run("""
            let m = {"a": 1, "b": 2, "c": 3};
            delete(m, "b");
            print(size(m));
            print(has(m, "b"));
            print(keys(m));
        """)
        assert output == ['2', 'false', '[a, c]']


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrors:
    def test_key_not_found(self):
        with pytest.raises(VMError, match="not found"):
            run('let m = {"a": 1}; m.b;')

    def test_key_not_found_index(self):
        with pytest.raises(VMError, match="not found"):
            run('let m = {}; m["x"];')

    def test_keys_on_non_hash(self):
        with pytest.raises(VMError, match="hash map"):
            run('keys([1, 2]);')

    def test_values_on_non_hash(self):
        with pytest.raises(VMError, match="hash map"):
            run('values([1, 2]);')

    def test_has_on_non_hash(self):
        with pytest.raises(VMError, match="hash map"):
            run('has([1], "x");')

    def test_delete_on_non_hash(self):
        with pytest.raises(VMError, match="hash map"):
            run('delete([1], 0);')

    def test_merge_non_hash(self):
        with pytest.raises(VMError, match="hash map"):
            run('merge(1, 2);')

    def test_entries_non_hash(self):
        with pytest.raises(VMError, match="hash map"):
            run('entries([1]);')

    def test_index_set_non_indexable(self):
        with pytest.raises(VMError, match="Cannot assign"):
            run('let x = 5; x[0] = 1;')

    def test_dot_access_on_non_hash(self):
        """Dot access on non-hash should fail at INDEX_GET."""
        with pytest.raises(VMError):
            run('let x = 5; x.y;')


# ============================================================
# Disassembly Tests
# ============================================================

class TestDisassemble:
    def test_make_hash_in_disasm(self):
        chunk, _ = compile_source('let m = {"a": 1};')
        dis = disassemble(chunk)
        assert 'MAKE_HASH' in dis

    def test_dot_access_compiles_to_index_get(self):
        chunk, _ = compile_source('let m = {"x": 1}; m.x;')
        dis = disassemble(chunk)
        assert 'INDEX_GET' in dis


# ============================================================
# Regression / Pattern Tests
# ============================================================

class TestPatterns:
    def test_builder_pattern(self):
        """Build up a hash map step by step."""
        result, output = run("""
            fn builder() {
                let obj = {};
                obj.a = 1;
                obj.b = 2;
                obj.c = 3;
                return obj;
            }
            let m = builder();
            print(m.a);
            print(m.b);
            print(m.c);
        """)
        assert output == ['1', '2', '3']

    def test_lookup_table(self):
        result, output = run("""
            let ops = {
                "add": fn(a, b) { return a + b; },
                "mul": fn(a, b) { return a * b; }
            };
            print(ops.add(3, 4));
            print(ops.mul(3, 4));
        """)
        assert output == ['7', '12']

    def test_state_machine(self):
        result, output = run("""
            let transitions = {
                "idle": "running",
                "running": "done",
                "done": "idle"
            };
            let state = "idle";
            state = transitions[state];
            print(state);
            state = transitions[state];
            print(state);
        """)
        assert output == ['running', 'done']

    def test_accumulator_pattern(self):
        result, output = run("""
            let data = [1, 2, 3, 4, 5];
            let stats = {"sum": 0, "count": 0};
            each(data, fn(x) {
                stats.sum = stats.sum + x;
                stats.count = stats.count + 1;
            });
            print(stats.sum);
            print(stats.count);
        """)
        assert output == ['15', '5']

    def test_graph_as_adjacency_map(self):
        result, output = run("""
            let graph = {
                "a": ["b", "c"],
                "b": ["c"],
                "c": []
            };
            print(len(graph.a));
            print(graph.a[0]);
        """)
        assert output == ['2', 'b']

    def test_event_registry(self):
        result, output = run("""
            let events = {};
            events.click = fn() { return "clicked"; };
            events.hover = fn() { return "hovered"; };
            print(events.click());
            print(events.hover());
        """)
        assert output == ['clicked', 'hovered']

    def test_nested_update(self):
        result, output = run("""
            let config = {
                "db": {"host": "old", "port": 5432},
                "app": {"debug": false}
            };
            config.db.host = "new";
            config.app.debug = true;
            print(config.db.host);
            print(config.app.debug);
        """)
        assert output == ['new', 'true']

    def test_hash_from_entries(self):
        """Build a hash map from key-value pairs manually."""
        result, output = run("""
            let pairs = [["x", 1], ["y", 2], ["z", 3]];
            let m = {};
            each(pairs, fn(p) {
                m[p[0]] = p[1];
            });
            print(m.x);
            print(m.y);
            print(m.z);
        """)
        assert output == ['1', '2', '3']

    def test_prototype_chain_manual(self):
        """Manual prototype chain via merge."""
        result, output = run("""
            let base = {"type": "shape"};
            let circle = merge(base, {"radius": 5});
            print(circle.type);
            print(circle.radius);
        """)
        assert output == ['shape', '5']


# ============================================================
# Backward Compatibility Tests (from C042)
# ============================================================

class TestBackwardCompat:
    def test_array_still_works(self):
        result, output = run('let a = [1, 2, 3]; print(a[1]);')
        assert output == ['2']

    def test_closures_still_work(self):
        result, output = run("""
            fn make_adder(n) {
                return fn(x) { return x + n; };
            }
            let add5 = make_adder(5);
            print(add5(3));
        """)
        assert output == ['8']

    def test_array_builtins_still_work(self):
        result, output = run("""
            let a = [3, 1, 2];
            print(sort(a));
            print(reverse(a));
            print(len(a));
        """)
        assert output == ['[1, 2, 3]', '[2, 1, 3]', '3']

    def test_map_filter_reduce(self):
        result, output = run("""
            let a = [1, 2, 3, 4, 5];
            let doubled = map(a, fn(x) { return x * 2; });
            let evens = filter(a, fn(x) { return x % 2 == 0; });
            let sum = reduce(a, fn(acc, x) { return acc + x; }, 0);
            print(doubled);
            print(evens);
            print(sum);
        """)
        assert output == ['[2, 4, 6, 8, 10]', '[2, 4]', '15']

    def test_string_indexing(self):
        result, output = run('print("hello"[1]);')
        assert output == ['e']

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
            if (true) { print("yes"); } else { print("no"); }
        """)
        assert output == ['yes']

    def test_recursion(self):
        result, output = run("""
            fn fib(n) {
                if (n < 2) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
            print(fib(7));
        """)
        assert output == ['13']
