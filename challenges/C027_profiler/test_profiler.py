"""
Tests for C027 Profiler -- Runtime Performance Analysis for Stack VM
Target: 130+ tests covering all profiler features
"""

import sys
import os
import json
import time
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from profiler import (
    Profiler, ProfiledVM, ProfileSnapshot, FunctionProfile, LineProfile,
    InstructionProfile, CallGraphEdge, FlameFrame,
    profile_source, flat_profile, hotspots,
    format_flat_profile, format_hotspots, format_call_graph,
    compile_source,
)


# ============================================================
# 1. Basic Profiling (correct execution)
# ============================================================

class TestBasicExecution:
    """Verify ProfiledVM produces correct results (same as regular VM)."""

    def test_simple_expression(self):
        r = profile_source("1 + 2;")
        assert r['result'] is None or r['result'] is not None  # runs without error

    def test_arithmetic(self):
        r = profile_source("let x = 3 + 4 * 2; print(x);")
        assert r['output'] == ['11']

    def test_variable_assignment(self):
        r = profile_source("let x = 10; let y = 20; let z = x + y; print(z);")
        assert r['output'] == ['30']

    def test_string_output(self):
        r = profile_source('let s = "hello"; print(s);')
        assert r['output'] == ['hello']

    def test_boolean_output(self):
        r = profile_source("print(true); print(false);")
        assert r['output'] == ['true', 'false']

    def test_comparison(self):
        r = profile_source("print(3 < 5); print(5 < 3);")
        assert r['output'] == ['true', 'false']

    def test_if_else(self):
        r = profile_source("if (true) { print(1); } else { print(2); }")
        assert r['output'] == ['1']

    def test_if_false_branch(self):
        r = profile_source("if (false) { print(1); } else { print(2); }")
        assert r['output'] == ['2']

    def test_while_loop(self):
        r = profile_source("let i = 0; while (i < 5) { i = i + 1; } print(i);")
        assert r['output'] == ['5']

    def test_function_call(self):
        r = profile_source("fn add(a, b) { return a + b; } print(add(3, 4));")
        assert r['output'] == ['7']

    def test_recursive_function(self):
        src = """
        fn fact(n) {
            if (n <= 1) { return 1; }
            return n * fact(n - 1);
        }
        print(fact(5));
        """
        r = profile_source(src)
        assert r['output'] == ['120']

    def test_nested_functions(self):
        src = """
        fn double(x) { return x * 2; }
        fn quad(x) { return double(double(x)); }
        print(quad(3));
        """
        r = profile_source(src)
        assert r['output'] == ['12']

    def test_multiple_functions(self):
        src = """
        fn a() { return 1; }
        fn b() { return 2; }
        fn c() { return 3; }
        print(a() + b() + c());
        """
        r = profile_source(src)
        assert r['output'] == ['6']

    def test_modulo(self):
        r = profile_source("print(10 % 3);")
        assert r['output'] == ['1']

    def test_unary_neg(self):
        r = profile_source("print(-5);")
        assert r['output'] == ['-5']

    def test_logical_and(self):
        r = profile_source("print(true and false);")
        assert r['output'] == ['false']

    def test_logical_or(self):
        r = profile_source("print(false or true);")
        assert r['output'] == ['true']

    def test_not(self):
        r = profile_source("print(not true);")
        assert r['output'] == ['false']

    def test_float_arithmetic(self):
        r = profile_source("print(3.5 + 1.5);")
        assert r['output'] == ['5.0']

    def test_division(self):
        r = profile_source("print(10 / 3);")
        assert r['output'] == ['3']  # integer division

    def test_nested_if(self):
        src = """
        let x = 10;
        if (x > 5) {
            if (x > 8) {
                print(1);
            } else {
                print(2);
            }
        } else {
            print(3);
        }
        """
        r = profile_source(src)
        assert r['output'] == ['1']


# ============================================================
# 2. Profile Data Structure
# ============================================================

class TestProfileData:
    """Test that profiling data is correctly collected."""

    def test_profile_exists(self):
        r = profile_source("1 + 2;")
        assert 'profile' in r
        assert isinstance(r['profile'], ProfileSnapshot)

    def test_total_steps_positive(self):
        r = profile_source("let x = 1;")
        assert r['profile'].total_steps > 0

    def test_total_time_positive(self):
        r = profile_source("let x = 1;")
        assert r['profile'].total_time >= 0

    def test_main_function_profiled(self):
        r = profile_source("let x = 1;")
        assert '<main>' in r['profile'].functions

    def test_main_call_count(self):
        r = profile_source("let x = 1;")
        assert r['profile'].functions['<main>'].call_count == 1

    def test_function_profiled(self):
        r = profile_source("fn foo() { return 1; } foo();")
        assert 'foo' in r['profile'].functions

    def test_function_call_count(self):
        src = "fn foo() { return 1; } foo(); foo(); foo();"
        r = profile_source(src)
        assert r['profile'].functions['foo'].call_count == 3

    def test_function_self_time(self):
        r = profile_source("fn foo() { return 1; } foo();")
        assert r['profile'].functions['foo'].self_time >= 0

    def test_function_total_time(self):
        r = profile_source("fn foo() { return 1; } foo();")
        fp = r['profile'].functions['foo']
        assert fp.total_time >= fp.self_time

    def test_function_self_steps(self):
        r = profile_source("fn foo() { return 1; } foo();")
        assert r['profile'].functions['foo'].self_steps > 0

    def test_function_total_steps(self):
        r = profile_source("fn foo() { return 1; } foo();")
        fp = r['profile'].functions['foo']
        assert fp.total_steps >= fp.self_steps

    def test_peak_stack_depth(self):
        r = profile_source("let x = 1 + 2 + 3 + 4;")
        assert r['profile'].peak_stack_depth > 0

    def test_peak_call_depth(self):
        src = """
        fn a() { return 1; }
        fn b() { return a(); }
        fn c() { return b(); }
        c();
        """
        r = profile_source(src)
        # main -> c -> b -> a = depth 4
        assert r['profile'].peak_call_depth >= 4

    def test_line_profiling(self):
        r = profile_source("let x = 1;\nlet y = 2;\nprint(x + y);")
        assert len(r['profile'].lines) > 0

    def test_instruction_profiling(self):
        r = profile_source("let x = 1;")
        assert len(r['profile'].instructions) > 0

    def test_const_instruction_counted(self):
        r = profile_source("let x = 1;")
        assert 'CONST' in r['profile'].instructions
        assert r['profile'].instructions['CONST'].count > 0

    def test_store_instruction_counted(self):
        r = profile_source("let x = 1;")
        assert 'STORE' in r['profile'].instructions

    def test_halt_instruction_counted(self):
        r = profile_source("let x = 1;")
        assert 'HALT' in r['profile'].instructions

    def test_steps_matches_instruction_sum(self):
        r = profile_source("let x = 1; let y = 2;")
        p = r['profile']
        instr_sum = sum(ip.count for ip in p.instructions.values())
        assert instr_sum == p.total_steps


# ============================================================
# 3. Call Graph
# ============================================================

class TestCallGraph:
    """Test call graph construction."""

    def test_no_calls_no_edges(self):
        r = profile_source("let x = 1;")
        assert len(r['profile'].call_graph) == 0

    def test_single_call_edge(self):
        r = profile_source("fn foo() { return 1; } foo();")
        edges = r['profile'].call_graph
        assert len(edges) == 1
        assert edges[0].caller == '<main>'
        assert edges[0].callee == 'foo'
        assert edges[0].count == 1

    def test_multiple_call_edges(self):
        src = """
        fn a() { return 1; }
        fn b() { return a(); }
        b();
        """
        r = profile_source(src)
        edges = r['profile'].call_graph
        callers = {(e.caller, e.callee) for e in edges}
        assert ('<main>', 'b') in callers
        assert ('b', 'a') in callers

    def test_call_count_in_edge(self):
        src = "fn foo() { return 1; } foo(); foo(); foo();"
        r = profile_source(src)
        edge = [e for e in r['profile'].call_graph if e.callee == 'foo'][0]
        assert edge.count == 3

    def test_caller_tracking(self):
        src = "fn foo() { return 1; } foo();"
        r = profile_source(src)
        fp = r['profile'].functions['foo']
        assert '<main>' in fp.callers
        assert fp.callers['<main>'] == 1

    def test_callee_tracking(self):
        src = "fn foo() { return 1; } foo();"
        r = profile_source(src)
        main_fp = r['profile'].functions['<main>']
        assert 'foo' in main_fp.callees
        assert main_fp.callees['foo'] == 1

    def test_recursive_call_graph(self):
        src = """
        fn fact(n) {
            if (n <= 1) { return 1; }
            return n * fact(n - 1);
        }
        fact(3);
        """
        r = profile_source(src)
        edges = r['profile'].call_graph
        # fact calls itself
        self_edges = [e for e in edges if e.caller == 'fact' and e.callee == 'fact']
        assert len(self_edges) == 1
        assert self_edges[0].count == 2  # fact(3)->fact(2)->fact(1), 2 recursive calls

    def test_diamond_call_graph(self):
        src = """
        fn leaf() { return 1; }
        fn left() { return leaf(); }
        fn right() { return leaf(); }
        left();
        right();
        """
        r = profile_source(src)
        leaf_edges = [e for e in r['profile'].call_graph if e.callee == 'leaf']
        callers = {e.caller for e in leaf_edges}
        assert 'left' in callers
        assert 'right' in callers


# ============================================================
# 4. Profiler High-Level API
# ============================================================

class TestProfilerAPI:
    """Test the Profiler class high-level methods."""

    def test_profiler_creation(self):
        p = Profiler()
        assert p.profiles == []

    def test_profiler_profile(self):
        p = Profiler()
        r = p.profile("let x = 1;")
        assert len(p.profiles) == 1

    def test_profiler_multiple_profiles(self):
        p = Profiler()
        p.profile("let x = 1;")
        p.profile("let y = 2;")
        assert len(p.profiles) == 2

    def test_get_latest_profile(self):
        p = Profiler()
        p.profile("let x = 1;")
        latest = p.get_latest_profile()
        assert latest is not None
        assert isinstance(latest, ProfileSnapshot)

    def test_get_latest_profile_empty(self):
        p = Profiler()
        assert p.get_latest_profile() is None

    def test_flat_profile_returns_list(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        fp = p.flat_profile()
        assert isinstance(fp, list)
        assert len(fp) > 0

    def test_flat_profile_has_main(self):
        p = Profiler()
        p.profile("let x = 1;")
        fp = p.flat_profile()
        names = [e['name'] for e in fp]
        assert '<main>' in names

    def test_flat_profile_has_function(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        fp = p.flat_profile()
        names = [e['name'] for e in fp]
        assert 'foo' in names

    def test_flat_profile_sorted_by_self_time(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        fp = p.flat_profile()
        # Should be sorted by self_time descending
        for i in range(len(fp) - 1):
            assert fp[i]['self_time'] >= fp[i + 1]['self_time']

    def test_flat_profile_fields(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        fp = p.flat_profile()
        entry = fp[0]
        assert 'name' in entry
        assert 'call_count' in entry
        assert 'total_time' in entry
        assert 'self_time' in entry
        assert 'avg_time' in entry
        assert 'total_steps' in entry
        assert 'self_steps' in entry
        assert 'pct_time' in entry
        assert 'pct_steps' in entry

    def test_flat_profile_pct_sums_to_100(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        fp = p.flat_profile()
        total_pct = sum(e['pct_steps'] for e in fp)
        assert abs(total_pct - 100.0) < 1.0  # within 1% tolerance

    def test_flat_profile_empty(self):
        p = Profiler()
        fp = p.flat_profile()
        assert fp == []

    def test_hotspots_returns_list(self):
        p = Profiler()
        p.profile("let x = 1;\nlet y = 2;")
        h = p.hotspots()
        assert isinstance(h, list)

    def test_hotspots_sorted_by_count(self):
        src = "let i = 0;\nwhile (i < 10) {\ni = i + 1;\n}\nprint(i);"
        p = Profiler()
        p.profile(src)
        h = p.hotspots()
        for i in range(len(h) - 1):
            assert h[i]['execution_count'] >= h[i + 1]['execution_count']

    def test_hotspots_top_n(self):
        src = "let a = 1;\nlet b = 2;\nlet c = 3;\nlet d = 4;\nlet e = 5;"
        p = Profiler()
        p.profile(src)
        h = p.hotspots(top_n=2)
        assert len(h) <= 2

    def test_hotspots_fields(self):
        p = Profiler()
        p.profile("let x = 1;")
        h = p.hotspots()
        if h:
            assert 'line' in h[0]
            assert 'execution_count' in h[0]
            assert 'total_time' in h[0]
            assert 'instructions' in h[0]

    def test_call_graph_api(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        cg = p.call_graph()
        assert isinstance(cg, list)
        assert len(cg) > 0
        assert 'caller' in cg[0]
        assert 'callee' in cg[0]
        assert 'count' in cg[0]

    def test_instruction_profile(self):
        p = Profiler()
        p.profile("let x = 1 + 2;")
        ip = p.instruction_profile()
        assert isinstance(ip, list)
        assert len(ip) > 0

    def test_instruction_profile_sorted(self):
        p = Profiler()
        p.profile("let x = 1 + 2;")
        ip = p.instruction_profile()
        for i in range(len(ip) - 1):
            assert ip[i]['count'] >= ip[i + 1]['count']

    def test_instruction_profile_fields(self):
        p = Profiler()
        p.profile("let x = 1;")
        ip = p.instruction_profile()
        entry = ip[0]
        assert 'opcode' in entry
        assert 'count' in entry
        assert 'total_time' in entry
        assert 'avg_time' in entry
        assert 'pct' in entry

    def test_coverage(self):
        p = Profiler()
        p.profile("let x = 1;\nlet y = 2;")
        cov = p.coverage()
        assert 'covered_lines' in cov
        assert 'total_covered' in cov
        assert 'line_counts' in cov
        assert cov['total_covered'] > 0

    def test_coverage_line_counts(self):
        src = "let i = 0;\nwhile (i < 3) {\ni = i + 1;\n}"
        p = Profiler()
        p.profile(src)
        cov = p.coverage()
        # Loop body lines should have higher counts
        assert len(cov['line_counts']) > 0

    def test_summary(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        s = p.summary()
        assert 'total_time' in s
        assert 'total_steps' in s
        assert 'function_count' in s
        assert 'total_calls' in s
        assert 'peak_stack_depth' in s
        assert 'peak_call_depth' in s
        assert 'hottest_function' in s
        assert 'hottest_line' in s

    def test_summary_function_count(self):
        p = Profiler()
        p.profile("fn a() { return 1; } fn b() { return 2; } a(); b();")
        s = p.summary()
        assert s['function_count'] == 3  # <main>, a, b

    def test_summary_empty(self):
        p = Profiler()
        s = p.summary()
        assert s == {}


# ============================================================
# 5. Flame Graph
# ============================================================

class TestFlameGraph:
    """Test flame graph data generation."""

    def test_flame_data_structure(self):
        p = Profiler()
        p.profile("let x = 1;")
        fd = p.flame_data()
        assert 'name' in fd
        assert 'value' in fd
        assert 'children' in fd

    def test_flame_data_main(self):
        p = Profiler()
        p.profile("let x = 1;")
        fd = p.flame_data()
        assert fd['name'] == '<main>'

    def test_flame_data_with_calls(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        fd = p.flame_data()
        assert fd['name'] == '<main>'
        assert len(fd['children']) > 0
        child_names = [c['name'] for c in fd['children']]
        assert 'foo' in child_names

    def test_flame_data_nested(self):
        src = """
        fn a() { return 1; }
        fn b() { return a(); }
        b();
        """
        p = Profiler()
        p.profile(src)
        fd = p.flame_data()
        # main -> b -> a
        b_node = [c for c in fd['children'] if c['name'] == 'b'][0]
        a_node = [c for c in b_node['children'] if c['name'] == 'a'][0]
        assert a_node['name'] == 'a'

    def test_flame_data_empty(self):
        p = Profiler()
        fd = p.flame_data()
        assert fd['name'] == '<root>'
        assert fd['value'] == 0


# ============================================================
# 6. Profile Comparison
# ============================================================

class TestComparison:
    """Test profile comparison feature."""

    def test_compare_returns_dict(self):
        p = Profiler()
        p.profile("let x = 1;")
        p.profile("let x = 1; let y = 2;")
        result = p.compare(p.profiles[0], p.profiles[1])
        assert isinstance(result, dict)

    def test_compare_time_diff(self):
        p = Profiler()
        p.profile("let x = 1;")
        p.profile("let x = 1;")
        result = p.compare(p.profiles[0], p.profiles[1])
        assert 'time_diff' in result
        assert 'time_ratio' in result

    def test_compare_steps_diff(self):
        p = Profiler()
        p.profile("let x = 1;")
        p.profile("let x = 1; let y = 2; let z = 3;")
        result = p.compare(p.profiles[0], p.profiles[1])
        assert result['steps_diff'] > 0  # second program has more steps

    def test_compare_function_diffs(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        p.profile("fn foo() { return 1; } foo(); foo();")
        result = p.compare(p.profiles[0], p.profiles[1])
        assert 'function_diffs' in result
        foo_diff = [d for d in result['function_diffs'] if d['name'] == 'foo'][0]
        assert foo_diff['calls_diff'] == 1  # one more call

    def test_compare_added_function(self):
        p = Profiler()
        p.profile("let x = 1;")
        p.profile("fn foo() { return 1; } foo();")
        result = p.compare(p.profiles[0], p.profiles[1])
        foo_diff = [d for d in result['function_diffs'] if d['name'] == 'foo'][0]
        assert foo_diff['status'] == 'added'

    def test_compare_removed_function(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        p.profile("let x = 1;")
        result = p.compare(p.profiles[0], p.profiles[1])
        foo_diff = [d for d in result['function_diffs'] if d['name'] == 'foo'][0]
        assert foo_diff['status'] == 'removed'

    def test_compare_stack_diff(self):
        p = Profiler()
        p.profile("let x = 1;")
        p.profile("let x = 1 + 2 + 3 + 4 + 5;")
        result = p.compare(p.profiles[0], p.profiles[1])
        assert 'stack_diff' in result


# ============================================================
# 7. Serialization
# ============================================================

class TestSerialization:
    """Test profile export/import."""

    def test_export_returns_dict(self):
        p = Profiler()
        p.profile("let x = 1;")
        exported = p.export_profile()
        assert isinstance(exported, dict)

    def test_export_has_fields(self):
        p = Profiler()
        p.profile("let x = 1;")
        exported = p.export_profile()
        assert 'total_time' in exported
        assert 'total_steps' in exported
        assert 'functions' in exported
        assert 'lines' in exported
        assert 'instructions' in exported
        assert 'call_graph' in exported

    def test_export_is_json_serializable(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        exported = p.export_profile()
        json_str = json.dumps(exported)
        assert isinstance(json_str, str)

    def test_import_roundtrip(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo(); foo();")
        exported = p.export_profile()
        json_str = json.dumps(exported)
        reimported_data = json.loads(json_str)

        p2 = Profiler()
        imported_profile = p2.import_profile(reimported_data)
        assert isinstance(imported_profile, ProfileSnapshot)
        assert imported_profile.total_steps == p.get_latest_profile().total_steps
        assert 'foo' in imported_profile.functions
        assert imported_profile.functions['foo'].call_count == 2

    def test_import_preserves_lines(self):
        p = Profiler()
        p.profile("let x = 1;\nlet y = 2;")
        exported = p.export_profile()

        p2 = Profiler()
        imported = p2.import_profile(exported)
        assert len(imported.lines) == len(p.get_latest_profile().lines)

    def test_import_preserves_instructions(self):
        p = Profiler()
        p.profile("let x = 1;")
        exported = p.export_profile()

        p2 = Profiler()
        imported = p2.import_profile(exported)
        assert len(imported.instructions) == len(p.get_latest_profile().instructions)

    def test_import_preserves_call_graph(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        exported = p.export_profile()

        p2 = Profiler()
        imported = p2.import_profile(exported)
        assert len(imported.call_graph) == len(p.get_latest_profile().call_graph)

    def test_export_empty(self):
        p = Profiler()
        assert p.export_profile() == {}

    def test_import_adds_to_profiles(self):
        p = Profiler()
        p.import_profile({'total_steps': 10, 'total_time': 0.1})
        assert len(p.profiles) == 1


# ============================================================
# 8. Formatting
# ============================================================

class TestFormatting:
    """Test text formatting functions."""

    def test_format_flat_profile(self):
        entries = [
            {'name': 'foo', 'call_count': 3, 'pct_time': 60.0,
             'self_time': 0.001, 'total_time': 0.002, 'self_steps': 100},
        ]
        text = format_flat_profile(entries)
        assert 'foo' in text
        assert '3' in text

    def test_format_flat_profile_empty(self):
        text = format_flat_profile([])
        assert 'No profile data' in text

    def test_format_hotspots(self):
        entries = [
            {'line': 5, 'execution_count': 100, 'total_time': 0.001, 'instructions': 100},
        ]
        text = format_hotspots(entries)
        assert '5' in text
        assert '100' in text

    def test_format_hotspots_empty(self):
        text = format_hotspots([])
        assert 'No hotspot data' in text

    def test_format_call_graph(self):
        edges = [
            {'caller': '<main>', 'callee': 'foo', 'count': 3},
        ]
        text = format_call_graph(edges)
        assert '<main>' in text
        assert 'foo' in text
        assert '3' in text

    def test_format_call_graph_empty(self):
        text = format_call_graph([])
        assert 'No call graph data' in text


# ============================================================
# 9. Sampling Profiler
# ============================================================

class TestSampling:
    """Test statistical sampling profiler."""

    def test_sampling_mode(self):
        p = Profiler()
        p.enable_sampling(interval=10)
        r = p.profile("let i = 0; while (i < 100) { i = i + 1; }")
        assert r['profile'].sample_count > 0

    def test_sampling_disabled(self):
        p = Profiler()
        r = p.profile("let i = 0; while (i < 100) { i = i + 1; }")
        assert r['profile'].sample_count == 0

    def test_sampling_interval(self):
        p = Profiler()
        p.enable_sampling(interval=50)
        r = p.profile("let i = 0; while (i < 100) { i = i + 1; }")
        # With ~800 steps and interval 50, should get ~16 samples
        assert r['profile'].sample_count > 5

    def test_disable_sampling(self):
        p = Profiler()
        p.enable_sampling(interval=10)
        p.disable_sampling()
        r = p.profile("let i = 0; while (i < 100) { i = i + 1; }")
        assert r['profile'].sample_count == 0


# ============================================================
# 10. Edge Cases and Stress
# ============================================================

class TestEdgeCases:
    """Test edge cases and stress scenarios."""

    def test_empty_program(self):
        # A program with just a semicolon expression
        r = profile_source("0;")
        assert r['profile'].total_steps > 0

    def test_deeply_nested_calls(self):
        src = """
        fn a() { return 1; }
        fn b() { return a(); }
        fn c() { return b(); }
        fn d() { return c(); }
        fn e() { return d(); }
        e();
        """
        r = profile_source(src)
        assert r['profile'].peak_call_depth >= 6  # main + e + d + c + b + a

    def test_many_function_calls(self):
        calls = "; ".join(["foo()" for _ in range(50)])
        src = f"fn foo() {{ return 1; }} {calls};"
        r = profile_source(src)
        assert r['profile'].functions['foo'].call_count == 50

    def test_loop_hotspot(self):
        src = "let i = 0;\nwhile (i < 20) {\ni = i + 1;\n}"
        p = Profiler()
        p.profile(src)
        h = p.hotspots(top_n=1)
        assert h[0]['execution_count'] > 1  # loop lines execute multiple times

    def test_self_time_less_than_total(self):
        src = """
        fn inner() { return 1; }
        fn outer() { return inner(); }
        outer();
        """
        r = profile_source(src)
        outer_fp = r['profile'].functions['outer']
        assert outer_fp.self_time <= outer_fp.total_time + 0.001  # small tolerance

    def test_self_steps_less_than_total(self):
        src = """
        fn inner() { return 1; }
        fn outer() { return inner(); }
        outer();
        """
        r = profile_source(src)
        outer_fp = r['profile'].functions['outer']
        assert outer_fp.self_steps <= outer_fp.total_steps

    def test_recursive_timing(self):
        src = """
        fn fib(n) {
            if (n <= 1) { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        fib(8);
        """
        r = profile_source(src)
        fib_fp = r['profile'].functions['fib']
        assert fib_fp.call_count > 1
        assert fib_fp.total_time > 0

    def test_multiple_callees(self):
        src = """
        fn a() { return 1; }
        fn b() { return 2; }
        fn c() { return a() + b(); }
        c();
        """
        r = profile_source(src)
        c_fp = r['profile'].functions['c']
        assert 'a' in c_fp.callees
        assert 'b' in c_fp.callees

    def test_profiled_vm_output_matches(self):
        """ProfiledVM should produce identical output to regular VM."""
        src = """
        fn greet(name) {
            print(name);
        }
        greet("world");
        print(42);
        """
        # Run with profiled VM
        r = profile_source(src)
        assert r['output'] == ['world', '42']

    def test_large_loop(self):
        src = "let sum = 0; let i = 0; while (i < 1000) { sum = sum + i; i = i + 1; } print(sum);"
        r = profile_source(src)
        assert r['output'] == ['499500']
        assert r['profile'].total_steps > 1000

    def test_profile_with_string_operations(self):
        src = 'let a = "hello"; let b = " world"; print(a + b);'
        r = profile_source(src)
        assert r['output'] == ['hello world']

    def test_function_max_stack_depth(self):
        src = "fn foo() { return 1 + 2 + 3; } foo();"
        r = profile_source(src)
        fp = r['profile'].functions['foo']
        assert fp.max_stack_depth >= 0


# ============================================================
# 11. Convenience Functions
# ============================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_profile_source(self):
        r = profile_source("print(1);")
        assert r['output'] == ['1']
        assert 'profile' in r

    def test_flat_profile_function(self):
        fp = flat_profile("fn foo() { return 1; } foo();")
        assert isinstance(fp, list)
        names = [e['name'] for e in fp]
        assert 'foo' in names

    def test_hotspots_function(self):
        h = hotspots("let i = 0; while (i < 10) { i = i + 1; }")
        assert isinstance(h, list)
        assert len(h) > 0


# ============================================================
# 12. Step Accounting
# ============================================================

class TestStepAccounting:
    """Verify step counts are consistent."""

    def test_main_total_steps_equals_total(self):
        r = profile_source("let x = 1; let y = 2;")
        p = r['profile']
        main_fp = p.functions['<main>']
        assert main_fp.total_steps == p.total_steps

    def test_self_steps_add_up(self):
        """Sum of all self_steps should equal total_steps."""
        src = """
        fn a() { return 1; }
        fn b() { return a(); }
        b();
        """
        r = profile_source(src)
        p = r['profile']
        self_sum = sum(fp.self_steps for fp in p.functions.values())
        assert self_sum == p.total_steps

    def test_loop_increases_steps(self):
        r1 = profile_source("let i = 0; while (i < 5) { i = i + 1; }")
        r2 = profile_source("let i = 0; while (i < 10) { i = i + 1; }")
        assert r2['profile'].total_steps > r1['profile'].total_steps

    def test_function_overhead(self):
        """Calling a function adds steps compared to inline."""
        r1 = profile_source("let x = 1 + 2;")
        r2 = profile_source("fn add(a, b) { return a + b; } let x = add(1, 2);")
        assert r2['profile'].total_steps > r1['profile'].total_steps


# ============================================================
# 13. Multiple Profile Runs
# ============================================================

class TestMultipleRuns:
    """Test running multiple profiles."""

    def test_independent_profiles(self):
        p = Profiler()
        p.profile("fn foo() { return 1; } foo();")
        p.profile("fn bar() { return 2; } bar();")
        assert 'foo' in p.profiles[0].functions
        assert 'foo' not in p.profiles[1].functions
        assert 'bar' in p.profiles[1].functions
        assert 'bar' not in p.profiles[0].functions

    def test_compare_different_programs(self):
        p = Profiler()
        p.profile("let x = 1;")
        p.profile("let i = 0; while (i < 100) { i = i + 1; }")
        result = p.compare(p.profiles[0], p.profiles[1])
        assert result['steps_diff'] > 0

    def test_latest_profile_updates(self):
        p = Profiler()
        p.profile("let x = 1;")
        first = p.get_latest_profile()
        p.profile("let y = 2;")
        second = p.get_latest_profile()
        assert first is not second


# ============================================================
# 14. Instruction Coverage
# ============================================================

class TestInstructionCoverage:
    """Test that various instruction types are profiled."""

    def test_add_instruction(self):
        r = profile_source("let x = 1 + 2;")
        assert 'ADD' in r['profile'].instructions

    def test_sub_instruction(self):
        r = profile_source("let x = 5 - 3;")
        assert 'SUB' in r['profile'].instructions

    def test_mul_instruction(self):
        r = profile_source("let x = 2 * 3;")
        assert 'MUL' in r['profile'].instructions

    def test_div_instruction(self):
        r = profile_source("let x = 10 / 2;")
        assert 'DIV' in r['profile'].instructions

    def test_mod_instruction(self):
        r = profile_source("let x = 10 % 3;")
        assert 'MOD' in r['profile'].instructions

    def test_neg_instruction(self):
        r = profile_source("let x = -5;")
        assert 'NEG' in r['profile'].instructions

    def test_eq_instruction(self):
        r = profile_source("let x = 1 == 1;")
        assert 'EQ' in r['profile'].instructions

    def test_lt_instruction(self):
        r = profile_source("let x = 1 < 2;")
        assert 'LT' in r['profile'].instructions

    def test_not_instruction(self):
        r = profile_source("let x = not true;")
        assert 'NOT' in r['profile'].instructions

    def test_jump_instruction(self):
        r = profile_source("let i = 0; while (i < 1) { i = i + 1; }")
        assert 'JUMP' in r['profile'].instructions

    def test_jump_if_false_instruction(self):
        r = profile_source("if (false) { let x = 1; }")
        assert 'JUMP_IF_FALSE' in r['profile'].instructions

    def test_call_instruction(self):
        r = profile_source("fn foo() { return 1; } foo();")
        assert 'CALL' in r['profile'].instructions

    def test_return_instruction(self):
        r = profile_source("fn foo() { return 1; } foo();")
        assert 'RETURN' in r['profile'].instructions

    def test_print_instruction(self):
        r = profile_source("print(1);")
        assert 'PRINT' in r['profile'].instructions

    def test_load_instruction(self):
        r = profile_source("let x = 1; let y = x;")
        assert 'LOAD' in r['profile'].instructions

    def test_store_instruction(self):
        r = profile_source("let x = 1;")
        assert 'STORE' in r['profile'].instructions


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
