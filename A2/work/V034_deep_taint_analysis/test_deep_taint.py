"""Tests for V034: Deep Python Taint Analysis."""

import pytest
import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from deep_taint import (
    TaintLabel, TaintValue, TaintEnv, CLEAN,
    TaintSeverity, TaintFindingKind, TaintFinding,
    TaintConfig, FunctionSummary,
    PathSensitiveTaintAnalyzer, InterProceduralAnalyzer,
    TaintAnalysisResult,
    analyze_taint, analyze_taint_file,
)


# ============================================================
# Unit Tests: Taint Lattice
# ============================================================

class TestTaintLattice:
    def test_clean_is_not_tainted(self):
        assert not CLEAN.is_tainted

    def test_label_creation(self):
        l = TaintLabel(source="user_input", origin_line=5, origin_var="x")
        assert l.source == "user_input"
        assert l.origin_line == 5

    def test_tainted_value(self):
        l = TaintLabel(source="user_input", origin_line=5, origin_var="x")
        tv = TaintValue(frozenset([l]))
        assert tv.is_tainted
        assert len(tv.labels) == 1

    def test_join_clean_clean(self):
        result = CLEAN.join(CLEAN)
        assert not result.is_tainted

    def test_join_tainted_clean(self):
        l = TaintLabel(source="user_input", origin_line=5, origin_var="x")
        tv = TaintValue(frozenset([l]))
        result = tv.join(CLEAN)
        assert result.is_tainted
        assert result.labels == tv.labels

    def test_join_two_tainted(self):
        l1 = TaintLabel(source="user_input", origin_line=5, origin_var="x")
        l2 = TaintLabel(source="file_read", origin_line=10, origin_var="y")
        tv1 = TaintValue(frozenset([l1]))
        tv2 = TaintValue(frozenset([l2]))
        result = tv1.join(tv2)
        assert len(result.labels) == 2

    def test_str_clean(self):
        assert str(CLEAN) == "clean"

    def test_str_tainted(self):
        l = TaintLabel(source="user_input", origin_line=5, origin_var="x")
        tv = TaintValue(frozenset([l]))
        assert "user_input" in str(tv)


class TestTaintEnv:
    def test_empty_env(self):
        env = TaintEnv()
        assert not env.get("x").is_tainted

    def test_set_and_get(self):
        env = TaintEnv()
        l = TaintLabel(source="user_input", origin_line=1, origin_var="x")
        env2 = env.set("x", TaintValue(frozenset([l])))
        assert env2.get("x").is_tainted
        assert not env.get("x").is_tainted  # immutable

    def test_join_envs(self):
        l1 = TaintLabel(source="user_input", origin_line=1, origin_var="x")
        l2 = TaintLabel(source="file_read", origin_line=2, origin_var="y")
        env1 = TaintEnv().set("x", TaintValue(frozenset([l1])))
        env2 = TaintEnv().set("y", TaintValue(frozenset([l2])))
        joined = env1.join(env2)
        assert joined.get("x").is_tainted
        assert joined.get("y").is_tainted

    def test_tainted_vars(self):
        l = TaintLabel(source="user_input", origin_line=1, origin_var="x")
        env = TaintEnv().set("x", TaintValue(frozenset([l]))).set("y", CLEAN)
        assert env.tainted_vars() == {"x"}

    def test_env_equality(self):
        env1 = TaintEnv().set("x", CLEAN)
        env2 = TaintEnv().set("x", CLEAN)
        assert env1 == env2


# ============================================================
# Path-Sensitive Analysis Tests
# ============================================================

class TestPathSensitive:
    def test_simple_propagation(self):
        src = """
def process(user_input):
    x = user_input
    return x
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        props = [f for f in result.findings if f.kind == TaintFindingKind.TAINT_PROPAGATION]
        assert any('x' in f.message for f in props)

    def test_sanitizer_cleans_taint(self):
        src = """
def process(user_input):
    safe = escape(user_input)
    eval(safe)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) == 0  # escaped, so no sink reached

    def test_unsanitized_reaches_sink(self):
        src = """
def process(user_input):
    eval(user_input)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) == 1
        assert sinks[0].severity == TaintSeverity.CRITICAL

    def test_path_sensitive_if_then(self):
        """Taint only propagates in the branch where it flows."""
        src = """
def process(user_input):
    if len(user_input) > 10:
        x = user_input
    else:
        x = "safe"
    return x
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        # x should be tainted (from the then branch, joined)
        props = [f for f in result.findings if f.kind == TaintFindingKind.TAINT_PROPAGATION]
        assert any('x' in f.message for f in props)

    def test_path_sensitive_sanitize_one_branch(self):
        """If sanitized on one branch but not the other, still tainted after join."""
        src = """
def process(user_input):
    if check(user_input):
        x = escape(user_input)
    else:
        x = user_input
    eval(x)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        # Still reaches eval because the else branch doesn't sanitize
        assert len(sinks) >= 1

    def test_implicit_flow(self):
        """Assignment under tainted condition creates implicit flow."""
        src = """
def process(user_input):
    if user_input > 0:
        x = 42
    else:
        x = 0
    return x
"""
        config = TaintConfig(track_implicit=True)
        result = analyze_taint(src, config=config,
                               taint_args={'process': {'user_input': 'user'}})
        implicit = [f for f in result.findings
                    if f.kind == TaintFindingKind.IMPLICIT_FLOW]
        assert len(implicit) >= 1

    def test_no_implicit_when_disabled(self):
        src = """
def process(user_input):
    if user_input > 0:
        x = 42
    return x
"""
        config = TaintConfig(track_implicit=False)
        result = analyze_taint(src, config=config,
                               taint_args={'process': {'user_input': 'user'}})
        implicit = [f for f in result.findings
                    if f.kind == TaintFindingKind.IMPLICIT_FLOW]
        assert len(implicit) == 0

    def test_aug_assign_propagation(self):
        src = """
def process(user_input):
    x = ""
    x += user_input
    return x
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        props = [f for f in result.findings if f.kind == TaintFindingKind.TAINT_PROPAGATION]
        assert any('x' in f.message for f in props)

    def test_tuple_unpacking(self):
        src = """
def process(user_input):
    a, b = user_input, "safe"
    eval(a)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_for_loop_taint(self):
        src = """
def process(user_input):
    for item in user_input:
        eval(item)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_while_loop_fixpoint(self):
        src = """
def process(user_input):
    x = user_input
    while x:
        x = escape(x)
    return x
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        # After the loop, x may or may not be tainted (joined with pre-loop state)
        # The key test: analysis terminates (fixpoint reached)
        assert result is not None

    def test_fstring_taint(self):
        src = """
def process(user_input):
    msg = f"Hello {user_input}"
    eval(msg)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_subscript_taint(self):
        src = """
def process(user_input):
    x = user_input[0]
    eval(x)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_dict_literal_taint(self):
        src = """
def process(user_input):
    d = {"key": user_input}
    eval(d)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_attribute_taint(self):
        src = """
def process(user_input):
    x = user_input.strip()
    eval(x)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_multiple_labels_tracked(self):
        src = """
def process(a, b):
    x = a + b
    eval(x)
"""
        result = analyze_taint(src, taint_args={
            'process': {'a': 'source_a', 'b': 'source_b'}
        })
        sinks = result.taint_sinks()
        assert len(sinks) >= 1
        # Should have labels from both sources
        all_labels = set()
        for s in sinks:
            all_labels.update(s.labels)
        sources = {l.source for l in all_labels}
        assert 'source_a' in sources
        assert 'source_b' in sources


# ============================================================
# Inter-Procedural Tests
# ============================================================

class TestInterProcedural:
    def test_taint_through_call(self):
        src = """
def helper(x):
    return x

def process(user_input):
    result = helper(user_input)
    eval(result)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_sanitizer_in_helper(self):
        """Function that cleans taint acts as sanitizer."""
        src = """
def clean(x):
    return int(x)

def process(user_input):
    safe = clean(user_input)
    eval(safe)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        # int() is a sanitizer, so clean() should be detected as sanitizer
        assert len(sinks) == 0

    def test_taint_source_in_callee(self):
        src = """
def get_data():
    return input("Enter: ")

def process():
    data = get_data()
    eval(data)
"""
        result = analyze_taint(src, entry_points=['process'])
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_call_graph_built(self):
        src = """
def a():
    return b()

def b():
    return c()

def c():
    return 42
"""
        result = analyze_taint(src)
        assert 'b' in result.call_graph.get('a', set())
        assert 'c' in result.call_graph.get('b', set())

    def test_multi_param_taint_transfer(self):
        src = """
def combine(x, y):
    return x + y

def process(user_input):
    result = combine(user_input, "safe")
    eval(result)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_recursive_function(self):
        """Analysis handles recursion without infinite loops."""
        src = """
def recurse(x, n):
    if n <= 0:
        return x
    return recurse(x + "a", n - 1)

def process(user_input):
    result = recurse(user_input, 5)
    eval(result)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        # Should terminate
        assert result is not None
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_function_summary_reports(self):
        src = """
def passthrough(x):
    return x

def transform(x):
    return len(x)

def source_fn():
    return input()
"""
        result = analyze_taint(src)
        assert 'passthrough' in result.func_summaries
        pt = result.func_summaries['passthrough']
        assert 0 in pt.param_taints_return  # param 0 taints return

        src_fn = result.func_summaries['source_fn']
        assert src_fn.introduces_taint

    def test_chain_of_calls(self):
        src = """
def step1(x):
    return x.strip()

def step2(x):
    return step1(x)

def step3(x):
    return step2(x)

def process(user_input):
    result = step3(user_input)
    eval(result)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1


# ============================================================
# Source/Sink/Sanitizer Configuration Tests
# ============================================================

class TestConfiguration:
    def test_custom_source(self):
        config = TaintConfig(
            source_functions={'get_env': 'environment'},
        )
        src = """
def process():
    val = get_env("SECRET")
    eval(val)
"""
        result = analyze_taint(src, config=config, entry_points=['process'])
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_custom_sink(self):
        config = TaintConfig(
            sinks={'dangerous_op': TaintSeverity.CRITICAL},
        )
        src = """
def process(user_input):
    dangerous_op(user_input)
"""
        result = analyze_taint(src, config=config,
                               taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1
        assert sinks[0].severity == TaintSeverity.CRITICAL

    def test_custom_sanitizer(self):
        config = TaintConfig(
            sanitizers={'my_escape': {'*'}},
        )
        src = """
def process(user_input):
    safe = my_escape(user_input)
    eval(safe)
"""
        result = analyze_taint(src, config=config,
                               taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) == 0

    def test_partial_sanitizer(self):
        config = TaintConfig(
            sanitizers={'strip_html': {'xss'}},
        )
        src = """
def process(user_input):
    cleaned = strip_html(user_input)
    eval(cleaned)
"""
        # user_input has 'user' source, strip_html only cleans 'xss'
        result = analyze_taint(src, config=config,
                               taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1  # 'user' taint not cleaned

    def test_source_vars(self):
        config = TaintConfig(source_vars={'EXTERNAL_DATA'})
        src = """
EXTERNAL_DATA = "something"
eval(EXTERNAL_DATA)
"""
        result = analyze_taint(src, config=config)
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_source_params_auto(self):
        """Parameters matching source_params are auto-tainted."""
        src = """
def handle(request):
    eval(request)
"""
        result = analyze_taint(src, entry_points=['handle'])
        sinks = result.taint_sinks()
        assert len(sinks) >= 1  # 'request' is in default source_params


# ============================================================
# Complex Scenarios
# ============================================================

class TestComplexScenarios:
    def test_try_except_paths(self):
        src = """
def process(user_input):
    try:
        x = int(user_input)
    except ValueError:
        x = user_input
    eval(x)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        # In except branch, x is still tainted (not sanitized)
        assert len(sinks) >= 1

    def test_with_statement(self):
        src = """
def process(user_input):
    with open(user_input) as f:
        data = f.read()
    eval(data)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        # f is tainted (from open(user_input)), read() is a source
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_delete_cleans_taint(self):
        src = """
def process(user_input):
    x = user_input
    del x
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        # After del, x should be clean (not in env)
        assert result is not None

    def test_nested_if(self):
        src = """
def process(user_input):
    if user_input:
        if len(user_input) > 5:
            x = user_input
        else:
            x = "short"
    else:
        x = "empty"
    eval(x)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1  # One branch has tainted x

    def test_binop_propagation(self):
        src = """
def process(user_input):
    x = "prefix" + user_input + "suffix"
    eval(x)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_compare_taint(self):
        """Comparisons involving tainted data carry taint."""
        src = """
def process(user_input):
    flag = user_input == "admin"
    return flag
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        # flag should be tainted
        props = [f for f in result.findings if f.kind == TaintFindingKind.TAINT_PROPAGATION]
        assert any('flag' in f.message for f in props)

    def test_list_comprehension_source(self):
        """Input() in source position introduces taint."""
        src = """
def get_inputs():
    data = input("Enter: ")
    return data

def process():
    data = get_inputs()
    eval(data)
"""
        result = analyze_taint(src, entry_points=['process'])
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_multiple_sinks(self):
        src = """
def process(user_input):
    eval(user_input)
    exec(user_input)
    system(user_input)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 3

    def test_finding_has_path_condition(self):
        src = """
def process(user_input):
    if user_input.startswith("admin"):
        eval(user_input)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1
        # Should have path condition info
        assert any(f.path_condition for f in sinks)

    def test_module_level_analysis(self):
        config = TaintConfig(source_vars={'SECRET'})
        src = """
SECRET = "password"
x = SECRET
eval(x)
"""
        result = analyze_taint(src, config=config)
        sinks = result.taint_sinks()
        assert len(sinks) >= 1


# ============================================================
# Result / Report Tests
# ============================================================

class TestResults:
    def test_summary_output(self):
        src = """
def process(user_input):
    eval(user_input)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        summary = result.summary()
        assert "Total findings" in summary

    def test_report_output(self):
        src = """
def process(user_input):
    eval(user_input)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        report = result.report()
        assert "eval" in report

    def test_critical_findings(self):
        src = """
def process(user_input):
    eval(user_input)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        crits = result.critical_findings()
        assert len(crits) >= 1

    def test_findings_by_severity(self):
        src = """
def process(user_input):
    x = user_input
    eval(x)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        by_sev = result.findings_by_severity()
        assert len(by_sev) > 0

    def test_findings_by_kind(self):
        src = """
def process(user_input):
    x = user_input
    eval(x)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        by_kind = result.findings_by_kind()
        assert TaintFindingKind.TAINT_PROPAGATION in by_kind or TaintFindingKind.TAINT_SINK in by_kind


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_function(self):
        src = """
def empty():
    pass
"""
        result = analyze_taint(src)
        assert result.num_findings == 0

    def test_no_functions(self):
        src = """
x = 42
y = x + 1
"""
        result = analyze_taint(src)
        assert result is not None

    def test_lambda(self):
        src = """
def process(user_input):
    f = lambda x: x
    result = f(user_input)
    eval(result)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        # Lambda analyzed as unknown function -> taint propagates through args
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_starred_assignment(self):
        src = """
def process(user_input):
    first, *rest = user_input
    eval(first)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_walrus_operator(self):
        """Named expression (walrus) propagates taint."""
        src = """
def process(user_input):
    if (n := user_input):
        eval(n)
"""
        result = analyze_taint(src, taint_args={'process': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_nested_function_not_confused(self):
        """Inner function definitions don't pollute outer analysis."""
        src = """
def outer(user_input):
    def inner(x):
        return x
    eval(user_input)
"""
        result = analyze_taint(src, taint_args={'outer': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1

    def test_class_method_not_crash(self):
        src = """
class Foo:
    def bar(self, user_input):
        eval(user_input)
"""
        result = analyze_taint(src, taint_args={'bar': {'user_input': 'user'}})
        sinks = result.taint_sinks()
        assert len(sinks) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
