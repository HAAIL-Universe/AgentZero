"""Tests for V035: Call Graph Analysis + Dead Code Detection."""

import pytest
import os
import sys
import textwrap

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from call_graph_analysis import (
    build_call_graph, analyze, find_dead_code, find_cycles,
    tarjan_scc, compute_max_depth, compute_fan_metrics, find_orphans,
    dependency_layers, report, analyze_graph,
    CallGraph, FuncNode, NodeKind, EdgeKind, CallSite,
    build_call_graph_from_directory,
)


# ============================================================
# Basic Graph Building
# ============================================================

class TestBasicGraphBuilding:
    def test_single_function(self):
        src = textwrap.dedent("""\
            def hello():
                print("hi")
        """)
        g = build_call_graph(src)
        assert "hello" in g.nodes
        assert g.nodes["hello"].kind == NodeKind.FUNCTION

    def test_two_functions_with_call(self):
        src = textwrap.dedent("""\
            def foo():
                bar()
            def bar():
                pass
        """)
        g = build_call_graph(src)
        assert "foo" in g.nodes
        assert "bar" in g.nodes
        assert "bar" in g.callees["foo"]
        assert "foo" in g.callers["bar"]

    def test_method_in_class(self):
        src = textwrap.dedent("""\
            class MyClass:
                def method_a(self):
                    self.method_b()
                def method_b(self):
                    pass
        """)
        g = build_call_graph(src)
        assert "MyClass.method_a" in g.nodes
        assert "MyClass.method_b" in g.nodes
        assert g.nodes["MyClass.method_a"].kind == NodeKind.METHOD
        # self.method_b() resolves to method_b
        assert "MyClass.method_b" in g.callees.get("MyClass.method_a", set())

    def test_nested_function(self):
        src = textwrap.dedent("""\
            def outer():
                def inner():
                    pass
                inner()
        """)
        g = build_call_graph(src)
        assert "outer" in g.nodes
        assert "outer.inner" in g.nodes
        assert g.nodes["outer.inner"].kind == NodeKind.NESTED
        assert "outer.inner" in g.callees["outer"]

    def test_lambda(self):
        src = textwrap.dedent("""\
            f = lambda x: x + 1
        """)
        g = build_call_graph(src)
        lambdas = [n for n in g.nodes if "lambda" in n]
        assert len(lambdas) == 1

    def test_staticmethod_classmethod(self):
        src = textwrap.dedent("""\
            class Foo:
                @staticmethod
                def static_one():
                    pass
                @classmethod
                def class_one(cls):
                    pass
        """)
        g = build_call_graph(src)
        assert g.nodes["Foo.static_one"].kind == NodeKind.STATICMETHOD
        assert g.nodes["Foo.class_one"].kind == NodeKind.CLASSMETHOD

    def test_property(self):
        src = textwrap.dedent("""\
            class Foo:
                @property
                def value(self):
                    return self._val
        """)
        g = build_call_graph(src)
        assert g.nodes["Foo.value"].kind == NodeKind.PROPERTY

    def test_async_function(self):
        src = textwrap.dedent("""\
            async def fetch():
                pass
            async def process():
                await fetch()
        """)
        g = build_call_graph(src)
        assert "fetch" in g.nodes
        assert "process" in g.nodes
        assert "fetch" in g.callees["process"]

    def test_decorator_call(self):
        src = textwrap.dedent("""\
            def my_decorator(f):
                return f
            @my_decorator
            def decorated():
                pass
        """)
        g = build_call_graph(src)
        assert "my_decorator" in g.nodes
        assert "decorated" in g.nodes
        # The decorator application is a call from module level
        edges = [e for e in g.edges if e.callee == "my_decorator"
                 and e.edge_kind == EdgeKind.DECORATOR_CALL]
        assert len(edges) >= 1

    def test_multiple_calls_same_function(self):
        src = textwrap.dedent("""\
            def helper():
                pass
            def a():
                helper()
            def b():
                helper()
            def c():
                helper()
        """)
        g = build_call_graph(src)
        assert len(g.callers["helper"]) == 3


# ============================================================
# Call Resolution
# ============================================================

class TestCallResolution:
    def test_method_call_resolution(self):
        src = textwrap.dedent("""\
            class Engine:
                def start(self):
                    self.ignite()
                def ignite(self):
                    pass
        """)
        g = build_call_graph(src)
        assert "Engine.ignite" in g.callees.get("Engine.start", set())

    def test_cross_function_call(self):
        src = textwrap.dedent("""\
            def validate(x):
                pass
            def process(data):
                validate(data)
                transform(data)
            def transform(data):
                pass
        """)
        g = build_call_graph(src)
        callees = g.callees["process"]
        assert "validate" in callees
        assert "transform" in callees

    def test_chained_calls(self):
        src = textwrap.dedent("""\
            def a():
                b()
            def b():
                c()
            def c():
                pass
        """)
        g = build_call_graph(src)
        assert "b" in g.callees["a"]
        assert "c" in g.callees["b"]
        # Transitive
        assert g.all_callees("a") == {"b", "c"}
        assert g.all_callers("c") == {"a", "b"}

    def test_conditional_call(self):
        src = textwrap.dedent("""\
            def maybe():
                pass
            def caller(x):
                if x > 0:
                    maybe()
        """)
        g = build_call_graph(src)
        edge = [e for e in g.edges if e.callee == "maybe"][0]
        assert edge.in_conditional is True

    def test_try_call(self):
        src = textwrap.dedent("""\
            def risky():
                pass
            def caller():
                try:
                    risky()
                except:
                    pass
        """)
        g = build_call_graph(src)
        edge = [e for e in g.edges if e.callee == "risky"][0]
        assert edge.in_try is True


# ============================================================
# Dead Code Detection
# ============================================================

class TestDeadCode:
    def test_no_dead_code(self):
        src = textwrap.dedent("""\
            def __init__():
                helper()
            def helper():
                pass
        """)
        g = build_call_graph(src)
        dc = find_dead_code(g)
        assert len(dc.dead_functions) == 0

    def test_simple_dead_function(self):
        src = textwrap.dedent("""\
            def __init__():
                used()
            def used():
                pass
            def dead():
                pass
        """)
        g = build_call_graph(src)
        dc = find_dead_code(g)
        dead_names = [f.name for f in dc.dead_functions]
        assert "dead" in dead_names
        assert "used" not in dead_names
        assert "__init__" not in dead_names

    def test_transitive_reachability(self):
        src = textwrap.dedent("""\
            def main():
                a()
            def a():
                b()
            def b():
                c()
            def c():
                pass
            def orphan():
                pass
        """)
        g = build_call_graph(src)
        dc = find_dead_code(g)
        dead_names = [f.name for f in dc.dead_functions]
        assert "orphan" in dead_names
        assert "a" not in dead_names
        assert "b" not in dead_names
        assert "c" not in dead_names

    def test_test_functions_are_entry_points(self):
        src = textwrap.dedent("""\
            def test_something():
                helper()
            def helper():
                pass
            def unused():
                pass
        """)
        g = build_call_graph(src)
        dc = find_dead_code(g)
        dead_names = [f.name for f in dc.dead_functions]
        assert "unused" in dead_names
        assert "test_something" not in dead_names
        assert "helper" not in dead_names

    def test_dunder_methods_not_dead(self):
        src = textwrap.dedent("""\
            class Foo:
                def __init__(self):
                    pass
                def __str__(self):
                    return "foo"
                def _private(self):
                    pass
        """)
        g = build_call_graph(src)
        dc = find_dead_code(g)
        dead_names = [f.name for f in dc.dead_functions]
        assert "Foo.__init__" not in dead_names
        assert "Foo.__str__" not in dead_names
        # _private is dead since nothing calls it
        assert "Foo._private" in dead_names

    def test_extra_entry_points(self):
        src = textwrap.dedent("""\
            def api_handler():
                process()
            def process():
                pass
            def unused():
                pass
        """)
        g = build_call_graph(src)
        dc = find_dead_code(g, extra_entry_points={"api_handler"})
        dead_names = [f.name for f in dc.dead_functions]
        assert "unused" in dead_names
        assert "api_handler" not in dead_names
        assert "process" not in dead_names

    def test_dead_lines_count(self):
        src = textwrap.dedent("""\
            def main():
                pass
            def dead_big():
                x = 1
                y = 2
                z = 3
                return x + y + z
        """)
        g = build_call_graph(src)
        dc = find_dead_code(g)
        assert dc.dead_lines > 0
        dead_fn = [f for f in dc.dead_functions if f.name == "dead_big"][0]
        assert dead_fn.num_lines == 5

    def test_callback_not_dead(self):
        src = textwrap.dedent("""\
            def main():
                map(transformer, data)
            def transformer(x):
                return x * 2
        """)
        g = build_call_graph(src)
        dc = find_dead_code(g)
        dead_names = [f.name for f in dc.dead_functions]
        assert "transformer" not in dead_names

    def test_property_not_dead(self):
        src = textwrap.dedent("""\
            class Foo:
                @property
                def value(self):
                    return 42
        """)
        g = build_call_graph(src)
        dc = find_dead_code(g)
        dead_names = [f.name for f in dc.dead_functions]
        assert "Foo.value" not in dead_names


# ============================================================
# Cycle Detection
# ============================================================

class TestCycles:
    def test_no_cycles(self):
        src = textwrap.dedent("""\
            def a():
                b()
            def b():
                c()
            def c():
                pass
        """)
        g = build_call_graph(src)
        cycles = find_cycles(g)
        assert len(cycles) == 0

    def test_direct_recursion(self):
        src = textwrap.dedent("""\
            def factorial(n):
                if n <= 1:
                    return 1
                return n * factorial(n - 1)
        """)
        g = build_call_graph(src)
        cycles = find_cycles(g)
        assert len(cycles) == 1
        assert cycles[0].is_direct_recursion
        assert cycles[0].members == ["factorial"]

    def test_mutual_recursion(self):
        src = textwrap.dedent("""\
            def is_even(n):
                if n == 0:
                    return True
                return is_odd(n - 1)
            def is_odd(n):
                if n == 0:
                    return False
                return is_even(n - 1)
        """)
        g = build_call_graph(src)
        cycles = find_cycles(g)
        mutual = [c for c in cycles if c.is_mutual_recursion]
        assert len(mutual) == 1
        assert set(mutual[0].members) == {"is_even", "is_odd"}

    def test_three_way_cycle(self):
        src = textwrap.dedent("""\
            def a():
                b()
            def b():
                c()
            def c():
                a()
        """)
        g = build_call_graph(src)
        cycles = find_cycles(g)
        mutual = [c for c in cycles if c.is_mutual_recursion]
        assert len(mutual) == 1
        assert set(mutual[0].members) == {"a", "b", "c"}


# ============================================================
# SCC Analysis
# ============================================================

class TestSCC:
    def test_no_cycles_all_singletons(self):
        src = textwrap.dedent("""\
            def a():
                b()
            def b():
                pass
        """)
        g = build_call_graph(src)
        sccs = tarjan_scc(g)
        assert all(s.size == 1 for s in sccs)
        assert all(not s.has_cycle for s in sccs)

    def test_self_loop_scc(self):
        src = textwrap.dedent("""\
            def rec():
                rec()
        """)
        g = build_call_graph(src)
        sccs = tarjan_scc(g)
        scc_with_cycle = [s for s in sccs if s.has_cycle]
        assert len(scc_with_cycle) == 1
        assert "rec" in scc_with_cycle[0].members

    def test_two_sccs(self):
        src = textwrap.dedent("""\
            def a():
                b()
            def b():
                a()
            def c():
                d()
            def d():
                c()
        """)
        g = build_call_graph(src)
        sccs = tarjan_scc(g)
        cyclic = [s for s in sccs if s.has_cycle]
        assert len(cyclic) == 2


# ============================================================
# Depth & Fan Metrics
# ============================================================

class TestMetrics:
    def test_max_depth_linear_chain(self):
        src = textwrap.dedent("""\
            def a():
                b()
            def b():
                c()
            def c():
                d()
            def d():
                pass
        """)
        g = build_call_graph(src)
        root, depth, path = compute_max_depth(g)
        assert depth == 3
        assert path == ["a", "b", "c", "d"]

    def test_max_depth_branching(self):
        src = textwrap.dedent("""\
            def root():
                left()
                right()
            def left():
                leaf()
            def right():
                pass
            def leaf():
                pass
        """)
        g = build_call_graph(src)
        root, depth, path = compute_max_depth(g)
        assert depth == 2
        assert root == "root"

    def test_fan_in(self):
        src = textwrap.dedent("""\
            def helper():
                pass
            def a():
                helper()
            def b():
                helper()
            def c():
                helper()
        """)
        g = build_call_graph(src)
        fan_in, fan_out = compute_fan_metrics(g)
        assert fan_in == ("helper", 3)

    def test_fan_out(self):
        src = textwrap.dedent("""\
            def hub():
                a()
                b()
                c()
                d()
            def a():
                pass
            def b():
                pass
            def c():
                pass
            def d():
                pass
        """)
        g = build_call_graph(src)
        fan_in, fan_out = compute_fan_metrics(g)
        assert fan_out == ("hub", 4)

    def test_orphans(self):
        src = textwrap.dedent("""\
            def connected_a():
                connected_b()
            def connected_b():
                pass
            def lonely():
                pass
        """)
        g = build_call_graph(src)
        orphans = find_orphans(g)
        assert "lonely" in orphans
        assert "connected_a" not in orphans
        assert "connected_b" not in orphans


# ============================================================
# Dependency Layers
# ============================================================

class TestDependencyLayers:
    def test_simple_layers(self):
        src = textwrap.dedent("""\
            def leaf():
                pass
            def mid():
                leaf()
            def top():
                mid()
        """)
        g = build_call_graph(src)
        layers = dependency_layers(g)
        assert len(layers) == 3
        assert "leaf" in layers[0]
        assert "mid" in layers[1]
        assert "top" in layers[2]

    def test_cycle_in_same_layer(self):
        src = textwrap.dedent("""\
            def a():
                b()
            def b():
                a()
            def c():
                pass
        """)
        g = build_call_graph(src)
        layers = dependency_layers(g)
        # a and b are in the same SCC, c is a leaf
        # Find which layer has both a and b
        for layer in layers:
            if "a" in layer and "b" in layer:
                break
        else:
            pytest.fail("a and b should be in the same layer")

    def test_diamond_dependency(self):
        src = textwrap.dedent("""\
            def top():
                left()
                right()
            def left():
                bottom()
            def right():
                bottom()
            def bottom():
                pass
        """)
        g = build_call_graph(src)
        layers = dependency_layers(g)
        # bottom=0, left+right=1, top=2
        assert "bottom" in layers[0]
        assert "left" in layers[1] or "right" in layers[1]
        assert "top" in layers[2]


# ============================================================
# Full Analysis
# ============================================================

class TestFullAnalysis:
    def test_analyze_simple(self):
        src = textwrap.dedent("""\
            def main():
                process()
            def process():
                validate()
            def validate():
                pass
            def dead_code():
                pass
        """)
        result = analyze(src)
        assert result.graph is not None
        assert len(result.dead_code.dead_functions) == 1
        assert result.dead_code.dead_functions[0].name == "dead_code"
        assert result.max_depth[1] == 2  # main -> process -> validate

    def test_analyze_with_recursion(self):
        src = textwrap.dedent("""\
            def main():
                fib(10)
            def fib(n):
                if n <= 1:
                    return n
                return fib(n-1) + fib(n-2)
        """)
        result = analyze(src)
        assert len(result.cycles) == 1
        assert result.cycles[0].is_direct_recursion

    def test_report_output(self):
        src = textwrap.dedent("""\
            def main():
                a()
            def a():
                b()
            def b():
                pass
            def unused():
                pass
        """)
        result = analyze(src)
        text = report(result)
        assert "Call Graph Analysis" in text
        assert "Dead Code" in text
        assert "unused" in text

    def test_complex_program(self):
        src = textwrap.dedent("""\
            class Parser:
                def __init__(self, tokens):
                    self.tokens = tokens
                    self.pos = 0

                def parse(self):
                    return self._expr()

                def _expr(self):
                    left = self._term()
                    return left

                def _term(self):
                    return self._factor()

                def _factor(self):
                    return self.tokens[self.pos]

                def _unused_helper(self):
                    pass

            def main():
                p = Parser([])
                p.parse()

            def dead_utility():
                pass
        """)
        result = analyze(src)
        dead_names = [f.name for f in result.dead_code.dead_functions]
        assert "Parser._unused_helper" in dead_names
        assert "dead_utility" in dead_names
        assert "Parser.parse" not in dead_names
        assert "Parser._expr" not in dead_names


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_source(self):
        g = build_call_graph("")
        assert len(g.nodes) == 0
        result = analyze("")
        assert result.dead_code.total_count == 0

    def test_no_functions(self):
        src = textwrap.dedent("""\
            x = 1
            y = x + 2
            print(y)
        """)
        g = build_call_graph(src)
        assert len(g.nodes) == 0

    def test_external_calls_ignored(self):
        src = textwrap.dedent("""\
            import os
            def work():
                os.path.exists("/tmp")
                print("hello")
        """)
        g = build_call_graph(src)
        assert "work" in g.nodes
        # os.path.exists and print are external -- not in graph nodes
        assert "print" not in g.nodes

    def test_super_call(self):
        src = textwrap.dedent("""\
            class Base:
                def method(self):
                    pass
            class Child(Base):
                def method(self):
                    super().method()
        """)
        g = build_call_graph(src)
        edges = [e for e in g.edges if e.edge_kind == EdgeKind.SUPER_CALL]
        assert len(edges) >= 1

    def test_all_callees_with_cycle(self):
        src = textwrap.dedent("""\
            def a():
                b()
            def b():
                c()
            def c():
                a()
                d()
            def d():
                pass
        """)
        g = build_call_graph(src)
        reachable = g.all_callees("a")
        assert reachable == {"b", "c", "d"}

    def test_all_callers_with_cycle(self):
        src = textwrap.dedent("""\
            def a():
                b()
            def b():
                c()
            def c():
                a()
        """)
        g = build_call_graph(src)
        callers = g.all_callers("c")
        assert callers == {"a", "b"}

    def test_multiple_classes_same_method_name(self):
        src = textwrap.dedent("""\
            class A:
                def process(self):
                    pass
            class B:
                def process(self):
                    pass
            def main():
                a = A()
                a.process()
        """)
        g = build_call_graph(src)
        # Both process methods should exist
        assert "A.process" in g.nodes
        assert "B.process" in g.nodes

    def test_deeply_nested_classes(self):
        src = textwrap.dedent("""\
            class Outer:
                class Inner:
                    def method(self):
                        pass
        """)
        g = build_call_graph(src)
        assert "Outer.Inner.method" in g.nodes

    def test_callback_detection(self):
        src = textwrap.dedent("""\
            def handler(event):
                pass
            def setup():
                register(handler)
        """)
        g = build_call_graph(src)
        assert g.nodes["handler"].is_callback_target is True


# ============================================================
# Realistic Scenarios
# ============================================================

class TestRealisticScenarios:
    def test_mvc_pattern(self):
        src = textwrap.dedent("""\
            class Model:
                def __init__(self):
                    self.data = []
                def add(self, item):
                    self.data.append(item)
                    self.notify()
                def notify(self):
                    pass

            class View:
                def render(self, data):
                    self.format_output(data)
                def format_output(self, data):
                    pass

            class Controller:
                def __init__(self):
                    self.model = Model()
                    self.view = View()
                def handle_input(self, action, data):
                    self.model.add(data)
                    self.view.render(self.model.data)

            def main():
                ctrl = Controller()
                ctrl.handle_input("add", "test")
        """)
        result = analyze(src)
        assert result.dead_code.reachable_count >= 5
        dead_names = [f.name for f in result.dead_code.dead_functions]
        assert "main" not in dead_names

    def test_plugin_system(self):
        src = textwrap.dedent("""\
            class Plugin:
                def activate(self):
                    pass
                def deactivate(self):
                    pass

            class LogPlugin(Plugin):
                def activate(self):
                    super().activate()
                    self.setup_logging()
                def setup_logging(self):
                    pass

            class CachePlugin(Plugin):
                def activate(self):
                    super().activate()
                    self.init_cache()
                def init_cache(self):
                    pass
                def unused_method(self):
                    pass

            def main():
                plugins = [LogPlugin(), CachePlugin()]
                for p in plugins:
                    p.activate()
        """)
        result = analyze(src)
        dead_names = [f.name for f in result.dead_code.dead_functions]
        assert "CachePlugin.unused_method" in dead_names

    def test_utility_library(self):
        """Many public functions, all should be considered reachable if no
        entry points exist (fallback: all top-level are entry)."""
        src = textwrap.dedent("""\
            def add(a, b):
                return a + b
            def subtract(a, b):
                return a - b
            def multiply(a, b):
                return a * b
            def divide(a, b):
                return a / b
        """)
        result = analyze(src)
        # No entry points detected -> all top-level treated as entry
        assert result.dead_code.reachable_count == 4
        assert len(result.dead_code.dead_functions) == 0


# ============================================================
# Integration with file system (basic)
# ============================================================

class TestFileIntegration:
    def test_build_from_self(self):
        """Analyze this test file itself."""
        this_file = os.path.abspath(__file__)
        from call_graph_analysis import build_call_graph_from_file
        g = build_call_graph_from_file(this_file)
        assert len(g.nodes) > 0

    def test_analyze_v035_directory(self):
        """Analyze the V035 directory itself."""
        g = build_call_graph_from_directory(_dir, exclude_tests=True)
        assert len(g.nodes) > 10  # call_graph_analysis.py has many functions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
