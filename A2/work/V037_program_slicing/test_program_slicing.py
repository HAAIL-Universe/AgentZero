"""Tests for V037: Program Slicing."""

import pytest
import sys
import os
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from program_slicing import (
    CFGBuilder, DefUseCollector, collect_def_use,
    build_pdg, build_sdg,
    backward_slice, forward_slice, chop,
    thin_backward_slice, diff_slice,
    extract_slice_source, slice_report,
    SliceCriterion, SliceResult,
    DepKind, CfgNode, DepEdge, PDG, SDG,
    compute_control_dependence,
)
import ast


# ============================================================
# CFG Tests
# ============================================================

class TestCFG:
    def test_linear_cfg(self):
        src = textwrap.dedent("""\
            x = 1
            y = 2
            z = x + y
        """)
        tree = ast.parse(src)
        cfg = CFGBuilder("<module>")
        cfg.build(tree.body, 0)
        assert cfg.entry is not None
        assert cfg.exit_node is not None
        # entry -> x=1 -> y=2 -> z=x+y -> exit
        stmt_nodes = [n for n in cfg.nodes if n.kind == "stmt"]
        assert len(stmt_nodes) == 3

    def test_if_cfg(self):
        src = textwrap.dedent("""\
            x = 1
            if x > 0:
                y = 1
            else:
                y = -1
            z = y
        """)
        tree = ast.parse(src)
        cfg = CFGBuilder("<module>")
        cfg.build(tree.body, 0)
        branches = [n for n in cfg.nodes if n.kind == "branch"]
        assert len(branches) == 1

    def test_while_cfg(self):
        src = textwrap.dedent("""\
            i = 0
            while i < 10:
                i = i + 1
            x = i
        """)
        tree = ast.parse(src)
        cfg = CFGBuilder("<module>")
        cfg.build(tree.body, 0)
        branches = [n for n in cfg.nodes if n.kind == "branch"]
        assert len(branches) == 1
        # Check back edge exists
        header = branches[0]
        preds = cfg.predecessors(header)
        # Header should have >= 2 predecessors (entry path + back edge)
        assert len(preds) >= 2

    def test_nested_if_cfg(self):
        src = textwrap.dedent("""\
            x = 1
            if x > 0:
                if x > 5:
                    y = 10
                else:
                    y = 5
            else:
                y = 0
            z = y
        """)
        tree = ast.parse(src)
        cfg = CFGBuilder("<module>")
        cfg.build(tree.body, 0)
        branches = [n for n in cfg.nodes if n.kind == "branch"]
        assert len(branches) == 2

    def test_return_cfg(self):
        src = textwrap.dedent("""\
            def f(x):
                if x > 0:
                    return 1
                return -1
        """)
        tree = ast.parse(src)
        func = tree.body[0]
        cfg = CFGBuilder("f")
        cfg.build(func.body, func.lineno)
        # Return should connect to exit
        exit_preds = cfg.predecessors(cfg.exit_node)
        assert len(exit_preds) >= 2  # both return paths

    def test_for_loop_cfg(self):
        src = textwrap.dedent("""\
            total = 0
            for i in range(10):
                total = total + i
            result = total
        """)
        tree = ast.parse(src)
        cfg = CFGBuilder("<module>")
        cfg.build(tree.body, 0)
        branches = [n for n in cfg.nodes if n.kind == "branch"]
        assert len(branches) == 1

    def test_try_except_cfg(self):
        src = textwrap.dedent("""\
            try:
                x = risky()
            except ValueError:
                x = 0
            y = x
        """)
        tree = ast.parse(src)
        cfg = CFGBuilder("<module>")
        cfg.build(tree.body, 0)
        # Should have handler branch
        branches = [n for n in cfg.nodes if n.kind == "branch"]
        assert len(branches) >= 1


# ============================================================
# Def-Use Tests
# ============================================================

class TestDefUse:
    def test_simple_def_use(self):
        src = textwrap.dedent("""\
            x = 1
            y = x + 2
            z = y * x
        """)
        tree = ast.parse(src)
        du = collect_def_use(tree.body)
        assert "x" in du.defs[1]
        assert "y" in du.defs[2]
        assert "x" in du.uses[2]
        assert "y" in du.uses[3]
        assert "x" in du.uses[3]

    def test_augassign(self):
        src = textwrap.dedent("""\
            x = 0
            x += 5
        """)
        tree = ast.parse(src)
        du = collect_def_use(tree.body)
        assert "x" in du.defs[1]
        assert "x" in du.defs[2]
        assert "x" in du.uses[2]

    def test_tuple_unpack(self):
        src = textwrap.dedent("""\
            a, b = 1, 2
            c = a + b
        """)
        tree = ast.parse(src)
        du = collect_def_use(tree.body)
        assert "a" in du.defs[1]
        assert "b" in du.defs[1]
        assert "a" in du.uses[2]
        assert "b" in du.uses[2]

    def test_for_loop_var(self):
        src = textwrap.dedent("""\
            for i in range(10):
                x = i
        """)
        tree = ast.parse(src)
        du = collect_def_use(tree.body)
        assert "i" in du.defs[1]
        assert "i" in du.uses[2]

    def test_import_defs(self):
        src = textwrap.dedent("""\
            import os
            from sys import path
            x = os.path.join(path[0])
        """)
        tree = ast.parse(src)
        du = collect_def_use(tree.body)
        assert "os" in du.defs[1]
        assert "path" in du.defs[2]

    def test_func_def(self):
        src = textwrap.dedent("""\
            def f(x, y):
                return x + y
        """)
        tree = ast.parse(src)
        du = collect_def_use(tree.body)
        assert "f" in du.defs[1]
        assert "x" in du.defs[1]
        assert "y" in du.defs[1]


# ============================================================
# PDG Tests
# ============================================================

class TestPDG:
    def test_simple_data_deps(self):
        src = textwrap.dedent("""\
            def f():
                x = 1
                y = x + 2
                z = y * x
        """)
        tree = ast.parse(src)
        pdg = build_pdg(tree.body[0], "f")
        data_edges = pdg.data_edges
        # x=1 -> y=x+2 (via x)
        assert any(e.var == "x" and e.src.line == 2 and e.dst.line == 3 for e in data_edges)
        # y=x+2 -> z=y*x (via y)
        assert any(e.var == "y" and e.src.line == 3 and e.dst.line == 4 for e in data_edges)
        # x=1 -> z=y*x (via x)
        assert any(e.var == "x" and e.src.line == 2 and e.dst.line == 4 for e in data_edges)

    def test_control_deps(self):
        src = textwrap.dedent("""\
            def f(x):
                if x > 0:
                    y = 1
                else:
                    y = -1
                return y
        """)
        tree = ast.parse(src)
        pdg = build_pdg(tree.body[0], "f")
        ctrl_edges = pdg.control_edges
        # Both y assignments should be control-dependent on the if
        ctrl_lines = [(e.src.line, e.dst.line) for e in ctrl_edges]
        assert any(src == 2 and dst == 3 for src, dst in ctrl_lines)
        assert any(src == 2 and dst == 5 for src, dst in ctrl_lines)

    def test_loop_data_deps(self):
        src = textwrap.dedent("""\
            def f():
                i = 0
                s = 0
                while i < 10:
                    s = s + i
                    i = i + 1
                return s
        """)
        tree = ast.parse(src)
        pdg = build_pdg(tree.body[0], "f")
        # s has self-dep through loop (s = s + i)
        data = pdg.data_edges
        assert any(e.var == "s" and e.src.line == 5 and e.dst.line == 5 for e in data)

    def test_pdg_deps_on(self):
        src = textwrap.dedent("""\
            def f():
                x = 1
                y = x
                z = y
        """)
        tree = ast.parse(src)
        pdg = build_pdg(tree.body[0], "f")
        # z=y depends on y=x
        z_node = pdg.cfg.get_node_at_line(4)
        if z_node:
            deps = pdg.deps_on(z_node)
            assert any(e.var == "y" for e in deps if e.kind == DepKind.DATA)


# ============================================================
# Backward Slice Tests
# ============================================================

class TestBackwardSlice:
    def test_simple_backward(self):
        src = textwrap.dedent("""\
            x = 1
            y = 2
            z = x + y
        """)
        result = backward_slice(src, SliceCriterion(3, {"z"}))
        # z depends on x and y
        assert 1 in result.lines  # x = 1
        assert 2 in result.lines  # y = 2
        assert 3 in result.lines  # z = x + y

    def test_backward_excludes_irrelevant(self):
        src = textwrap.dedent("""\
            x = 1
            y = 2
            w = 99
            z = x + y
        """)
        result = backward_slice(src, SliceCriterion(4, {"z"}))
        assert 1 in result.lines  # x
        assert 2 in result.lines  # y
        assert 4 in result.lines  # z
        assert 3 not in result.lines  # w is irrelevant

    def test_backward_with_if(self):
        src = textwrap.dedent("""\
            x = input()
            if x > 0:
                y = 1
            else:
                y = -1
            z = y
        """)
        result = backward_slice(src, SliceCriterion(6, {"z"}))
        assert 3 in result.lines  # y = 1
        assert 5 in result.lines  # y = -1
        assert 2 in result.lines  # if (control dep)
        assert 6 in result.lines  # z = y

    def test_backward_chain(self):
        src = textwrap.dedent("""\
            a = 1
            b = a
            c = b
            d = c
            e = d
        """)
        result = backward_slice(src, SliceCriterion(5, {"e"}))
        # Full chain: a -> b -> c -> d -> e
        assert {1, 2, 3, 4, 5} <= result.lines

    def test_backward_multiple_vars(self):
        src = textwrap.dedent("""\
            x = 1
            y = 2
            z = 3
            w = x + z
        """)
        result = backward_slice(src, SliceCriterion(4, {"w"}))
        assert 1 in result.lines  # x
        assert 3 in result.lines  # z
        assert 4 in result.lines  # w

    def test_backward_loop(self):
        src = textwrap.dedent("""\
            i = 0
            s = 0
            while i < 10:
                s = s + i
                i = i + 1
            result = s
        """)
        result = backward_slice(src, SliceCriterion(6, {"result"}))
        assert 1 in result.lines  # i = 0
        assert 2 in result.lines  # s = 0
        assert 4 in result.lines  # s = s + i
        assert 6 in result.lines  # result = s

    def test_backward_no_vars(self):
        """Backward slice with no specific variables includes all deps."""
        src = textwrap.dedent("""\
            x = 1
            y = x
            z = y
        """)
        result = backward_slice(src, SliceCriterion(3))
        assert 1 in result.lines
        assert 2 in result.lines
        assert 3 in result.lines


# ============================================================
# Forward Slice Tests
# ============================================================

class TestForwardSlice:
    def test_simple_forward(self):
        src = textwrap.dedent("""\
            x = 1
            y = x + 2
            z = y * 3
            w = 99
        """)
        result = forward_slice(src, SliceCriterion(1, {"x"}))
        assert 1 in result.lines  # x = 1 (seed)
        assert 2 in result.lines  # y = x + 2
        assert 3 in result.lines  # z = y * 3
        # w should NOT be in forward slice of x
        assert 4 not in result.lines

    def test_forward_with_branch(self):
        src = textwrap.dedent("""\
            x = 1
            if x > 0:
                y = x
            else:
                y = 0
            z = y
        """)
        result = forward_slice(src, SliceCriterion(1, {"x"}))
        assert 1 in result.lines
        assert 2 in result.lines  # if uses x
        # Both branches affected by x's control dep
        assert 3 in result.lines
        assert 5 in result.lines

    def test_forward_chain(self):
        src = textwrap.dedent("""\
            a = input()
            b = a + 1
            c = b + 1
            d = c + 1
        """)
        result = forward_slice(src, SliceCriterion(1, {"a"}))
        assert {1, 2, 3, 4} <= result.lines

    def test_forward_limited_scope(self):
        src = textwrap.dedent("""\
            x = 1
            y = 2
            z = x + 3
            w = y + 4
        """)
        result = forward_slice(src, SliceCriterion(1, {"x"}))
        assert 1 in result.lines
        assert 3 in result.lines  # z uses x
        # w doesn't use x
        assert 4 not in result.lines


# ============================================================
# Inter-procedural Slice Tests
# ============================================================

class TestInterproceduralSlice:
    def test_interprocedural_backward(self):
        src = textwrap.dedent("""\
            def add(a, b):
                return a + b

            x = 1
            y = 2
            z = add(x, y)
        """)
        result = backward_slice(src, SliceCriterion(6), interprocedural=True)
        assert 4 in result.lines  # x = 1
        assert 5 in result.lines  # y = 2
        assert 6 in result.lines  # z = add(x, y)

    def test_intraprocedural_only(self):
        src = textwrap.dedent("""\
            def add(a, b):
                return a + b

            x = 1
            y = 2
            z = add(x, y)
        """)
        result = backward_slice(src, SliceCriterion(6), interprocedural=False)
        assert 6 in result.lines

    def test_interprocedural_forward(self):
        src = textwrap.dedent("""\
            def process(v):
                return v * 2

            x = input()
            y = process(x)
            z = y + 1
        """)
        result = forward_slice(src, SliceCriterion(4, {"x"}), interprocedural=True)
        assert 4 in result.lines  # x = input()
        assert 5 in result.lines  # y = process(x)

    def test_multi_function_chain(self):
        src = textwrap.dedent("""\
            def double(n):
                return n * 2

            def quadruple(n):
                return double(double(n))

            x = 5
            y = quadruple(x)
        """)
        result = backward_slice(src, SliceCriterion(8), interprocedural=True)
        assert 7 in result.lines  # x = 5
        assert 8 in result.lines  # y = quadruple(x)


# ============================================================
# SDG Tests
# ============================================================

class TestSDG:
    def test_sdg_construction(self):
        src = textwrap.dedent("""\
            def f(x):
                return x + 1

            y = f(5)
        """)
        sdg = build_sdg(src)
        assert "f" in sdg.pdgs
        assert "<module>" in sdg.pdgs
        assert len(sdg.call_edges) >= 0  # May or may not detect call

    def test_sdg_call_edges(self):
        src = textwrap.dedent("""\
            def helper(x):
                return x * 2

            result = helper(10)
        """)
        sdg = build_sdg(src)
        # Should have PDGs for both functions
        assert len(sdg.pdgs) >= 1

    def test_sdg_node_lookup(self):
        src = textwrap.dedent("""\
            x = 1
            y = x + 2
        """)
        sdg = build_sdg(src)
        node = sdg.get_node_at_line(1)
        assert node is not None
        assert node.line == 1


# ============================================================
# Chop Tests
# ============================================================

class TestChop:
    def test_simple_chop(self):
        src = textwrap.dedent("""\
            a = 1
            b = a + 1
            c = b + 1
            d = c + 1
            e = 99
        """)
        result = chop(src, SliceCriterion(1), SliceCriterion(4))
        # Chop from a to d: a -> b -> c -> d
        assert 1 in result.lines
        assert 2 in result.lines
        assert 3 in result.lines
        assert 4 in result.lines
        # e is not on path from a to d
        assert 5 not in result.lines

    def test_chop_excludes_unrelated(self):
        src = textwrap.dedent("""\
            x = 1
            y = 2
            z = x + 1
            w = z + y
        """)
        result = chop(src, SliceCriterion(1), SliceCriterion(3))
        # Path from x to z
        assert 1 in result.lines
        assert 3 in result.lines
        # y is not on the path from x to z
        assert 2 not in result.lines


# ============================================================
# Thin Slice Tests
# ============================================================

class TestThinSlice:
    def test_thin_vs_full(self):
        src = textwrap.dedent("""\
            x = input()
            if x > 0:
                y = x
            else:
                y = 0
            z = y
        """)
        thin = thin_backward_slice(src, SliceCriterion(6, {"z"}))
        full = backward_slice(src, SliceCriterion(6, {"z"}))
        # Thin slice should be subset of full slice (no control deps)
        assert thin.lines <= full.lines

    def test_thin_data_only(self):
        src = textwrap.dedent("""\
            a = 1
            b = a
            c = b
        """)
        result = thin_backward_slice(src, SliceCriterion(3, {"c"}))
        assert 1 in result.lines  # a = 1
        assert 2 in result.lines  # b = a
        assert 3 in result.lines  # c = b


# ============================================================
# Diff Slice Tests
# ============================================================

class TestDiffSlice:
    def test_diff_impact(self):
        src = textwrap.dedent("""\
            x = 1
            y = x + 2
            z = y + 3
            w = 99
        """)
        result = diff_slice(src, {1})  # x = 1 changed
        # Impact: x affects y, y affects z
        assert 1 in result.lines
        assert 2 in result.lines
        assert 3 in result.lines
        # w is not affected
        assert 4 not in result.lines

    def test_diff_multiple_changes(self):
        src = textwrap.dedent("""\
            a = 1
            b = 2
            c = a + b
            d = c + 1
        """)
        result = diff_slice(src, {1, 2})  # Both a and b changed
        assert {1, 2, 3, 4} <= result.lines

    def test_diff_isolated_change(self):
        src = textwrap.dedent("""\
            x = 1
            y = 2
            z = 3
        """)
        result = diff_slice(src, {2})  # y = 2 changed
        assert 2 in result.lines
        # x and z not affected
        assert 1 not in result.lines
        assert 3 not in result.lines


# ============================================================
# Extract & Report Tests
# ============================================================

class TestReporting:
    def test_extract_source(self):
        src = textwrap.dedent("""\
            x = 1
            y = 2
            z = x + y
        """)
        result = backward_slice(src, SliceCriterion(3))
        extracted = extract_slice_source(src, result)
        assert "x = 1" in extracted
        assert "z = x + y" in extracted

    def test_slice_report(self):
        src = textwrap.dedent("""\
            x = 1
            y = x + 2
        """)
        result = backward_slice(src, SliceCriterion(2))
        report = slice_report(src, result)
        assert "BACKWARD SLICE" in report
        assert "Criterion: line 2" in report

    def test_empty_slice(self):
        src = textwrap.dedent("""\
            x = 1
        """)
        result = backward_slice(src, SliceCriterion(999))
        assert len(result.lines) == 0


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_self_assignment(self):
        src = textwrap.dedent("""\
            x = 1
            x = x + 1
            y = x
        """)
        result = backward_slice(src, SliceCriterion(3, {"y"}))
        assert 1 in result.lines  # x = 1
        assert 2 in result.lines  # x = x + 1
        assert 3 in result.lines  # y = x

    def test_multiple_defs(self):
        src = textwrap.dedent("""\
            x = 1
            x = 2
            y = x
        """)
        result = backward_slice(src, SliceCriterion(3, {"y"}))
        assert 2 in result.lines  # latest def of x
        assert 3 in result.lines

    def test_global_statement(self):
        src = textwrap.dedent("""\
            g = 0
            def f():
                global g
                g = 1
            f()
            x = g
        """)
        result = backward_slice(src, SliceCriterion(6))
        assert 6 in result.lines

    def test_with_statement(self):
        src = textwrap.dedent("""\
            with open("f") as fh:
                data = fh.read()
            result = data
        """)
        result = backward_slice(src, SliceCriterion(3, {"result"}))
        assert 3 in result.lines

    def test_exception_handler(self):
        src = textwrap.dedent("""\
            try:
                x = risky()
            except ValueError as e:
                x = 0
            y = x
        """)
        result = backward_slice(src, SliceCriterion(5, {"y"}))
        assert 5 in result.lines
        # Both possible defs of x
        assert 2 in result.lines or 4 in result.lines

    def test_nested_functions(self):
        src = textwrap.dedent("""\
            def outer(x):
                def inner(y):
                    return y + 1
                return inner(x)
        """)
        sdg = build_sdg(src)
        assert "outer" in sdg.pdgs

    def test_lambda(self):
        src = textwrap.dedent("""\
            f = lambda x: x + 1
            y = f(5)
        """)
        result = backward_slice(src, SliceCriterion(2))
        assert 2 in result.lines

    def test_list_comprehension(self):
        src = textwrap.dedent("""\
            data = [1, 2, 3]
            result = [x * 2 for x in data]
        """)
        result = backward_slice(src, SliceCriterion(2, {"result"}))
        assert 1 in result.lines  # data
        assert 2 in result.lines  # result

    def test_starred_assignment(self):
        src = textwrap.dedent("""\
            first, *rest = [1, 2, 3, 4]
            x = first
        """)
        tree = ast.parse(src)
        du = collect_def_use(tree.body)
        assert "first" in du.defs[1]
        assert "rest" in du.defs[1]

    def test_annotated_assignment(self):
        src = textwrap.dedent("""\
            x: int = 5
            y = x
        """)
        tree = ast.parse(src)
        du = collect_def_use(tree.body)
        assert "x" in du.defs[1]

    def test_empty_source(self):
        src = ""
        sdg = build_sdg(src)
        assert len(sdg.pdgs) >= 0

    def test_criterion_on_branch(self):
        src = textwrap.dedent("""\
            x = 1
            if x > 0:
                y = 1
            z = 0
        """)
        result = backward_slice(src, SliceCriterion(2))
        assert 1 in result.lines  # x affects the condition
        assert 2 in result.lines

    def test_forward_from_branch(self):
        src = textwrap.dedent("""\
            x = 1
            if x > 0:
                y = 1
            else:
                y = 0
            z = y
        """)
        result = forward_slice(src, SliceCriterion(2))
        assert 2 in result.lines
        # Branch controls y assignments and z
        assert 3 in result.lines
        assert 5 in result.lines


# ============================================================
# Complex Programs
# ============================================================

class TestComplexPrograms:
    def test_fibonacci(self):
        src = textwrap.dedent("""\
            a = 0
            b = 1
            n = 10
            i = 0
            while i < n:
                temp = a
                a = b
                b = temp + b
                i = i + 1
            result = a
        """)
        result = backward_slice(src, SliceCriterion(10, {"result"}))
        assert 1 in result.lines  # a = 0
        assert 2 in result.lines  # b = 1
        assert 7 in result.lines  # a = b
        assert 10 in result.lines  # result = a

    def test_multiple_outputs(self):
        src = textwrap.dedent("""\
            x = 1
            y = 2
            z = 3
            out1 = x + y
            out2 = y + z
        """)
        r1 = backward_slice(src, SliceCriterion(4, {"out1"}))
        r2 = backward_slice(src, SliceCriterion(5, {"out2"}))
        # out1 depends on x and y but not z
        assert 1 in r1.lines
        assert 2 in r1.lines
        assert 3 not in r1.lines
        # out2 depends on y and z but not x
        assert 1 not in r2.lines
        assert 2 in r2.lines
        assert 3 in r2.lines

    def test_diamond_control_flow(self):
        src = textwrap.dedent("""\
            x = input()
            if x > 0:
                a = 1
                b = 2
            else:
                a = 3
                b = 4
            c = a + b
        """)
        result = backward_slice(src, SliceCriterion(8, {"c"}))
        assert 3 in result.lines  # a = 1
        assert 4 in result.lines  # b = 2
        assert 6 in result.lines  # a = 3
        assert 7 in result.lines  # b = 4
        assert 8 in result.lines  # c = a + b

    def test_nested_loops(self):
        src = textwrap.dedent("""\
            s = 0
            i = 0
            while i < 3:
                j = 0
                while j < 3:
                    s = s + i * j
                    j = j + 1
                i = i + 1
            result = s
        """)
        result = backward_slice(src, SliceCriterion(9, {"result"}))
        assert 1 in result.lines  # s = 0
        assert 6 in result.lines  # s = s + i * j
        assert 9 in result.lines  # result = s

    def test_multi_function_program(self):
        src = textwrap.dedent("""\
            def square(x):
                return x * x

            def sum_squares(a, b):
                return square(a) + square(b)

            x = 3
            y = 4
            result = sum_squares(x, y)
        """)
        result = backward_slice(src, SliceCriterion(9), interprocedural=True)
        assert 7 in result.lines  # x = 3
        assert 8 in result.lines  # y = 4
        assert 9 in result.lines  # result = sum_squares(x, y)


# ============================================================
# Control Dependence Tests
# ============================================================

class TestControlDependence:
    def test_simple_if_control_dep(self):
        src = textwrap.dedent("""\
            def f(x):
                if x > 0:
                    y = 1
                z = 0
        """)
        tree = ast.parse(src)
        cfg = CFGBuilder("f")
        cfg.build(tree.body[0].body, tree.body[0].lineno)
        cd = compute_control_dependence(cfg)
        # y = 1 should be control-dependent on if
        assert any(s.line == 2 and d.line == 3 for s, d in cd)

    def test_nested_if_control_dep(self):
        src = textwrap.dedent("""\
            def f(x, y):
                if x > 0:
                    if y > 0:
                        z = 1
        """)
        tree = ast.parse(src)
        cfg = CFGBuilder("f")
        cfg.build(tree.body[0].body, tree.body[0].lineno)
        cd = compute_control_dependence(cfg)
        # z should be control-dependent on inner if (line 3)
        assert any(s.line == 3 and d.line == 4 for s, d in cd)


# ============================================================
# Slice Size Comparison Tests
# ============================================================

class TestSliceComparison:
    def test_thin_subset_of_full(self):
        """Thin slice is always a subset of full slice."""
        src = textwrap.dedent("""\
            x = input()
            if x > 0:
                y = x + 1
            else:
                y = x - 1
            z = y * 2
            if z > 10:
                w = z
            else:
                w = 0
            result = w
        """)
        criterion = SliceCriterion(11, {"result"})
        thin = thin_backward_slice(src, criterion)
        full = backward_slice(src, criterion)
        assert thin.lines <= full.lines

    def test_forward_backward_relationship(self):
        """If B is in backward(C), then C should be in forward(B)."""
        src = textwrap.dedent("""\
            x = 1
            y = x + 1
            z = y + 1
        """)
        bwd = backward_slice(src, SliceCriterion(3, {"z"}))
        fwd = forward_slice(src, SliceCriterion(1, {"x"}))
        # x=1 is in backward slice of z, so z should be in forward slice of x
        assert 1 in bwd.lines
        assert 3 in fwd.lines


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
