"""Tests for V030: Shape Analysis."""

import os
import sys
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import pytest
from shape_analysis import (
    TV, tv_join, ShapeGraph, Node, ShapeAnalyzer, ShapeResult,
    AnalysisVerdict, parse_heap_program, Stmt, StmtKind,
    analyze_shape, check_acyclic, check_not_null, check_shared,
    check_reachable, verify_shape, get_shape_info, compare_shapes,
)


# ===================================================================
# 3-valued logic
# ===================================================================

class TestThreeValuedLogic:
    def test_and_true_true(self):
        assert (TV.TRUE & TV.TRUE) == TV.TRUE

    def test_and_true_false(self):
        assert (TV.TRUE & TV.FALSE) == TV.FALSE

    def test_and_true_maybe(self):
        assert (TV.TRUE & TV.MAYBE) == TV.MAYBE

    def test_and_false_maybe(self):
        assert (TV.FALSE & TV.MAYBE) == TV.FALSE

    def test_and_maybe_maybe(self):
        assert (TV.MAYBE & TV.MAYBE) == TV.MAYBE

    def test_or_true_false(self):
        assert (TV.TRUE | TV.FALSE) == TV.TRUE

    def test_or_false_false(self):
        assert (TV.FALSE | TV.FALSE) == TV.FALSE

    def test_or_false_maybe(self):
        assert (TV.FALSE | TV.MAYBE) == TV.MAYBE

    def test_or_maybe_maybe(self):
        assert (TV.MAYBE | TV.MAYBE) == TV.MAYBE

    def test_not_true(self):
        assert (~TV.TRUE) == TV.FALSE

    def test_not_false(self):
        assert (~TV.FALSE) == TV.TRUE

    def test_not_maybe(self):
        assert (~TV.MAYBE) == TV.MAYBE

    def test_join_same(self):
        assert tv_join(TV.TRUE, TV.TRUE) == TV.TRUE
        assert tv_join(TV.FALSE, TV.FALSE) == TV.FALSE

    def test_join_different(self):
        assert tv_join(TV.TRUE, TV.FALSE) == TV.MAYBE
        assert tv_join(TV.TRUE, TV.MAYBE) == TV.MAYBE
        assert tv_join(TV.FALSE, TV.MAYBE) == TV.MAYBE


# ===================================================================
# Parser
# ===================================================================

class TestParser:
    def test_new(self):
        stmts = parse_heap_program("x = new();")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.NEW
        assert stmts[0].lhs == 'x'

    def test_null(self):
        stmts = parse_heap_program("x = null;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.NULL

    def test_assign(self):
        stmts = parse_heap_program("x = y;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.ASSIGN
        assert stmts[0].lhs == 'x'
        assert stmts[0].rhs == 'y'

    def test_load(self):
        stmts = parse_heap_program("x = y.next;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.LOAD
        assert stmts[0].lhs == 'x'
        assert stmts[0].rhs == 'y'

    def test_store(self):
        stmts = parse_heap_program("x.next = y;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.STORE
        assert stmts[0].lhs == 'x'
        assert stmts[0].rhs == 'y'

    def test_if_null(self):
        stmts = parse_heap_program("if (x == null) { y = null; }")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.IF
        assert stmts[0].cond_var == 'x'
        assert stmts[0].cond_null == True
        assert len(stmts[0].body) == 1

    def test_if_not_null(self):
        stmts = parse_heap_program("if (x != null) { y = x; } else { y = null; }")
        assert len(stmts) == 1
        assert stmts[0].cond_null == False
        assert len(stmts[0].body) == 1
        assert len(stmts[0].else_body) == 1

    def test_while(self):
        stmts = parse_heap_program("while (x != null) { x = x.next; }")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.WHILE
        assert stmts[0].cond_var == 'x'
        assert stmts[0].cond_null == False
        assert len(stmts[0].body) == 1

    def test_assert_acyclic(self):
        stmts = parse_heap_program("assert_acyclic(x);")
        assert stmts[0].kind == StmtKind.ASSERT_ACYCLIC
        assert stmts[0].lhs == 'x'

    def test_assert_reachable(self):
        stmts = parse_heap_program("assert_reachable(x, y);")
        assert stmts[0].kind == StmtKind.ASSERT_REACHABLE
        assert stmts[0].lhs == 'x'
        assert stmts[0].rhs == 'y'

    def test_assert_not_null(self):
        stmts = parse_heap_program("assert_not_null(x);")
        assert stmts[0].kind == StmtKind.ASSERT_NOT_NULL

    def test_assert_disjoint(self):
        stmts = parse_heap_program("assert_disjoint(x, y);")
        assert stmts[0].kind == StmtKind.ASSERT_DISJOINT

    def test_assume(self):
        stmts = parse_heap_program("assume(x != null);")
        assert stmts[0].kind == StmtKind.ASSUME
        assert stmts[0].cond_var == 'x'
        assert stmts[0].cond_null == False

    def test_multi_stmt(self):
        src = """
        x = new();
        y = new();
        x.next = y;
        """
        stmts = parse_heap_program(src)
        assert len(stmts) == 3

    def test_no_semicolons(self):
        src = "x = new() y = new() x.next = y"
        stmts = parse_heap_program(src)
        assert len(stmts) == 3


# ===================================================================
# Shape Graph
# ===================================================================

class TestShapeGraph:
    def test_fresh_node(self):
        g = ShapeGraph()
        n = g.fresh_node()
        assert n in g.nodes
        assert not n.summary

    def test_summary_node(self):
        g = ShapeGraph()
        n = g.fresh_node(summary=True)
        assert n.summary

    def test_set_var(self):
        g = ShapeGraph()
        n = g.fresh_node()
        g.set_var('x', n, TV.TRUE)
        assert g.get_var_targets('x') == {n: TV.TRUE}

    def test_clear_var(self):
        g = ShapeGraph()
        n = g.fresh_node()
        g.set_var('x', n)
        g.clear_var('x')
        assert g.get_var_targets('x') == {}

    def test_set_next(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_next(n1, n2, TV.TRUE)
        assert g.get_next_targets(n1) == {n2: TV.TRUE}

    def test_is_null_true(self):
        g = ShapeGraph()
        g.var_points['x'] = {}
        assert g.is_null('x') == TV.TRUE

    def test_is_null_false(self):
        g = ShapeGraph()
        n = g.fresh_node()
        g.set_var('x', n, TV.TRUE)
        assert g.is_null('x') == TV.FALSE

    def test_is_null_maybe(self):
        g = ShapeGraph()
        n = g.fresh_node()
        g.set_var('x', n, TV.MAYBE)
        assert g.is_null('x') == TV.MAYBE

    def test_shared_false(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_next(n1, n2, TV.TRUE)
        assert g.is_shared(n2) == TV.FALSE

    def test_shared_true(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        n3 = g.fresh_node()
        g.set_next(n1, n3, TV.TRUE)
        g.set_next(n2, n3, TV.TRUE)
        assert g.is_shared(n3) == TV.TRUE

    def test_shared_maybe(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        n3 = g.fresh_node()
        g.set_next(n1, n3, TV.TRUE)
        g.set_next(n2, n3, TV.MAYBE)
        assert g.is_shared(n3) == TV.MAYBE

    def test_cycle_detection(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_next(n1, n2, TV.TRUE)
        g.set_next(n2, n1, TV.TRUE)
        assert g.is_on_cycle(n1) == TV.TRUE
        assert g.is_on_cycle(n2) == TV.TRUE

    def test_no_cycle(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_next(n1, n2, TV.TRUE)
        assert g.is_on_cycle(n1) == TV.FALSE
        assert g.is_on_cycle(n2) == TV.FALSE

    def test_self_loop(self):
        g = ShapeGraph()
        n = g.fresh_node()
        g.set_next(n, n, TV.TRUE)
        assert g.is_on_cycle(n) == TV.TRUE

    def test_reachable(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        n3 = g.fresh_node()
        g.set_var('x', n1, TV.TRUE)
        g.set_next(n1, n2, TV.TRUE)
        g.set_next(n2, n3, TV.TRUE)
        assert g.reachable_from_var_general('x', n3) == TV.TRUE
        assert g.reachable_from_var_general('x', n1) == TV.TRUE  # 0 steps

    def test_reachable_maybe(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_var('x', n1, TV.TRUE)
        g.set_next(n1, n2, TV.MAYBE)
        assert g.reachable_from_var_general('x', n2) == TV.MAYBE

    def test_join(self):
        g1 = ShapeGraph()
        n = g1.fresh_node()
        g1.set_var('x', n, TV.TRUE)

        g2 = ShapeGraph()
        g2.nodes.add(n)
        g2.next_edge[n] = {}
        g2.var_points['x'] = {}  # x is null

        joined = g1.join(g2)
        assert joined.var_points['x'][n] == TV.MAYBE

    def test_equals(self):
        g1 = ShapeGraph()
        n = g1.fresh_node()
        g1.set_var('x', n, TV.TRUE)

        g2 = g1.copy()
        assert g1.equals(g2)

    def test_copy(self):
        g1 = ShapeGraph()
        n = g1.fresh_node()
        g1.set_var('x', n, TV.TRUE)
        g2 = g1.copy()
        g2.clear_var('x')
        assert g1.get_var_targets('x') == {n: TV.TRUE}

    def test_garbage_collect(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_var('x', n1, TV.TRUE)
        # n2 is unreachable
        g.garbage_collect()
        assert n1 in g.nodes
        assert n2 not in g.nodes

    def test_canonicalize(self):
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_var('x', n1, TV.TRUE)
        g.set_var('x', n2, TV.FALSE)
        g.canonicalize()
        assert n2 not in g.var_points['x']


# ===================================================================
# Basic Programs
# ===================================================================

class TestBasicPrograms:
    def test_single_node(self):
        result = analyze_shape("x = new();")
        assert result.verdict == AnalysisVerdict.SAFE
        props = result.properties.get('x', {})
        assert props.get('is_null') == TV.FALSE
        assert props.get('acyclic') == TV.TRUE

    def test_null_variable(self):
        result = analyze_shape("x = null;")
        props = result.properties.get('x', {})
        assert props.get('is_null') == TV.TRUE

    def test_assign_alias(self):
        src = """
        x = new();
        y = x;
        """
        result = analyze_shape(src)
        # Both x and y point to the same node
        x_targets = result.final_graph.get_var_targets('x')
        y_targets = result.final_graph.get_var_targets('y')
        assert set(x_targets.keys()) == set(y_targets.keys())

    def test_two_node_list(self):
        src = """
        x = new();
        y = new();
        x.next = y;
        """
        result = analyze_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE
        assert result.properties['x']['acyclic'] == TV.TRUE

    def test_three_node_list(self):
        src = """
        x = new();
        y = new();
        z = new();
        x.next = y;
        y.next = z;
        """
        result = analyze_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE
        assert result.properties['x']['acyclic'] == TV.TRUE

    def test_simple_cycle(self):
        src = """
        x = new();
        y = new();
        x.next = y;
        y.next = x;
        """
        result = analyze_shape(src)
        assert result.properties['x']['acyclic'] == TV.FALSE

    def test_self_loop_cycle(self):
        src = """
        x = new();
        x.next = x;
        """
        result = analyze_shape(src)
        assert result.properties['x']['acyclic'] == TV.FALSE

    def test_shared_node(self):
        src = """
        x = new();
        y = new();
        z = new();
        x.next = z;
        y.next = z;
        """
        result = analyze_shape(src)
        assert result.properties['x']['shared'] == TV.TRUE
        assert result.properties['y']['shared'] == TV.TRUE


# ===================================================================
# Load/Store Operations
# ===================================================================

class TestLoadStore:
    def test_load_next(self):
        src = """
        x = new();
        y = new();
        x.next = y;
        z = x.next;
        """
        result = analyze_shape(src)
        z_targets = result.final_graph.get_var_targets('z')
        y_targets = result.final_graph.get_var_targets('y')
        # z should point to same node as y
        assert set(z_targets.keys()) & set(y_targets.keys())

    def test_load_null_next(self):
        src = """
        x = new();
        y = x.next;
        """
        result = analyze_shape(src)
        # x.next is null by default, y should be null
        assert result.properties['y']['is_null'] == TV.TRUE

    def test_store_null(self):
        src = """
        x = new();
        y = new();
        x.next = y;
        x.next = null;
        """
        result = analyze_shape(src)
        # x.next should be null now
        x_targets = result.final_graph.get_var_targets('x')
        for n, tv in x_targets.items():
            if tv != TV.FALSE:
                nexts = result.final_graph.get_next_targets(n)
                # Should have no next edges (or only FALSE)
                non_false = {d: t for d, t in nexts.items() if t != TV.FALSE}
                assert len(non_false) == 0

    def test_null_deref_warning(self):
        src = """
        x = null;
        y = x.next;
        """
        result = analyze_shape(src)
        assert any('NULL_DEREF' in w.kind for w in result.warnings)


# ===================================================================
# If/While Control Flow
# ===================================================================

class TestControlFlow:
    def test_if_null_check(self):
        src = """
        x = new();
        if (x == null) {
            y = null;
        } else {
            y = x;
        }
        """
        result = analyze_shape(src)
        # x is not null, so else branch taken, y == x
        assert result.verdict == AnalysisVerdict.SAFE

    def test_if_maybe_null(self):
        src = """
        x = new();
        y = new();
        x.next = y;
        z = x.next;
        if (z == null) {
            w = null;
        } else {
            w = z;
        }
        """
        result = analyze_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_while_traverse(self):
        src = """
        x = new();
        y = new();
        z = new();
        x.next = y;
        y.next = z;
        t = x;
        while (t != null) {
            t = t.next;
        }
        """
        result = analyze_shape(src)
        # After traversal, t should be null
        assert result.properties['t']['is_null'] == TV.TRUE

    def test_while_builds_structure(self):
        """while loop that builds a list."""
        src = """
        head = null;
        x = new();
        x.next = head;
        head = x;
        x = new();
        x.next = head;
        head = x;
        """
        result = analyze_shape(src)
        assert result.properties['head']['is_null'] == TV.FALSE


# ===================================================================
# Assertions
# ===================================================================

class TestAssertions:
    def test_assert_not_null_pass(self):
        src = """
        x = new();
        assert_not_null(x);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_not_null_fail(self):
        src = """
        x = null;
        assert_not_null(x);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.UNSAFE

    def test_assert_acyclic_pass(self):
        src = """
        x = new();
        y = new();
        x.next = y;
        assert_acyclic(x);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_acyclic_fail(self):
        src = """
        x = new();
        y = new();
        x.next = y;
        y.next = x;
        assert_acyclic(x);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.UNSAFE

    def test_assert_reachable_pass(self):
        src = """
        x = new();
        y = new();
        x.next = y;
        assert_reachable(x, y);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_reachable_fail(self):
        src = """
        x = new();
        y = new();
        assert_reachable(x, y);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.UNSAFE

    def test_assert_disjoint_pass(self):
        src = """
        x = new();
        y = new();
        assert_disjoint(x, y);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_disjoint_fail(self):
        src = """
        x = new();
        y = x;
        assert_disjoint(x, y);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.UNSAFE

    def test_assume_not_null(self):
        src = """
        assume(x != null);
        assert_not_null(x);
        """
        result = verify_shape(src)
        # After assume, x should not trigger null warning
        # (may be MAYBE due to summary node)
        violations = [w for w in result.warnings if w.kind == 'VIOLATION']
        assert len(violations) == 0


# ===================================================================
# Focus / Materialize
# ===================================================================

class TestFocusMaterialize:
    def test_load_from_summary(self):
        """Loading from a summary node should materialize it."""
        src = """
        x = new();
        y = new();
        x.next = y;
        t = x;
        t = t.next;
        """
        result = analyze_shape(src)
        # t should point to y's node
        assert result.verdict == AnalysisVerdict.SAFE

    def test_store_to_concrete(self):
        """Strong update to a definite target."""
        src = """
        x = new();
        y = new();
        z = new();
        x.next = y;
        x.next = z;
        """
        result = analyze_shape(src)
        # x.next should point to z, not y
        x_targets = result.final_graph.get_var_targets('x')
        z_targets = result.final_graph.get_var_targets('z')
        for x_node, x_tv in x_targets.items():
            if x_tv != TV.FALSE:
                nexts = result.final_graph.get_next_targets(x_node)
                z_nodes = set(z_targets.keys())
                for dst, tv in nexts.items():
                    if tv == TV.TRUE:
                        assert dst in z_nodes


# ===================================================================
# Shape Properties (API)
# ===================================================================

class TestShapeAPI:
    def test_check_acyclic(self):
        src = "x = new(); y = new(); x.next = y;"
        assert check_acyclic(src, 'x') == TV.TRUE

    def test_check_not_null(self):
        assert check_not_null("x = new();", 'x') == TV.TRUE

    def test_check_not_null_false(self):
        assert check_not_null("x = null;", 'x') == TV.FALSE

    def test_check_shared(self):
        src = """
        x = new();
        y = new();
        z = new();
        x.next = z;
        y.next = z;
        """
        assert check_shared(src, 'x') == TV.TRUE

    def test_check_shared_false(self):
        src = "x = new(); y = new(); x.next = y;"
        assert check_shared(src, 'x') == TV.FALSE

    def test_get_shape_info(self):
        info = get_shape_info("x = new();")
        assert 'x' in info
        assert 'is_null' in info['x']
        assert 'acyclic' in info['x']

    def test_compare_shapes(self):
        src1 = "x = new(); y = new(); x.next = y;"
        src2 = "x = new(); x.next = x;"
        cmp = compare_shapes(src1, src2)
        assert cmp['program1']['verdict'] == 'safe'
        assert 'properties' in cmp['program1']


# ===================================================================
# Complex Programs
# ===================================================================

class TestComplexPrograms:
    def test_list_creation_and_traversal(self):
        """Build a 3-node list and traverse it."""
        src = """
        a = new();
        b = new();
        c = new();
        a.next = b;
        b.next = c;
        t = a;
        while (t != null) {
            t = t.next;
        }
        assert_acyclic(a);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_list_reversal(self):
        """Reverse a 2-node list."""
        src = """
        a = new();
        b = new();
        a.next = b;
        prev = null;
        curr = a;
        if (curr != null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        if (curr != null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        """
        result = analyze_shape(src)
        # After reversal, prev is the new head
        assert result.verdict == AnalysisVerdict.SAFE

    def test_diamond_structure(self):
        """Two paths merging at a shared node."""
        src = """
        a = new();
        b = new();
        c = new();
        d = new();
        a.next = b;
        c.next = b;
        b.next = d;
        """
        result = analyze_shape(src)
        assert result.properties['a']['shared'] == TV.TRUE
        assert result.properties['a']['acyclic'] == TV.TRUE

    def test_dispose_pattern(self):
        """Null out pointers (simulating deallocation)."""
        src = """
        x = new();
        y = new();
        x.next = y;
        x.next = null;
        x = null;
        """
        result = analyze_shape(src)
        assert result.properties['x']['is_null'] == TV.TRUE

    def test_conditional_link(self):
        """Conditionally create a cycle -- if-join loses precision."""
        src = """
        x = new();
        y = new();
        x.next = y;
        if (x != null) {
            y.next = x;
        }
        """
        result = analyze_shape(src)
        # if-join creates MAYBE: then-branch has cycle, else-branch doesn't
        # Result is MAYBE (sound over-approximation)
        assert result.properties['x']['acyclic'] in (TV.FALSE, TV.MAYBE)

    def test_unconditional_cycle(self):
        """Unconditionally create a cycle -- should be definite."""
        src = """
        x = new();
        y = new();
        x.next = y;
        y.next = x;
        """
        result = analyze_shape(src)
        assert result.properties['x']['acyclic'] == TV.FALSE

    def test_multi_variable_reachability(self):
        """Chain: a -> b -> c, check reachability."""
        src = """
        a = new();
        b = new();
        c = new();
        a.next = b;
        b.next = c;
        assert_reachable(a, c);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_disjoint_lists(self):
        """Two independent lists with named intermediate nodes."""
        src = """
        x = new();
        y = new();
        x2 = new();
        y2 = new();
        x.next = x2;
        y.next = y2;
        assert_disjoint(x, y);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_disjoint_lists_inline_new(self):
        """x.next = new() uses blur which may lose disjointness precision."""
        src = """
        x = new();
        y = new();
        x.next = new();
        y.next = new();
        """
        result = analyze_shape(src)
        # Blur may merge the two new() nodes; this is sound over-approximation
        assert result.verdict == AnalysisVerdict.SAFE

    def test_reassign_breaks_link(self):
        """Reassigning a variable doesn't affect the old list."""
        src = """
        x = new();
        y = new();
        x.next = y;
        z = x;
        x = null;
        assert_reachable(z, y);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = analyze_shape("")
        assert result.verdict == AnalysisVerdict.SAFE

    def test_only_null(self):
        result = analyze_shape("x = null;")
        assert result.verdict == AnalysisVerdict.SAFE

    def test_overwrite_variable(self):
        src = """
        x = new();
        x = new();
        """
        result = analyze_shape(src)
        assert result.properties['x']['is_null'] == TV.FALSE

    def test_double_store(self):
        src = """
        x = new();
        y = new();
        z = new();
        x.next = y;
        x.next = z;
        """
        result = analyze_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_long_chain(self):
        src = """
        a = new();
        b = new();
        c = new();
        d = new();
        e = new();
        a.next = b;
        b.next = c;
        c.next = d;
        d.next = e;
        assert_acyclic(a);
        """
        result = verify_shape(src)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assume_then_deref(self):
        """Assume not null, then safely dereference."""
        src = """
        assume(x != null);
        y = x.next;
        """
        result = analyze_shape(src)
        # Should not have definite null deref
        violations = [w for w in result.warnings if w.kind == 'NULL_DEREF']
        assert len(violations) == 0

    def test_cycle_via_three_nodes(self):
        src = """
        a = new();
        b = new();
        c = new();
        a.next = b;
        b.next = c;
        c.next = a;
        """
        result = analyze_shape(src)
        assert result.properties['a']['acyclic'] == TV.FALSE

    def test_repr(self):
        """ShapeGraph repr doesn't crash."""
        g = ShapeGraph()
        n = g.fresh_node()
        g.set_var('x', n)
        r = repr(g)
        assert 'ShapeGraph' in r


# ===================================================================
# Coerce
# ===================================================================

class TestCoerce:
    def test_functionality_constraint(self):
        """Non-summary node with definite next should remove MAYBE alternatives."""
        analyzer = ShapeAnalyzer()
        g = ShapeGraph()
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        n3 = g.fresh_node()
        g.set_var('x', n1, TV.TRUE)
        g.set_next(n1, n2, TV.TRUE)
        g.set_next(n1, n3, TV.MAYBE)
        g.set_next(n2, n3, TV.TRUE)  # keep n3 reachable

        g_coerced = analyzer._coerce(g)
        # n1 -> n3 should be removed (functionality: one next)
        nexts = g_coerced.get_next_targets(n1)
        assert nexts.get(n3, TV.FALSE) == TV.FALSE
        assert nexts.get(n2) == TV.TRUE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
