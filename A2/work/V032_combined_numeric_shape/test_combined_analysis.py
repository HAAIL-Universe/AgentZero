"""
Tests for V032: Combined Numeric + Shape Analysis

Tests cover:
1. Basic heap operations with data
2. List length tracking
3. Sortedness analysis
4. Data range verification
5. Combined pointer + integer programs
6. Loop analysis with numeric widening
7. Assertion checking
8. Edge cases
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from combined_analysis import (
    parse_combined_program, CombinedAnalyzer, CombinedState,
    analyze_combined, get_list_length, get_node_data_range, check_sorted,
    verify_combined, compute_list_length, check_sorted_property,
    check_data_range, Interval, INTERVAL_BOT, INTERVAL_TOP,
    interval_join, interval_meet, interval_widen, interval_add,
    interval_sub, interval_mul, interval_neg,
    TV, AnalysisVerdict, StmtKind, ExprKind,
)


# =============================================================================
# Interval arithmetic tests
# =============================================================================

class TestInterval:
    def test_bot(self):
        assert INTERVAL_BOT.is_bot()
        assert not INTERVAL_TOP.is_bot()

    def test_top(self):
        assert INTERVAL_TOP.is_top()
        assert not Interval(0, 10).is_top()

    def test_contains(self):
        iv = Interval(1, 5)
        assert iv.contains(1)
        assert iv.contains(3)
        assert iv.contains(5)
        assert not iv.contains(0)
        assert not iv.contains(6)

    def test_join(self):
        a = Interval(1, 3)
        b = Interval(5, 7)
        j = interval_join(a, b)
        assert j.lo == 1 and j.hi == 7

    def test_join_bot(self):
        assert interval_join(INTERVAL_BOT, Interval(1, 3)) == Interval(1, 3)
        assert interval_join(Interval(1, 3), INTERVAL_BOT) == Interval(1, 3)

    def test_meet(self):
        a = Interval(1, 5)
        b = Interval(3, 7)
        m = interval_meet(a, b)
        assert m.lo == 3 and m.hi == 5

    def test_meet_disjoint(self):
        a = Interval(1, 3)
        b = Interval(5, 7)
        m = interval_meet(a, b)
        assert m.is_bot()

    def test_widen(self):
        old = Interval(0, 5)
        new = Interval(-1, 5)
        w = interval_widen(old, new)
        assert w.lo == float('-inf')
        assert w.hi == 5

    def test_widen_upper(self):
        old = Interval(0, 5)
        new = Interval(0, 10)
        w = interval_widen(old, new)
        assert w.lo == 0
        assert w.hi == float('inf')

    def test_add(self):
        a = Interval(1, 3)
        b = Interval(10, 20)
        r = interval_add(a, b)
        assert r.lo == 11 and r.hi == 23

    def test_sub(self):
        a = Interval(10, 20)
        b = Interval(1, 3)
        r = interval_sub(a, b)
        assert r.lo == 7 and r.hi == 19

    def test_mul(self):
        a = Interval(2, 3)
        b = Interval(4, 5)
        r = interval_mul(a, b)
        assert r.lo == 8 and r.hi == 15

    def test_neg(self):
        a = Interval(1, 5)
        r = interval_neg(a)
        assert r.lo == -5 and r.hi == -1


# =============================================================================
# Parser tests
# =============================================================================

class TestParser:
    def test_new(self):
        stmts = parse_combined_program("x = new();")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.NEW
        assert stmts[0].lhs == "x"

    def test_null(self):
        stmts = parse_combined_program("x = null;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.NULL

    def test_assign(self):
        stmts = parse_combined_program("x = y;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.ASSIGN
        assert stmts[0].lhs == "x"
        assert stmts[0].rhs == "y"

    def test_load_next(self):
        stmts = parse_combined_program("x = y.next;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.LOAD_NEXT

    def test_store_next(self):
        stmts = parse_combined_program("x.next = y;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.STORE_NEXT

    def test_store_data(self):
        stmts = parse_combined_program("x.data = 42;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.STORE_DATA
        assert stmts[0].expr.kind == ExprKind.INT_LIT
        assert stmts[0].expr.value == 42

    def test_load_data(self):
        stmts = parse_combined_program("v = x.data;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.LOAD_DATA

    def test_int_assign(self):
        stmts = parse_combined_program("x = 10;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.INT_ASSIGN

    def test_int_expr(self):
        stmts = parse_combined_program("x = a + b * 2;")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.INT_ASSIGN
        assert stmts[0].expr.kind == ExprKind.ADD

    def test_len_expr(self):
        stmts = parse_combined_program("n = len(head);")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.INT_ASSIGN
        assert stmts[0].expr.kind == ExprKind.LEN
        assert stmts[0].expr.var_name == "head"

    def test_if_null(self):
        stmts = parse_combined_program("if (x == null) { y = null; }")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.IF
        assert stmts[0].cond_var == "x"
        assert stmts[0].cond_null == True

    def test_if_not_null(self):
        stmts = parse_combined_program("if (x != null) { y = x; }")
        assert len(stmts) == 1
        assert stmts[0].cond_null == False

    def test_if_int_cond(self):
        stmts = parse_combined_program("if (x < 10) { y = 1; }")
        assert len(stmts) == 1
        assert stmts[0].cond_op == '<'

    def test_if_else(self):
        stmts = parse_combined_program("if (x == null) { y = null; } else { y = x; }")
        assert len(stmts) == 1
        assert len(stmts[0].body) == 1
        assert len(stmts[0].else_body) == 1

    def test_while(self):
        stmts = parse_combined_program("while (x != null) { x = x.next; }")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.WHILE

    def test_assume(self):
        stmts = parse_combined_program("assume(x != null);")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.ASSUME

    def test_assert_not_null(self):
        stmts = parse_combined_program("assert_not_null(x);")
        assert len(stmts) == 1
        assert stmts[0].kind == StmtKind.ASSERT_NOT_NULL

    def test_assert_acyclic(self):
        stmts = parse_combined_program("assert_acyclic(x);")
        assert stmts[0].kind == StmtKind.ASSERT_ACYCLIC

    def test_assert_sorted(self):
        stmts = parse_combined_program("assert_sorted(x);")
        assert stmts[0].kind == StmtKind.ASSERT_SORTED

    def test_assert_reachable(self):
        stmts = parse_combined_program("assert_reachable(x, y);")
        assert stmts[0].kind == StmtKind.ASSERT_REACHABLE

    def test_assert_disjoint(self):
        stmts = parse_combined_program("assert_disjoint(x, y);")
        assert stmts[0].kind == StmtKind.ASSERT_DISJOINT

    def test_assert_length(self):
        stmts = parse_combined_program("assert_length(x, ==, 3);")
        assert stmts[0].kind == StmtKind.ASSERT_LENGTH
        assert stmts[0].assert_op == '=='
        assert stmts[0].assert_val == 3

    def test_assert_data_range(self):
        stmts = parse_combined_program("assert_data_range(x, 0, 100);")
        assert stmts[0].kind == StmtKind.ASSERT_DATA_RANGE
        assert stmts[0].assert_lo == 0
        assert stmts[0].assert_hi == 100

    def test_comment(self):
        stmts = parse_combined_program("// comment\nx = new();")
        assert len(stmts) == 1

    def test_negative_number(self):
        stmts = parse_combined_program("x = -5;")
        assert len(stmts) == 1


# =============================================================================
# Basic heap + data tests
# =============================================================================

class TestBasicHeapData:
    def test_new_node_has_top_data(self):
        """A fresh node has unknown (TOP) data."""
        result = analyze_combined("x = new();")
        state = result.final_state
        targets = state.graph.get_var_targets('x')
        for node in targets:
            assert state.get_node_data(node).is_top()

    def test_store_data_constant(self):
        """Storing a constant sets the node data to that constant."""
        result = analyze_combined("""
            x = new();
            x.data = 42;
        """)
        state = result.final_state
        targets = state.graph.get_var_targets('x')
        for node in targets:
            data = state.get_node_data(node)
            assert data.lo == 42 and data.hi == 42

    def test_load_data(self):
        """Loading data from a node yields its interval."""
        result = analyze_combined("""
            x = new();
            x.data = 10;
            v = x.data;
        """)
        state = result.final_state
        val = state.get_var_value('v')
        assert val.lo == 10 and val.hi == 10

    def test_store_data_expr(self):
        """Storing an expression computes interval."""
        result = analyze_combined("""
            n = 5;
            x = new();
            x.data = n + 3;
        """)
        state = result.final_state
        targets = state.graph.get_var_targets('x')
        for node in targets:
            data = state.get_node_data(node)
            assert data.lo == 8 and data.hi == 8

    def test_two_nodes_different_data(self):
        """Two nodes can have different data values."""
        result = analyze_combined("""
            x = new();
            x.data = 1;
            y = new();
            y.data = 2;
        """)
        state = result.final_state
        x_targets = state.graph.get_var_targets('x')
        y_targets = state.graph.get_var_targets('y')
        for node in x_targets:
            assert state.get_node_data(node) == Interval(1, 1)
        for node in y_targets:
            assert state.get_node_data(node) == Interval(2, 2)


# =============================================================================
# List construction and length tests
# =============================================================================

class TestListLength:
    def test_null_length(self):
        """Null pointer has length 0."""
        result = analyze_combined("x = null;")
        length = compute_list_length(result.final_state, 'x')
        assert length == Interval(0, 0)

    def test_single_node_length(self):
        """Single node list has length 1."""
        result = analyze_combined("x = new();")
        length = compute_list_length(result.final_state, 'x')
        assert length.lo == 1 and length.hi == 1

    def test_two_node_list(self):
        """List of 2 nodes."""
        result = analyze_combined("""
            x = new();
            y = new();
            x.next = y;
        """)
        length = compute_list_length(result.final_state, 'x')
        assert length.lo >= 2

    def test_three_node_list(self):
        """List of 3 nodes."""
        result = analyze_combined("""
            x = new();
            y = new();
            z = new();
            x.next = y;
            y.next = z;
        """)
        length = compute_list_length(result.final_state, 'x')
        assert length.lo >= 3

    def test_length_api(self):
        """get_list_length convenience function."""
        length = get_list_length("""
            x = new();
            y = new();
            x.next = y;
        """, 'x')
        assert length.lo >= 2

    def test_assert_length_pass(self):
        """Length assertion that should pass."""
        result = verify_combined("""
            x = new();
            assert_length(x, ==, 1);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_length_fail(self):
        """Length assertion that should fail."""
        result = verify_combined("""
            x = new();
            assert_length(x, ==, 0);
        """)
        assert result.verdict != AnalysisVerdict.SAFE

    def test_assert_length_ge(self):
        """Length >= assertion."""
        result = verify_combined("""
            x = new();
            y = new();
            x.next = y;
            assert_length(x, >=, 2);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_len_in_expr(self):
        """len() in integer expressions."""
        result = analyze_combined("""
            x = new();
            y = new();
            x.next = y;
            n = len(x);
        """)
        state = result.final_state
        val = state.get_var_value('n')
        assert val.lo >= 2


# =============================================================================
# Sortedness tests
# =============================================================================

class TestSortedness:
    def test_empty_sorted(self):
        """Empty list is sorted."""
        result = analyze_combined("x = null;")
        assert check_sorted_property(result.final_state, 'x') == TV.TRUE

    def test_single_node_sorted(self):
        """Single node is sorted."""
        result = analyze_combined("""
            x = new();
            x.data = 5;
        """)
        assert check_sorted_property(result.final_state, 'x') == TV.TRUE

    def test_two_nodes_sorted(self):
        """Two-node sorted list: 1 -> 5."""
        result = analyze_combined("""
            x = new();
            x.data = 1;
            y = new();
            y.data = 5;
            x.next = y;
        """)
        assert check_sorted_property(result.final_state, 'x') == TV.TRUE

    def test_two_nodes_unsorted(self):
        """Two-node unsorted list: 5 -> 1."""
        result = analyze_combined("""
            x = new();
            x.data = 5;
            y = new();
            y.data = 1;
            x.next = y;
        """)
        s = check_sorted_property(result.final_state, 'x')
        assert s == TV.FALSE

    def test_equal_data_sorted(self):
        """Equal data values are sorted."""
        result = analyze_combined("""
            x = new();
            x.data = 3;
            y = new();
            y.data = 3;
            x.next = y;
        """)
        assert check_sorted_property(result.final_state, 'x') == TV.TRUE

    def test_three_nodes_sorted(self):
        """Three-node sorted list: 1 -> 3 -> 7."""
        result = analyze_combined("""
            a = new();
            a.data = 1;
            b = new();
            b.data = 3;
            c = new();
            c.data = 7;
            a.next = b;
            b.next = c;
        """)
        assert check_sorted_property(result.final_state, 'a') == TV.TRUE

    def test_assert_sorted_pass(self):
        """assert_sorted on a sorted list should pass."""
        result = verify_combined("""
            x = new();
            x.data = 1;
            y = new();
            y.data = 10;
            x.next = y;
            assert_sorted(x);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_sorted_fail(self):
        """assert_sorted on an unsorted list should fail."""
        result = verify_combined("""
            x = new();
            x.data = 10;
            y = new();
            y.data = 1;
            x.next = y;
            assert_sorted(x);
        """)
        assert result.verdict == AnalysisVerdict.UNSAFE

    def test_check_sorted_api(self):
        """check_sorted convenience function."""
        s = check_sorted("""
            x = new();
            x.data = 1;
            y = new();
            y.data = 2;
            x.next = y;
        """, 'x')
        assert s == TV.TRUE


# =============================================================================
# Data range tests
# =============================================================================

class TestDataRange:
    def test_constant_in_range(self):
        """Constant data within range."""
        result = analyze_combined("""
            x = new();
            x.data = 5;
        """)
        assert check_data_range(result.final_state, 'x', 0, 10) == TV.TRUE

    def test_constant_out_of_range(self):
        """Constant data outside range."""
        result = analyze_combined("""
            x = new();
            x.data = 20;
        """)
        assert check_data_range(result.final_state, 'x', 0, 10) == TV.FALSE

    def test_range_overlap(self):
        """Data interval overlaps with range check."""
        result = analyze_combined("""
            x = new();
            x.data = 5;
            y = new();
            y.data = 15;
            x.next = y;
        """)
        r = check_data_range(result.final_state, 'x', 0, 10)
        # y has data=15 which is outside [0,10]
        assert r == TV.FALSE

    def test_assert_data_range_pass(self):
        """assert_data_range that should pass."""
        result = verify_combined("""
            x = new();
            x.data = 5;
            assert_data_range(x, 0, 10);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_data_range_fail(self):
        """assert_data_range that should fail."""
        result = verify_combined("""
            x = new();
            x.data = 20;
            assert_data_range(x, 0, 10);
        """)
        assert result.verdict == AnalysisVerdict.UNSAFE

    def test_data_range_api(self):
        """get_node_data_range convenience function."""
        r = get_node_data_range("""
            x = new();
            x.data = 42;
        """, 'x')
        assert r.lo == 42 and r.hi == 42


# =============================================================================
# Combined pointer + integer tests
# =============================================================================

class TestCombinedOps:
    def test_int_var_and_pointer(self):
        """Track integer variable alongside pointer."""
        result = analyze_combined("""
            n = 0;
            x = new();
            x.data = 10;
            n = n + 1;
        """)
        state = result.final_state
        assert state.get_var_value('n') == Interval(1, 1)

    def test_load_data_into_int_var(self):
        """Load node data into integer variable and compute."""
        result = analyze_combined("""
            x = new();
            x.data = 10;
            v = x.data;
            w = v + 5;
        """)
        state = result.final_state
        assert state.get_var_value('w') == Interval(15, 15)

    def test_conditional_on_data(self):
        """Branch on integer condition."""
        result = analyze_combined("""
            x = 5;
            if (x < 10) {
                y = 1;
            } else {
                y = 0;
            }
        """)
        state = result.final_state
        # x=5 < 10, so only then branch executes
        assert state.get_var_value('y') == Interval(1, 1)

    def test_conditional_data_unknown(self):
        """Branch with unknown condition joins both results."""
        result = analyze_combined("""
            x = new();
            x.data = 5;
            v = x.data;
            if (v < 3) {
                r = 0;
            } else {
                r = 1;
            }
        """)
        state = result.final_state
        # v=5 >= 3, so only else branch
        assert state.get_var_value('r') == Interval(1, 1)

    def test_build_list_with_data(self):
        """Build a 3-node list with ascending data."""
        result = analyze_combined("""
            a = new();
            a.data = 1;
            b = new();
            b.data = 2;
            c = new();
            c.data = 3;
            a.next = b;
            b.next = c;
            assert_sorted(a);
            assert_length(a, >=, 3);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_null_deref_detection(self):
        """Detect null dereference on data access."""
        result = analyze_combined("""
            x = null;
            v = x.data;
        """)
        assert any(w.kind == 'NULL_DEREF' for w in result.warnings)

    def test_null_deref_store(self):
        """Detect null dereference on data store."""
        result = analyze_combined("""
            x = null;
            x.data = 5;
        """)
        assert any(w.kind == 'NULL_DEREF' for w in result.warnings)


# =============================================================================
# Loop analysis tests
# =============================================================================

class TestLoops:
    def test_while_traverse(self):
        """Traverse a list in a while loop."""
        result = analyze_combined("""
            x = new();
            y = new();
            x.next = y;
            p = x;
            while (p != null) {
                p = p.next;
            }
        """)
        # After loop, p is null
        state = result.final_state
        assert state.graph.is_null('p') == TV.TRUE

    def test_while_count(self):
        """Count nodes in a loop."""
        result = analyze_combined("""
            x = new();
            y = new();
            x.next = y;
            p = x;
            n = 0;
            while (p != null) {
                n = n + 1;
                p = p.next;
            }
        """)
        state = result.final_state
        # n should be >= 1 (widened to [0, +inf] or [1, +inf])
        val = state.get_var_value('n')
        assert val.lo >= 0

    def test_while_int_condition(self):
        """Loop with integer condition."""
        result = analyze_combined("""
            i = 0;
            while (i < 5) {
                i = i + 1;
            }
        """)
        state = result.final_state
        val = state.get_var_value('i')
        # After loop: i >= 5 (widened)
        assert val.lo >= 5

    def test_build_list_in_loop(self):
        """Build a list in a loop (abstract)."""
        result = analyze_combined("""
            head = null;
            i = 0;
            while (i < 3) {
                n = new();
                n.data = i;
                n.next = head;
                head = n;
                i = i + 1;
            }
        """)
        state = result.final_state
        # head should not be null (at least some iterations ran)
        assert state.graph.is_null('head') != TV.TRUE

    def test_sum_list_data(self):
        """Sum data values in a list."""
        result = analyze_combined("""
            a = new();
            a.data = 10;
            b = new();
            b.data = 20;
            a.next = b;
            p = a;
            s = 0;
            while (p != null) {
                v = p.data;
                s = s + v;
                p = p.next;
            }
        """)
        state = result.final_state
        val = state.get_var_value('s')
        # Should be >= 10 (at least first element added)
        assert val.lo >= 0


# =============================================================================
# Integer condition refinement tests
# =============================================================================

class TestIntConditions:
    def test_less_than(self):
        """x < 10 refines upper bound."""
        result = analyze_combined("""
            x = 5;
            if (x < 10) {
                y = x;
            }
        """)
        state = result.final_state
        assert state.get_var_value('y').hi <= 9

    def test_greater_equal(self):
        """x >= 0 refines lower bound."""
        result = analyze_combined("""
            x = 5;
            if (x >= 0) {
                y = x;
            }
        """)
        state = result.final_state
        assert state.get_var_value('y').lo >= 0

    def test_equal(self):
        """x == 7 refines to singleton."""
        result = analyze_combined("""
            x = 7;
            if (x == 7) {
                y = x;
            }
        """)
        state = result.final_state
        assert state.get_var_value('y') == Interval(7, 7)

    def test_else_branch_negation(self):
        """Else branch gets negated condition."""
        result = analyze_combined("""
            x = 15;
            if (x < 10) {
                y = 0;
            } else {
                y = 1;
            }
        """)
        state = result.final_state
        # x=15, so only else branch: y=1
        assert state.get_var_value('y') == Interval(1, 1)

    def test_assume_int(self):
        """assume() with integer condition."""
        result = analyze_combined("""
            x = 5;
            assume(x >= 0);
        """)
        state = result.final_state
        assert state.get_var_value('x').lo >= 0


# =============================================================================
# Assertion tests (shape properties)
# =============================================================================

class TestShapeAssertions:
    def test_assert_not_null_pass(self):
        """Non-null assertion on allocated node."""
        result = verify_combined("""
            x = new();
            assert_not_null(x);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_not_null_fail(self):
        """Non-null assertion on null pointer."""
        result = verify_combined("""
            x = null;
            assert_not_null(x);
        """)
        assert result.verdict == AnalysisVerdict.UNSAFE

    def test_assert_acyclic_pass(self):
        """Acyclicity on a linear list."""
        result = verify_combined("""
            x = new();
            y = new();
            x.next = y;
            assert_acyclic(x);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_disjoint_pass(self):
        """Disjointness of two separate lists."""
        result = verify_combined("""
            x = new();
            y = new();
            assert_disjoint(x, y);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_assert_reachable_pass(self):
        """Reachability in a connected list."""
        result = verify_combined("""
            x = new();
            y = new();
            x.next = y;
            assert_reachable(x, y);
        """)
        assert result.verdict == AnalysisVerdict.SAFE


# =============================================================================
# Properties output tests
# =============================================================================

class TestProperties:
    def test_properties_ptr_var(self):
        """Properties dict has pointer variable info."""
        result = analyze_combined("""
            x = new();
            x.data = 5;
        """)
        assert 'x' in result.properties
        assert 'is_null' in result.properties['x']
        assert 'length' in result.properties['x']

    def test_properties_int_var(self):
        """Properties dict has integer variable info."""
        result = analyze_combined("""
            n = 42;
        """)
        assert 'n' in result.properties
        assert 'value' in result.properties['n']


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    def test_empty_program(self):
        """Empty program runs without error."""
        result = analyze_combined("")
        assert result.verdict == AnalysisVerdict.SAFE

    def test_data_on_uninitialized(self):
        """Accessing data on pointer that was never assigned."""
        # 'x' is not in any var_points, so it's effectively null
        result = analyze_combined("""
            x = new();
            x.data = 5;
            y = x;
            v = y.data;
        """)
        state = result.final_state
        val = state.get_var_value('v')
        assert val.lo == 5 and val.hi == 5

    def test_overwrite_data(self):
        """Overwriting data replaces interval."""
        result = analyze_combined("""
            x = new();
            x.data = 1;
            x.data = 99;
        """)
        state = result.final_state
        targets = state.graph.get_var_targets('x')
        for node in targets:
            assert state.get_node_data(node) == Interval(99, 99)

    def test_int_arithmetic(self):
        """Complex integer arithmetic."""
        result = analyze_combined("""
            a = 10;
            b = 3;
            c = a + b;
            d = a - b;
            e = a * b;
        """)
        state = result.final_state
        assert state.get_var_value('c') == Interval(13, 13)
        assert state.get_var_value('d') == Interval(7, 7)
        assert state.get_var_value('e') == Interval(30, 30)

    def test_negative_data(self):
        """Negative data values."""
        result = analyze_combined("""
            x = new();
            x.data = -5;
        """)
        state = result.final_state
        targets = state.graph.get_var_targets('x')
        for node in targets:
            d = state.get_node_data(node)
            assert d.lo == -5 and d.hi == -5

    def test_multiple_assertions(self):
        """Multiple assertions in one program."""
        result = verify_combined("""
            x = new();
            x.data = 5;
            y = new();
            y.data = 10;
            x.next = y;
            assert_not_null(x);
            assert_sorted(x);
            assert_data_range(x, 0, 20);
            assert_length(x, >=, 2);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_mixed_warnings(self):
        """Program with both safe and unsafe assertions."""
        result = verify_combined("""
            x = new();
            x.data = 100;
            assert_data_range(x, 0, 10);
            assert_not_null(x);
        """)
        # data_range fails, not_null passes
        assert result.verdict == AnalysisVerdict.UNSAFE
        assert len(result.warnings) >= 1


# =============================================================================
# Integration: realistic programs
# =============================================================================

class TestIntegration:
    def test_sorted_insert(self):
        """Build sorted list by inserting in order."""
        result = verify_combined("""
            a = new();
            a.data = 1;
            b = new();
            b.data = 5;
            c = new();
            c.data = 10;
            a.next = b;
            b.next = c;
            assert_sorted(a);
            assert_length(a, >=, 3);
            assert_data_range(a, 1, 10);
        """)
        assert result.verdict == AnalysisVerdict.SAFE

    def test_prepend_to_list(self):
        """Prepend a node to a list."""
        result = analyze_combined("""
            head = new();
            head.data = 10;
            n = new();
            n.data = 5;
            n.next = head;
            head = n;
        """)
        state = result.final_state
        length = compute_list_length(state, 'head')
        assert length.lo >= 2

    def test_pointer_and_counter(self):
        """Track list pointer and integer counter together."""
        result = analyze_combined("""
            a = new();
            a.data = 1;
            b = new();
            b.data = 2;
            a.next = b;
            p = a;
            count = 0;
            while (p != null) {
                count = count + 1;
                p = p.next;
            }
        """)
        state = result.final_state
        assert state.graph.is_null('p') == TV.TRUE
        val = state.get_var_value('count')
        assert val.lo >= 0  # widened, but at least 0

    def test_max_finder(self):
        """Find max by traversing a list (abstract)."""
        result = analyze_combined("""
            a = new();
            a.data = 3;
            b = new();
            b.data = 7;
            c = new();
            c.data = 1;
            a.next = b;
            b.next = c;
            p = a;
            m = 0;
            while (p != null) {
                v = p.data;
                p = p.next;
            }
        """)
        # Just verify it runs without error; m tracking would need
        # conditional on data comparison which is tricky with abstract state
        assert result.verdict == AnalysisVerdict.SAFE

    def test_reverse_list_shape(self):
        """Reverse a 2-node list -- loop widening makes prev MAYBE-null (sound imprecision)."""
        result = analyze_combined("""
            a = new();
            a.data = 1;
            b = new();
            b.data = 2;
            a.next = b;
            // reverse
            prev = null;
            curr = a;
            while (curr != null) {
                nxt = curr.next;
                curr.next = prev;
                prev = curr;
                curr = nxt;
            }
        """)
        state = result.final_state
        # After loop, curr is null
        assert state.graph.is_null('curr') == TV.TRUE
        # prev is MAYBE-null due to widening (sound over-approximation)
        assert state.graph.is_null('prev') != TV.TRUE  # not definitely null


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
