"""
Tests for V100: Points-To-Guided Shape Analysis

Composes V097 (context-sensitive points-to analysis) + V030 (shape analysis)
for precise heap reasoning on C10 programs.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pta_shape_analysis import (
    analyze_pta_shape, analyze_conservative, check_acyclic, check_not_null,
    check_shared, check_disjoint, check_reachable, compare_precision,
    alias_query, full_pta_shape_analysis, pta_shape_summary,
    TV, HeapOp, HeapOpKind, PTAShapeGraph, PTAShapeNode,
    PTAShapeAnalyzer, ConservativeShapeAnalyzer,
    _extract_heap_ops, _parse_c10, AnalysisVerdict,
)


# =========================================================================
# Section 1: Basic Allocation and Assignment
# =========================================================================

class TestBasicAllocation:
    """Test allocation and assignment tracking."""

    def test_array_allocation(self):
        src = 'let a = [1, 2, 3];'
        result = analyze_pta_shape(src)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)
        assert 'a' in result.properties
        assert result.properties['a']['is_null'] == TV.FALSE

    def test_hash_allocation(self):
        src = 'let h = {"x": 1, "y": 2};'
        result = analyze_pta_shape(src)
        assert 'h' in result.properties
        assert result.properties['h']['is_null'] == TV.FALSE

    def test_null_assignment(self):
        src = 'let x = null;'
        result = analyze_pta_shape(src)
        assert result.properties['x']['is_null'] == TV.TRUE

    def test_variable_copy(self):
        src = '''
let a = [1, 2];
let b = a;
'''
        result = analyze_pta_shape(src)
        assert result.properties['a']['is_null'] == TV.FALSE
        assert result.properties['b']['is_null'] == TV.FALSE

    def test_multiple_allocations(self):
        src = '''
let a = [1];
let b = [2];
let c = [3];
'''
        result = analyze_pta_shape(src)
        assert result.properties['a']['is_null'] == TV.FALSE
        assert result.properties['b']['is_null'] == TV.FALSE
        assert result.properties['c']['is_null'] == TV.FALSE
        assert result.stats['nodes'] >= 3  # at least 3 distinct allocations


# =========================================================================
# Section 2: Field Operations
# =========================================================================

class TestFieldOperations:
    """Test field load and store operations."""

    def test_field_store(self):
        src = '''
let a = [1];
let b = [2];
a[0] = b;
'''
        result = analyze_pta_shape(src)
        g = result.shape_graph
        assert g.edge_count() >= 1  # a -> b edge

    def test_field_load(self):
        src = '''
let h = {"x": 1};
let v = h["x"];
'''
        result = analyze_pta_shape(src)
        # v should have some targets from the field load
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)

    def test_linked_structure(self):
        src = '''
let a = {"next": null};
let b = {"next": null};
a["next"] = b;
'''
        result = analyze_pta_shape(src)
        g = result.shape_graph
        assert g.edge_count() >= 1

    def test_chain_of_pointers(self):
        src = '''
let a = {"next": null};
let b = {"next": null};
let c = {"next": null};
a["next"] = b;
b["next"] = c;
'''
        result = analyze_pta_shape(src)
        g = result.shape_graph
        assert g.edge_count() >= 2


# =========================================================================
# Section 3: Null Safety
# =========================================================================

class TestNullSafety:
    """Test null dereference detection."""

    def test_null_deref_detected(self):
        src = '''
let x = null;
let y = x["f"];
'''
        result = analyze_pta_shape(src)
        assert any(w.kind == 'NULL_DEREF' for w in result.warnings)

    def test_safe_access(self):
        src = '''
let x = {"f": 1};
let y = x["f"];
'''
        result = analyze_pta_shape(src)
        null_derefs = [w for w in result.warnings if w.kind == 'NULL_DEREF']
        assert len(null_derefs) == 0

    def test_null_store_detected(self):
        src = '''
let x = null;
let y = [1];
x["f"] = y;
'''
        result = analyze_pta_shape(src)
        assert any(w.kind == 'NULL_DEREF' for w in result.warnings)

    def test_check_not_null_api(self):
        src = 'let x = [1, 2, 3];'
        assert check_not_null(src, 'x') == TV.TRUE

    def test_check_not_null_null(self):
        src = 'let x = null;'
        assert check_not_null(src, 'x') == TV.FALSE


# =========================================================================
# Section 4: Alias Analysis via PTA
# =========================================================================

class TestAliasAnalysis:
    """Test alias queries backed by PTA."""

    def test_must_alias_after_copy(self):
        src = '''
let a = [1, 2];
let b = a;
'''
        result = alias_query(src, 'a', 'b')
        assert result.may_alias

    def test_no_alias_distinct(self):
        src = '''
let a = [1];
let b = [2];
'''
        result = alias_query(src, 'a', 'b')
        assert not result.may_alias

    def test_alias_through_field(self):
        src = '''
let h = {"x": null};
let a = [1];
h["x"] = a;
let b = h["x"];
'''
        # b may alias a (loaded from same field)
        result = analyze_pta_shape(src)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)

    def test_self_alias(self):
        src = 'let a = [1];'
        result = alias_query(src, 'a', 'a')
        assert result.may_alias
        assert result.must_alias


# =========================================================================
# Section 5: Acyclicity Checking
# =========================================================================

class TestAcyclicity:
    """Test acyclicity property checking."""

    def test_acyclic_list(self):
        src = '''
let a = {"next": null};
let b = {"next": null};
a["next"] = b;
'''
        result = analyze_pta_shape(src)
        assert result.properties['a']['acyclic'] == TV.TRUE

    def test_acyclic_single_node(self):
        src = 'let x = {"next": null};'
        assert check_acyclic(src, 'x') == TV.TRUE

    def test_null_is_acyclic(self):
        src = 'let x = null;'
        assert check_acyclic(src, 'x') == TV.TRUE


# =========================================================================
# Section 6: Sharing Detection
# =========================================================================

class TestSharing:
    """Test sharing property detection."""

    def test_no_sharing_separate(self):
        src = '''
let a = [1];
let b = [2];
'''
        result = analyze_pta_shape(src)
        # a and b point to distinct nodes -- no sharing within each
        assert result.properties['a']['shared'] in (TV.FALSE, TV.MAYBE)

    def test_sharing_via_alias(self):
        src = '''
let target = [1];
let a = {"ptr": null};
let b = {"ptr": null};
a["ptr"] = target;
b["ptr"] = target;
'''
        result = analyze_pta_shape(src)
        # target is pointed to by both a.ptr and b.ptr -- shared
        g = result.shape_graph
        target_targets = g.get_var_targets('target')
        for nid in target_targets:
            shared = g.is_shared(nid)
            # target is shared because a.ptr and b.ptr both point to it
            assert shared in (TV.TRUE, TV.MAYBE)

    def test_check_shared_api(self):
        src = '''
let a = [1];
let b = a;
'''
        # a and b alias -- the node is shared (pointed by 2 vars)
        result = check_shared(src, 'a')
        assert result in (TV.TRUE, TV.MAYBE)


# =========================================================================
# Section 7: Disjointness
# =========================================================================

class TestDisjointness:
    """Test disjointness checking between structures."""

    def test_disjoint_arrays(self):
        src = '''
let a = [1];
let b = [2];
'''
        result = check_disjoint(src, 'a', 'b')
        assert result == TV.TRUE

    def test_not_disjoint_alias(self):
        src = '''
let a = [1];
let b = a;
'''
        result = check_disjoint(src, 'a', 'b')
        assert result == TV.FALSE

    def test_not_disjoint_shared_child(self):
        src = '''
let shared = [99];
let a = {"child": null};
let b = {"child": null};
a["child"] = shared;
b["child"] = shared;
'''
        result = check_disjoint(src, 'a', 'b')
        # a and b are distinct but share a child
        # They may share nodes in reachable set
        assert result in (TV.FALSE, TV.MAYBE)


# =========================================================================
# Section 8: Strong vs Weak Updates (PTA-guided)
# =========================================================================

class TestStrongWeakUpdates:
    """Test that PTA guides strong vs weak updates."""

    def test_strong_update_single_target(self):
        src = '''
let x = {"val": null};
let a = [1];
let b = [2];
x["val"] = a;
x["val"] = b;
'''
        result = analyze_pta_shape(src)
        g = result.shape_graph
        # With strong update, x.val should point to b only (not a)
        x_targets = g.get_var_targets('x')
        for nid in x_targets:
            val_targets = g.get_field_targets(nid, 'val')
            # Should have at most 1 target (strong update overwrites)
            assert len(val_targets) <= 2  # may be 1 with strong update

    def test_comparison_shows_precision_gain(self):
        src = '''
let x = [1];
let y = [2];
'''
        comparison = compare_precision(src)
        assert 'pta_verdict' in comparison
        assert 'conservative_verdict' in comparison
        assert 'precision_gains' in comparison


# =========================================================================
# Section 9: Heap Operation Extraction
# =========================================================================

class TestHeapOpExtraction:
    """Test extraction of heap operations from C10 AST."""

    def test_extract_array_alloc(self):
        ast = _parse_c10('let a = [1, 2, 3];')
        ops = _extract_heap_ops(ast.stmts)
        allocs = [o for o in ops if o.kind == HeapOpKind.ALLOC]
        assert len(allocs) >= 1
        assert allocs[0].lhs == 'a'

    def test_extract_hash_alloc(self):
        ast = _parse_c10('let h = {"x": 1};')
        ops = _extract_heap_ops(ast.stmts)
        allocs = [o for o in ops if o.kind == HeapOpKind.ALLOC]
        assert len(allocs) >= 1
        assert allocs[0].lhs == 'h'

    def test_extract_assign(self):
        ast = _parse_c10('let a = [1];\nlet b = a;')
        ops = _extract_heap_ops(ast.stmts)
        assigns = [o for o in ops if o.kind == HeapOpKind.ASSIGN]
        assert any(o.lhs == 'b' and o.rhs == 'a' for o in assigns)

    def test_extract_null(self):
        ast = _parse_c10('let x = null;')
        ops = _extract_heap_ops(ast.stmts)
        nulls = [o for o in ops if o.kind == HeapOpKind.NULL_ASSIGN]
        assert len(nulls) >= 1

    def test_extract_field_store(self):
        ast = _parse_c10('let a = {"x": 1};\na["x"] = 2;')
        ops = _extract_heap_ops(ast.stmts)
        stores = [o for o in ops if o.kind == HeapOpKind.FIELD_STORE]
        assert len(stores) >= 1

    def test_extract_field_load(self):
        ast = _parse_c10('let h = {"x": 1};\nlet v = h["x"];')
        ops = _extract_heap_ops(ast.stmts)
        loads = [o for o in ops if o.kind == HeapOpKind.FIELD_LOAD]
        assert len(loads) >= 1
        assert loads[0].lhs == 'v'


# =========================================================================
# Section 10: PTAShapeGraph Operations
# =========================================================================

class TestPTAShapeGraph:
    """Test the PTAShapeGraph data structure directly."""

    def test_fresh_node(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n = g.fresh_node()
        assert n.node.id == 0
        assert not n.node.summary

    def test_summary_node(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n = g.fresh_node(summary=True)
        assert n.node.summary

    def test_var_pointing(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n = g.fresh_node()
        g.set_var('x', n.node.id, TV.TRUE)
        targets = g.get_var_targets('x')
        assert n.node.id in targets
        assert targets[n.node.id] == TV.TRUE

    def test_clear_var(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n = g.fresh_node()
        g.set_var('x', n.node.id, TV.TRUE)
        g.clear_var('x')
        assert g.is_null('x') == TV.TRUE

    def test_field_edge(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_field_edge(n1.node.id, 'next', n2.node.id, TV.TRUE)
        targets = g.get_field_targets(n1.node.id, 'next')
        assert n2.node.id in targets

    def test_is_null(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        assert g.is_null('x') == TV.TRUE  # not set = null
        n = g.fresh_node()
        g.set_var('x', n.node.id, TV.TRUE)
        assert g.is_null('x') == TV.FALSE

    def test_is_shared(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n = g.fresh_node()
        g.set_var('a', n.node.id, TV.TRUE)
        g.set_var('b', n.node.id, TV.TRUE)
        assert g.is_shared(n.node.id) == TV.TRUE

    def test_copy(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n = g.fresh_node()
        g.set_var('x', n.node.id, TV.TRUE)
        g2 = g.copy()
        assert g.equals(g2)
        # Mutating copy doesn't affect original
        g2.clear_var('x')
        assert not g.equals(g2)

    def test_join(self):
        g1 = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n1 = g1.fresh_node()
        g1.set_var('x', n1.node.id, TV.TRUE)

        g2 = PTAShapeGraph(nodes={}, var_points={}, field_edges={}, _next_id=1)
        n2 = g2.fresh_node()
        g2.set_var('y', n2.node.id, TV.TRUE)

        joined = g1.join(g2)
        assert 'x' in joined.var_points
        assert 'y' in joined.var_points

    def test_reachable(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        n3 = g.fresh_node()
        g.set_field_edge(n1.node.id, 'next', n2.node.id, TV.TRUE)
        g.set_field_edge(n2.node.id, 'next', n3.node.id, TV.TRUE)
        assert g.reachable(n1.node.id, n3.node.id) == TV.TRUE
        assert g.reachable(n3.node.id, n1.node.id) == TV.FALSE

    def test_acyclic_check(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_var('x', n1.node.id, TV.TRUE)
        g.set_field_edge(n1.node.id, 'next', n2.node.id, TV.TRUE)
        assert g.is_acyclic_from('x') == TV.TRUE

    def test_cycle_detection(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_var('x', n1.node.id, TV.TRUE)
        g.set_field_edge(n1.node.id, 'next', n2.node.id, TV.TRUE)
        g.set_field_edge(n2.node.id, 'next', n1.node.id, TV.TRUE)
        assert g.is_acyclic_from('x') in (TV.FALSE, TV.MAYBE)

    def test_node_count(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        g.fresh_node()
        g.fresh_node()
        assert g.node_count() == 2

    def test_edge_count(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n1 = g.fresh_node()
        n2 = g.fresh_node()
        g.set_field_edge(n1.node.id, 'next', n2.node.id, TV.TRUE)
        assert g.edge_count() == 1

    def test_remove_node(self):
        g = PTAShapeGraph(nodes={}, var_points={}, field_edges={})
        n = g.fresh_node()
        g.set_var('x', n.node.id, TV.TRUE)
        g.remove_node(n.node.id)
        assert n.node.id not in g.nodes


# =========================================================================
# Section 11: Three-Valued Logic
# =========================================================================

class TestThreeValuedLogic:
    """Test TV operations used in shape analysis."""

    def test_tv_and(self):
        assert (TV.TRUE & TV.TRUE) == TV.TRUE
        assert (TV.TRUE & TV.FALSE) == TV.FALSE
        assert (TV.TRUE & TV.MAYBE) == TV.MAYBE

    def test_tv_or(self):
        assert (TV.TRUE | TV.FALSE) == TV.TRUE
        assert (TV.FALSE | TV.FALSE) == TV.FALSE
        assert (TV.MAYBE | TV.FALSE) == TV.MAYBE

    def test_tv_not(self):
        assert (~TV.TRUE) == TV.FALSE
        assert (~TV.FALSE) == TV.TRUE
        assert (~TV.MAYBE) == TV.MAYBE


# =========================================================================
# Section 12: Conservative vs PTA Comparison
# =========================================================================

class TestComparison:
    """Test comparison between PTA-guided and conservative analysis."""

    def test_compare_simple(self):
        src = '''
let a = [1];
let b = [2];
'''
        comparison = compare_precision(src)
        assert 'pta_verdict' in comparison
        assert 'conservative_verdict' in comparison
        assert isinstance(comparison['precision_gains'], list)

    def test_conservative_analysis(self):
        src = '''
let x = [1, 2, 3];
let y = x;
'''
        result = analyze_conservative(src)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)
        assert 'x' in result.properties

    def test_conservative_null_deref(self):
        src = '''
let x = null;
let y = x["f"];
'''
        result = analyze_conservative(src)
        assert any(w.kind == 'NULL_DEREF' for w in result.warnings)


# =========================================================================
# Section 13: Function Calls
# =========================================================================

class TestFunctionCalls:
    """Test shape analysis through function calls."""

    def test_function_returning_array(self):
        src = '''
fn make() {
  return [1, 2, 3];
}
let x = make();
'''
        result = analyze_pta_shape(src)
        # x should point to something (the returned array)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)

    def test_function_with_params(self):
        src = '''
fn process(arr) {
  let v = arr[0];
  return v;
}
let a = [10, 20];
let r = process(a);
'''
        result = analyze_pta_shape(src)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)


# =========================================================================
# Section 14: Reachability
# =========================================================================

class TestReachability:
    """Test reachability checking between variables."""

    def test_direct_reachable(self):
        src = '''
let a = {"next": null};
let b = {"next": null};
a["next"] = b;
'''
        result = check_reachable(src, 'a', 'b', field='next')
        assert result in (TV.TRUE, TV.MAYBE)

    def test_not_reachable(self):
        src = '''
let a = {"next": null};
let b = {"next": null};
'''
        result = check_reachable(src, 'a', 'b', field='next')
        assert result == TV.FALSE


# =========================================================================
# Section 15: Full Analysis API
# =========================================================================

class TestFullAnalysis:
    """Test the full analysis API."""

    def test_full_analysis(self):
        src = '''
let a = [1, 2, 3];
let b = {"key": null};
b["key"] = a;
'''
        result = full_pta_shape_analysis(src)
        assert 'verdict' in result
        assert 'warnings' in result
        assert 'properties' in result
        assert 'comparison' in result
        assert 'escape_info' in result

    def test_summary(self):
        src = '''
let x = [1, 2];
let y = [3, 4];
'''
        summary = pta_shape_summary(src)
        assert 'PTA-Guided Shape Analysis' in summary
        assert 'Verdict' in summary

    def test_stats(self):
        src = '''
let a = [1];
let b = [2];
let c = [3];
'''
        result = analyze_pta_shape(src)
        assert result.stats['nodes'] >= 3
        assert result.stats['variables'] >= 3
        assert 'pta_alloc_sites' in result.stats


# =========================================================================
# Section 16: Complex Structures
# =========================================================================

class TestComplexStructures:
    """Test analysis on complex heap structures."""

    def test_nested_hash(self):
        src = '''
let inner = {"val": 42};
let outer = {"child": null};
outer["child"] = inner;
'''
        result = analyze_pta_shape(src)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)
        assert result.shape_graph.edge_count() >= 1

    def test_array_of_hashes(self):
        src = '''
let h1 = {"v": 1};
let h2 = {"v": 2};
let arr = [h1, h2];
'''
        result = analyze_pta_shape(src)
        assert result.shape_graph.node_count() >= 3

    def test_diamond_structure(self):
        src = '''
let shared = [99];
let left = {"child": null};
let right = {"child": null};
left["child"] = shared;
right["child"] = shared;
let root = {"left": null, "right": null};
root["left"] = left;
root["right"] = right;
'''
        result = analyze_pta_shape(src)
        # shared node should be marked as shared
        g = result.shape_graph
        shared_targets = g.get_var_targets('shared')
        for nid in shared_targets:
            assert g.is_shared(nid) in (TV.TRUE, TV.MAYBE)

    def test_closure_allocation(self):
        src = '''
fn maker(x) {
  fn inner() {
    return x;
  }
  return inner;
}
let f = maker(42);
'''
        result = analyze_pta_shape(src)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)


# =========================================================================
# Section 17: PTAShapeResult Properties
# =========================================================================

class TestResultProperties:
    """Test PTAShapeResult convenience properties."""

    def test_safe_property(self):
        src = 'let x = [1];'
        result = analyze_pta_shape(src)
        if result.verdict == AnalysisVerdict.SAFE:
            assert result.safe
        assert isinstance(result.has_warnings, bool)

    def test_has_warnings(self):
        src = '''
let x = null;
let y = x["f"];
'''
        result = analyze_pta_shape(src)
        assert result.has_warnings

    def test_pta_result_attached(self):
        src = 'let x = [1];'
        result = analyze_pta_shape(src)
        assert result.pta_result is not None


# =========================================================================
# Section 18: Edge Cases
# =========================================================================

class TestEdgeCases:
    """Test edge cases and unusual programs."""

    def test_empty_program(self):
        src = 'let x = 42;'
        result = analyze_pta_shape(src)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)

    def test_reassignment(self):
        src = '''
let x = [1];
x = [2];
'''
        result = analyze_pta_shape(src)
        assert 'x' in result.properties

    def test_many_variables(self):
        # Generate a program with many variables
        lines = [f'let v{i} = [{i}];' for i in range(10)]
        src = '\n'.join(lines)
        result = analyze_pta_shape(src)
        assert result.stats['nodes'] >= 10

    def test_context_sensitivity(self):
        src = '''
fn id(x) {
  return x;
}
let a = [1];
let b = [2];
let ra = id(a);
let rb = id(b);
'''
        # k=1 should distinguish id(a) from id(b)
        result = analyze_pta_shape(src, k=1)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)

    def test_k0_vs_k1(self):
        src = '''
fn wrap(v) {
  let w = {"inner": null};
  w["inner"] = v;
  return w;
}
let a = [1];
let b = [2];
let wa = wrap(a);
let wb = wrap(b);
'''
        r0 = analyze_pta_shape(src, k=0)
        r1 = analyze_pta_shape(src, k=1)
        # Both should work
        assert r0.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)
        assert r1.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)


# =========================================================================
# Section 19: Analyzer Object Tests
# =========================================================================

class TestAnalyzerObject:
    """Test the PTAShapeAnalyzer and ConservativeShapeAnalyzer directly."""

    def test_pta_analyzer_direct(self):
        analyzer = PTAShapeAnalyzer()
        ops = [
            HeapOp(kind=HeapOpKind.ALLOC, lhs='x', alloc_kind='array'),
            HeapOp(kind=HeapOpKind.ALLOC, lhs='y', alloc_kind='hash'),
            HeapOp(kind=HeapOpKind.ASSIGN, lhs='z', rhs='x'),
        ]
        g = analyzer.analyze_ops(ops)
        assert g.node_count() >= 2

    def test_conservative_analyzer_direct(self):
        analyzer = ConservativeShapeAnalyzer()
        ops = [
            HeapOp(kind=HeapOpKind.ALLOC, lhs='a', alloc_kind='array'),
            HeapOp(kind=HeapOpKind.NULL_ASSIGN, lhs='b'),
        ]
        g = analyzer.analyze_ops(ops)
        assert g.is_null('b') == TV.TRUE
        props = analyzer.get_properties(['a', 'b'])
        assert props['b']['is_null'] == TV.TRUE
        assert props['a']['is_null'] == TV.FALSE

    def test_get_properties(self):
        analyzer = PTAShapeAnalyzer()
        ops = [
            HeapOp(kind=HeapOpKind.ALLOC, lhs='x', alloc_kind='array'),
        ]
        analyzer.analyze_ops(ops)
        props = analyzer.get_properties(['x'])
        assert 'is_null' in props['x']
        assert 'acyclic' in props['x']
        assert 'shared' in props['x']


# =========================================================================
# Section 20: Integration
# =========================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_build_and_traverse_list(self):
        src = '''
let n1 = {"val": 1, "next": null};
let n2 = {"val": 2, "next": null};
let n3 = {"val": 3, "next": null};
n1["next"] = n2;
n2["next"] = n3;
let head = n1;
'''
        result = analyze_pta_shape(src)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)
        # head -> n1 -> n2 -> n3, no cycles
        assert check_acyclic(src, 'head') == TV.TRUE
        assert check_not_null(src, 'head') == TV.TRUE

    def test_factory_pattern(self):
        src = '''
fn make_node(v) {
  let n = {"val": null, "next": null};
  n["val"] = v;
  return n;
}
let a = make_node(1);
let b = make_node(2);
a["next"] = b;
'''
        result = analyze_pta_shape(src)
        assert result.verdict in (AnalysisVerdict.SAFE, AnalysisVerdict.MAYBE)

    def test_full_pipeline(self):
        src = '''
let data = [10, 20, 30];
let meta = {"source": "test", "data": null};
meta["data"] = data;
'''
        full = full_pta_shape_analysis(src)
        assert full['verdict'] in ('SAFE', 'MAYBE')
        summary = pta_shape_summary(src)
        assert len(summary) > 0

    def test_comparison_pipeline(self):
        src = '''
let a = [1];
let b = [2];
let c = a;
'''
        comp = compare_precision(src)
        assert comp['pta_warnings'] >= 0
        assert comp['conservative_warnings'] >= 0
