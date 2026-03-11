"""Tests for V099: Alias-Aware Program Slicing."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from alias_aware_slicing import (
    backward_slice, forward_slice, thin_backward_slice, chop,
    alias_query, slice_with_pta, compare_slices, get_heap_deps,
    full_slicing_analysis, slice_summary,
    build_sdg, build_pdg, SliceCriterion, SliceResult,
    DepKind, CfgNode, DepEdge, PDG, SDG,
    _flatten_stmts, _collect_vars_used, _collect_vars_defined,
    _collect_heap_reads, _collect_heap_writes, _stmt_uses,
    _prune_heap_edges_with_alias, _prune_sdg_with_alias,
)


# =========================================================================
# Section 1: AST Helpers
# =========================================================================

class TestASTHelpers:
    """Test AST traversal helpers."""

    def test_collect_vars_from_var(self):
        from hash_maps import lex, Parser
        tokens = lex("let x = y;")
        p = Parser(tokens)
        prog = p.parse()
        stmt = prog.stmts[0]
        assert _collect_vars_used(stmt.value) == {'y'}

    def test_collect_vars_from_binop(self):
        from hash_maps import lex, Parser
        tokens = lex("let z = x + y;")
        p = Parser(tokens)
        prog = p.parse()
        stmt = prog.stmts[0]
        assert _collect_vars_used(stmt.value) == {'x', 'y'}

    def test_collect_vars_defined_let(self):
        from hash_maps import lex, Parser
        tokens = lex("let x = 1;")
        p = Parser(tokens)
        prog = p.parse()
        assert _collect_vars_defined(prog.stmts[0]) == {'x'}

    def test_collect_vars_defined_assign(self):
        from hash_maps import lex, Parser
        tokens = lex("let x = 0; x = 1;")
        p = Parser(tokens)
        prog = p.parse()
        assert _collect_vars_defined(prog.stmts[1]) == {'x'}

    def test_collect_heap_reads(self):
        from hash_maps import lex, Parser
        tokens = lex('let x = h["key"];')
        p = Parser(tokens)
        prog = p.parse()
        reads = _collect_heap_reads(prog.stmts[0].value)
        assert ('h', 'key') in reads

    def test_collect_heap_writes(self):
        from hash_maps import lex, Parser
        tokens = lex('h["key"] = 42;')
        p = Parser(tokens)
        prog = p.parse()
        writes = _collect_heap_writes(prog.stmts[0])
        assert ('h', 'key') in writes

    def test_stmt_uses_return(self):
        from hash_maps import lex, Parser
        tokens = lex("fn f(x) { return x + 1; }")
        p = Parser(tokens)
        prog = p.parse()
        fn = prog.stmts[0]
        body = fn.body.stmts if hasattr(fn.body, 'stmts') else fn.body
        ret = body[0]
        assert 'x' in _stmt_uses(ret)


# =========================================================================
# Section 2: Flatten Statements
# =========================================================================

class TestFlattenStmts:
    """Test statement flattening for CFG construction."""

    def test_flat_sequence(self):
        from hash_maps import lex, Parser
        tokens = lex("let x = 1; let y = 2; let z = x + y;")
        p = Parser(tokens)
        prog = p.parse()
        flat = _flatten_stmts(prog.stmts)
        assert len(flat) == 3
        assert flat[0][0] == 0  # first index
        assert flat[2][0] == 2  # third index

    def test_flat_if(self):
        from hash_maps import lex, Parser
        tokens = lex("let x = 1; if (x > 0) { let y = 1; } let z = 2;")
        p = Parser(tokens)
        prog = p.parse()
        flat = _flatten_stmts(prog.stmts)
        # x=1, if, y=1, z=2
        assert len(flat) == 4
        # y=1 should have if's index in its cond_indices
        y_stmt = flat[2]  # y=1 is third
        assert y_stmt[2] == [1]  # controlled by if at index 1

    def test_flat_while(self):
        from hash_maps import lex, Parser
        tokens = lex("let i = 0; while (i < 10) { i = i + 1; }")
        p = Parser(tokens)
        prog = p.parse()
        flat = _flatten_stmts(prog.stmts)
        assert len(flat) == 3
        # i=i+1 controlled by while
        assert flat[2][2] == [1]

    def test_flat_nested_if(self):
        from hash_maps import lex, Parser
        tokens = lex("if (a) { if (b) { let c = 1; } }")
        p = Parser(tokens)
        prog = p.parse()
        flat = _flatten_stmts(prog.stmts)
        # if(a), if(b), c=1
        assert len(flat) == 3
        assert flat[2][2] == [0, 1]  # controlled by both


# =========================================================================
# Section 3: PDG Construction
# =========================================================================

class TestPDGConstruction:
    """Test Program Dependence Graph construction."""

    def test_basic_pdg(self):
        from hash_maps import lex, Parser
        tokens = lex("let x = 1; let y = x + 2; let z = y;")
        p = Parser(tokens)
        prog = p.parse()
        pdg = build_pdg(prog.stmts)
        assert pdg.func_name == "__main__"
        assert len(pdg.nodes) >= 5  # entry + 3 stmts + exit

    def test_pdg_data_edges(self):
        from hash_maps import lex, Parser
        tokens = lex("let x = 1; let y = x;")
        p = Parser(tokens)
        prog = p.parse()
        pdg = build_pdg(prog.stmts)
        data_edges = [e for e in pdg.edges if e.kind == DepKind.DATA]
        # y uses x, should have data edge from x=1 to y=x
        vars_in_data = {e.var for e in data_edges}
        assert 'x' in vars_in_data

    def test_pdg_control_edges(self):
        from hash_maps import lex, Parser
        tokens = lex("let x = 1; if (x > 0) { let y = 1; }")
        p = Parser(tokens)
        prog = p.parse()
        pdg = build_pdg(prog.stmts)
        ctrl_edges = [e for e in pdg.edges if e.kind == DepKind.CONTROL]
        assert len(ctrl_edges) > 0

    def test_pdg_heap_edges(self):
        from hash_maps import lex, Parser
        tokens = lex('let h = {"a": 1}; h["a"] = 2; let v = h["a"];')
        p = Parser(tokens)
        prog = p.parse()
        pdg = build_pdg(prog.stmts)
        heap_edges = [e for e in pdg.edges if e.kind == DepKind.HEAP_DATA]
        assert len(heap_edges) > 0

    def test_pdg_with_params(self):
        pdg = build_pdg([], "__main__", params=["a", "b"])
        assert 'a' in pdg.defs[-1]
        assert 'b' in pdg.defs[-1]


# =========================================================================
# Section 4: SDG Construction
# =========================================================================

class TestSDGConstruction:
    """Test System Dependence Graph construction."""

    def test_basic_sdg(self):
        source = """
let x = 1;
let y = x + 2;
"""
        sdg = build_sdg(source)
        assert "__main__" in sdg.pdgs
        assert len(sdg.all_nodes) > 0

    def test_sdg_with_functions(self):
        source = """
fn add(a, b) {
    return a + b;
}
let result = add(3, 4);
"""
        sdg = build_sdg(source)
        assert "add" in sdg.pdgs
        assert "__main__" in sdg.pdgs

    def test_sdg_interprocedural_edges(self):
        source = """
fn double(x) {
    return x * 2;
}
let r = double(5);
"""
        sdg = build_sdg(source)
        call_edges = [e for e in sdg.inter_edges if e.kind == DepKind.CALL]
        assert len(call_edges) > 0

    def test_sdg_get_node(self):
        source = "let x = 1; let y = 2;"
        sdg = build_sdg(source)
        node = sdg.get_node("__main__", 0)
        assert node is not None
        assert node.index == 0


# =========================================================================
# Section 5: Basic Backward Slicing
# =========================================================================

class TestBackwardSlicing:
    """Test backward slicing without alias awareness."""

    def test_simple_chain(self):
        source = "let x = 1; let y = x + 2; let z = y * 3;"
        crit = SliceCriterion(node_index=2, variables={'y'}, func="__main__")
        result = backward_slice(source, crit, alias_aware=False)
        assert result.size > 0
        assert 2 in result.indices  # criterion itself

    def test_independent_variable(self):
        source = "let a = 1; let b = 2; let c = a;"
        crit = SliceCriterion(node_index=2, variables={'a'}, func="__main__")
        result = backward_slice(source, crit, alias_aware=False)
        # Should include a=1 and c=a, but not b=2
        assert 0 in result.indices  # a=1
        assert 2 in result.indices  # c=a

    def test_control_dependency(self):
        source = "let x = 1; if (x > 0) { let y = 1; }"
        crit = SliceCriterion(node_index=2, func="__main__")
        result = backward_slice(source, crit, alias_aware=False)
        # y=1 depends on if, which depends on x
        assert 1 in result.indices  # if statement

    def test_empty_criterion(self):
        source = "let x = 1;"
        crit = SliceCriterion(node_index=99, func="__main__")
        result = backward_slice(source, crit, alias_aware=False)
        assert result.size == 0


# =========================================================================
# Section 6: Basic Forward Slicing
# =========================================================================

class TestForwardSlicing:
    """Test forward slicing without alias awareness."""

    def test_simple_forward(self):
        source = "let x = 1; let y = x; let z = y;"
        crit = SliceCriterion(node_index=0, func="__main__")
        result = forward_slice(source, crit, alias_aware=False)
        assert result.size >= 2  # x=1 affects y=x and z=y

    def test_no_forward_deps(self):
        source = "let a = 1; let b = 2; let c = 3;"
        crit = SliceCriterion(node_index=2, func="__main__")
        result = forward_slice(source, crit, alias_aware=False)
        assert result.size == 1  # just itself

    def test_forward_through_condition(self):
        source = "let x = 1; if (x > 0) { let y = 2; }"
        crit = SliceCriterion(node_index=0, func="__main__")
        result = forward_slice(source, crit, alias_aware=False)
        assert result.size >= 2  # x flows to condition and body


# =========================================================================
# Section 7: Alias-Aware Backward Slicing
# =========================================================================

class TestAliasAwareBackward:
    """Test backward slicing with alias-aware heap analysis."""

    def test_no_alias_prune(self):
        """Two independent hashes: writes to h1 shouldn't affect reads from h2."""
        source = """
let h1 = {"x": 1};
let h2 = {"x": 2};
h1["x"] = 10;
let v = h2["x"];
"""
        crit = SliceCriterion(node_index=3, variables={'v'}, func="__main__")
        result = backward_slice(source, crit, alias_aware=True)
        # Alias-aware should know h1 and h2 don't alias
        assert result.alias_aware

    def test_alias_keeps_deps(self):
        """When variables do alias, heap deps are preserved."""
        source = """
let h = {"x": 1};
let ref = h;
ref["x"] = 42;
let v = h["x"];
"""
        crit = SliceCriterion(node_index=3, func="__main__")
        result = backward_slice(source, crit, alias_aware=True)
        assert result.size > 0

    def test_precision_gain_independent_hashes(self):
        """Alias-aware should produce smaller or equal slice for independent heaps."""
        source = """
let a = {"val": 1};
let b = {"val": 2};
a["val"] = 100;
let r = b["val"];
"""
        crit = SliceCriterion(node_index=3, func="__main__")
        aware = backward_slice(source, crit, alias_aware=True)
        conservative = backward_slice(source, crit, alias_aware=False)
        assert aware.size <= conservative.size

    def test_array_no_alias(self):
        """Two independent arrays."""
        source = """
let arr1 = [1, 2, 3];
let arr2 = [4, 5, 6];
arr1[0] = 99;
let v = arr2[0];
"""
        crit = SliceCriterion(node_index=3, func="__main__")
        aware = backward_slice(source, crit, alias_aware=True)
        conservative = backward_slice(source, crit, alias_aware=False)
        assert aware.size <= conservative.size


# =========================================================================
# Section 8: Alias-Aware Forward Slicing
# =========================================================================

class TestAliasAwareForward:
    """Test forward slicing with alias awareness."""

    def test_forward_alias_aware(self):
        source = """
let h = {"a": 1};
let other = {"a": 2};
h["a"] = 10;
let v1 = h["a"];
let v2 = other["a"];
"""
        crit = SliceCriterion(node_index=2, func="__main__")
        result = forward_slice(source, crit, alias_aware=True)
        assert result.alias_aware

    def test_forward_conservative_vs_aware(self):
        source = """
let x = {"k": 0};
let y = {"k": 0};
x["k"] = 5;
let a = x["k"];
let b = y["k"];
"""
        crit = SliceCriterion(node_index=2, func="__main__")
        aware = forward_slice(source, crit, alias_aware=True)
        conservative = forward_slice(source, crit, alias_aware=False)
        assert aware.size <= conservative.size


# =========================================================================
# Section 9: Thin Backward Slicing
# =========================================================================

class TestThinBackwardSlicing:
    """Test thin backward slicing (data-only, no control deps)."""

    def test_thin_vs_full(self):
        source = "let x = 1; if (x > 0) { let y = x; } let z = 2;"
        crit = SliceCriterion(node_index=2, func="__main__")
        thin = thin_backward_slice(source, crit, alias_aware=False)
        full = backward_slice(source, crit, alias_aware=False)
        assert thin.size <= full.size
        assert thin.direction == "thin_backward"

    def test_thin_alias_aware(self):
        source = """
let a = {"v": 1};
let b = {"v": 2};
a["v"] = 10;
let r = b["v"];
"""
        crit = SliceCriterion(node_index=3, func="__main__")
        result = thin_backward_slice(source, crit, alias_aware=True)
        assert result.alias_aware
        assert result.direction == "thin_backward"


# =========================================================================
# Section 10: Chop Slicing
# =========================================================================

class TestChop:
    """Test chop slicing (intersection of forward and backward)."""

    def test_basic_chop(self):
        source = "let x = 1; let y = x; let z = y; let w = z;"
        src_crit = SliceCriterion(node_index=0, func="__main__")
        tgt_crit = SliceCriterion(node_index=3, func="__main__")
        result = chop(source, src_crit, tgt_crit, alias_aware=False)
        assert result.direction == "chop"
        assert result.size >= 2  # at least source and target

    def test_chop_no_path(self):
        source = "let a = 1; let b = 2;"
        src_crit = SliceCriterion(node_index=0, func="__main__")
        tgt_crit = SliceCriterion(node_index=1, func="__main__")
        result = chop(source, src_crit, tgt_crit, alias_aware=False)
        # a and b are independent, chop should have no connecting path
        # (only overlapping nodes from forward(a) and backward(b))
        assert result.direction == "chop"

    def test_chop_alias_aware(self):
        source = """
let h1 = {"x": 1};
let h2 = {"x": 2};
h1["x"] = 10;
let v = h2["x"];
"""
        src_crit = SliceCriterion(node_index=2, func="__main__")
        tgt_crit = SliceCriterion(node_index=3, func="__main__")
        result = chop(source, src_crit, tgt_crit, alias_aware=True)
        assert result.alias_aware


# =========================================================================
# Section 11: Interprocedural Slicing
# =========================================================================

class TestInterproceduralSlicing:
    """Test slicing across function boundaries."""

    def test_backward_through_call(self):
        source = """
fn inc(x) {
    return x + 1;
}
let a = 5;
let b = inc(a);
"""
        crit = SliceCriterion(node_index=1, func="__main__")
        result = backward_slice(source, crit, alias_aware=False,
                              interprocedural=True)
        assert len(result.functions_involved) >= 1

    def test_intraprocedural_only(self):
        source = """
fn inc(x) {
    return x + 1;
}
let a = 5;
let b = inc(a);
"""
        crit = SliceCriterion(node_index=1, func="__main__")
        intra = backward_slice(source, crit, alias_aware=False,
                             interprocedural=False)
        inter = backward_slice(source, crit, alias_aware=False,
                             interprocedural=True)
        assert intra.size <= inter.size

    def test_forward_through_call(self):
        source = """
fn double(x) {
    return x * 2;
}
let a = 3;
let b = double(a);
"""
        crit = SliceCriterion(node_index=0, func="__main__")
        result = forward_slice(source, crit, alias_aware=False,
                             interprocedural=True)
        assert result.size >= 1


# =========================================================================
# Section 12: Alias Query
# =========================================================================

class TestAliasQuery:
    """Test direct alias queries."""

    def test_same_variable_aliases(self):
        source = """
let h = {"x": 1};
let ref = h;
"""
        result = alias_query(source, "h", "ref")
        assert result.may_alias

    def test_different_allocs_no_alias(self):
        source = """
let a = {"x": 1};
let b = {"x": 2};
"""
        result = alias_query(source, "a", "b")
        assert not result.may_alias

    def test_alias_through_function(self):
        source = """
fn identity(x) {
    return x;
}
let h = {"k": 1};
let ref = identity(h);
"""
        result = alias_query(source, "h", "ref")
        # May or may not alias depending on PTA precision
        assert isinstance(result.may_alias, bool)


# =========================================================================
# Section 13: Compare Slices
# =========================================================================

class TestCompareSlices:
    """Test comparison API."""

    def test_compare_backward(self):
        source = """
let a = {"v": 1};
let b = {"v": 2};
a["v"] = 10;
let r = b["v"];
"""
        crit = SliceCriterion(node_index=3, func="__main__")
        comp = compare_slices(source, crit, direction="backward")
        assert 'alias_aware' in comp
        assert 'conservative' in comp
        assert 'precision_gain' in comp
        assert comp['aware_size'] <= comp['conservative_size']

    def test_compare_forward(self):
        source = """
let x = {"a": 1};
let y = {"a": 2};
x["a"] = 99;
let p = x["a"];
let q = y["a"];
"""
        crit = SliceCriterion(node_index=2, func="__main__")
        comp = compare_slices(source, crit, direction="forward")
        assert comp['aware_size'] <= comp['conservative_size']

    def test_compare_thin(self):
        source = "let x = 1; let y = x; let z = y;"
        crit = SliceCriterion(node_index=2, func="__main__")
        comp = compare_slices(source, crit, direction="thin_backward")
        assert 'precision_gain' in comp


# =========================================================================
# Section 14: Heap Dependencies
# =========================================================================

class TestHeapDeps:
    """Test heap dependency analysis."""

    def test_heap_deps_basic(self):
        source = """
let h = {"a": 1, "b": 2};
h["a"] = 10;
let v = h["b"];
"""
        deps = get_heap_deps(source, alias_aware=True)
        assert 'heap_writes' in deps
        assert 'heap_reads' in deps
        assert deps['conservative_heap_edges'] >= deps['alias_aware_heap_edges']

    def test_heap_deps_no_alias(self):
        source = """
let h1 = {"x": 1};
let h2 = {"x": 2};
h1["x"] = 10;
let v = h2["x"];
"""
        deps = get_heap_deps(source, alias_aware=True)
        assert deps['conservative_heap_edges'] >= deps['alias_aware_heap_edges']

    def test_heap_deps_conservative(self):
        source = """
let h = {"k": 1};
h["k"] = 2;
let v = h["k"];
"""
        deps = get_heap_deps(source, alias_aware=False)
        assert deps['conservative_heap_edges'] >= 0


# =========================================================================
# Section 15: Full Slicing Analysis
# =========================================================================

class TestFullAnalysis:
    """Test the comprehensive analysis API."""

    def test_full_analysis(self):
        source = """
let a = {"x": 1};
let b = {"x": 2};
a["x"] = 10;
let r = b["x"];
"""
        crit = SliceCriterion(node_index=3, func="__main__")
        result = full_slicing_analysis(source, crit)
        assert 'backward_aware' in result
        assert 'backward_conservative' in result
        assert 'forward_aware' in result
        assert 'thin_backward_aware' in result
        assert result['backward_aware_size'] <= result['backward_conservative_size']

    def test_full_analysis_no_heap(self):
        source = "let x = 1; let y = x + 2; let z = y * 3;"
        crit = SliceCriterion(node_index=2, func="__main__")
        result = full_slicing_analysis(source, crit)
        # No heap, so alias-aware == conservative
        assert result['backward_aware_size'] == result['backward_conservative_size']


# =========================================================================
# Section 16: Slice Summary
# =========================================================================

class TestSliceSummary:
    """Test human-readable summary."""

    def test_summary_output(self):
        source = "let x = 1; let y = x; let z = y;"
        crit = SliceCriterion(node_index=2, func="__main__")
        summary = slice_summary(source, crit)
        assert "Alias-Aware Slicing Summary" in summary
        assert "Backward slice" in summary
        assert "Precision gain" in summary

    def test_summary_with_heap(self):
        source = """
let h = {"a": 1};
h["a"] = 10;
let v = h["a"];
"""
        crit = SliceCriterion(node_index=2, func="__main__")
        summary = slice_summary(source, crit)
        assert "alias-aware" in summary


# =========================================================================
# Section 17: SliceResult Properties
# =========================================================================

class TestSliceResultProperties:
    """Test SliceResult computed properties."""

    def test_size(self):
        result = SliceResult(
            criterion=SliceCriterion(0),
            direction="backward",
            nodes={CfgNode("f", 0, "stmt"), CfgNode("f", 1, "stmt")},
            edges=[], functions_involved={"f"}, alias_aware=False
        )
        assert result.size == 2

    def test_indices(self):
        result = SliceResult(
            criterion=SliceCriterion(0),
            direction="backward",
            nodes={CfgNode("f", 0, "stmt"), CfgNode("f", 2, "stmt")},
            edges=[], functions_involved={"f"}, alias_aware=False
        )
        assert result.indices == {0, 2}

    def test_precision_gain_zero(self):
        result = SliceResult(
            criterion=SliceCriterion(0),
            direction="backward",
            nodes={CfgNode("f", 0, "stmt")},
            edges=[], functions_involved={"f"}, alias_aware=True,
            conservative_size=1
        )
        assert result.precision_gain == 0.0

    def test_precision_gain_positive(self):
        result = SliceResult(
            criterion=SliceCriterion(0),
            direction="backward",
            nodes={CfgNode("f", 0, "stmt")},
            edges=[], functions_involved={"f"}, alias_aware=True,
            conservative_size=4
        )
        assert result.precision_gain == 0.75


# =========================================================================
# Section 18: Slice With PTA Convenience
# =========================================================================

class TestSliceWithPTA:
    """Test convenience slice_with_pta API."""

    def test_backward_via_convenience(self):
        source = "let x = 1; let y = x;"
        crit = SliceCriterion(node_index=1, func="__main__")
        result = slice_with_pta(source, crit, direction="backward")
        assert result.alias_aware
        assert result.size >= 1

    def test_forward_via_convenience(self):
        source = "let x = 1; let y = x;"
        crit = SliceCriterion(node_index=0, func="__main__")
        result = slice_with_pta(source, crit, direction="forward")
        assert result.alias_aware

    def test_thin_via_convenience(self):
        source = "let x = 1; let y = x;"
        crit = SliceCriterion(node_index=1, func="__main__")
        result = slice_with_pta(source, crit, direction="thin_backward")
        assert result.direction == "thin_backward"


# =========================================================================
# Section 19: Context Sensitivity
# =========================================================================

class TestContextSensitivity:
    """Test varying context depth for points-to analysis."""

    def test_k0_vs_k1(self):
        source = """
let a = {"v": 1};
let b = {"v": 2};
a["v"] = 10;
let r = b["v"];
"""
        crit = SliceCriterion(node_index=3, func="__main__")
        k0 = backward_slice(source, crit, alias_aware=True, k=0)
        k1 = backward_slice(source, crit, alias_aware=True, k=1)
        assert k0.alias_aware
        assert k1.alias_aware
        # Higher k should be at least as precise
        assert k1.size <= k0.size or k1.size == k0.size


# =========================================================================
# Section 20: Edge Cases
# =========================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_program(self):
        source = ""
        sdg = build_sdg(source)
        assert "__main__" in sdg.pdgs

    def test_single_statement(self):
        source = "let x = 42;"
        crit = SliceCriterion(node_index=0, func="__main__")
        result = backward_slice(source, crit, alias_aware=False)
        assert result.size >= 1

    def test_nonexistent_function(self):
        source = "let x = 1;"
        crit = SliceCriterion(node_index=0, func="nonexistent")
        result = backward_slice(source, crit, alias_aware=False)
        assert result.size == 0

    def test_heap_deps_nonexistent_func(self):
        source = "let x = 1;"
        deps = get_heap_deps(source, func="nonexistent")
        assert 'error' in deps

    def test_precision_gain_zero_conservative(self):
        result = SliceResult(
            criterion=SliceCriterion(0),
            direction="backward",
            nodes=set(), edges=[], functions_involved=set(),
            alias_aware=True, conservative_size=0
        )
        assert result.precision_gain == 0.0


# =========================================================================
# Section 21: Complex Programs
# =========================================================================

class TestComplexPrograms:
    """Test with more realistic programs."""

    def test_multiple_hash_operations(self):
        source = """
let config = {"port": 8080, "host": "localhost"};
let cache = {"size": 100};
config["port"] = 3000;
cache["size"] = 200;
let p = config["port"];
let s = cache["size"];
"""
        # Slice for cache["size"] read
        crit = SliceCriterion(node_index=5, func="__main__")
        comp = compare_slices(source, crit, direction="backward")
        assert comp['aware_size'] <= comp['conservative_size']

    def test_function_with_heap(self):
        source = """
fn set_val(h, v) {
    h["data"] = v;
}
let obj = {"data": 0};
set_val(obj, 42);
let result = obj["data"];
"""
        crit = SliceCriterion(node_index=2, func="__main__")
        result = backward_slice(source, crit, alias_aware=True)
        assert result.size >= 1

    def test_chain_of_assignments(self):
        source = """
let a = 1;
let b = a;
let c = b;
let d = c;
let e = d;
"""
        crit = SliceCriterion(node_index=4, func="__main__")
        result = backward_slice(source, crit, alias_aware=False)
        assert 0 in result.indices  # a=1 should be in slice

    def test_mixed_heap_and_scalar(self):
        source = """
let x = 10;
let h = {"val": x};
let y = h["val"];
let z = y + 1;
"""
        crit = SliceCriterion(node_index=3, func="__main__")
        result = backward_slice(source, crit, alias_aware=True)
        assert result.size >= 2


# =========================================================================
# Section 22: SDG Node Lookup
# =========================================================================

class TestSDGNodeLookup:
    """Test SDG node lookup functionality."""

    def test_get_existing_node(self):
        source = "let x = 1; let y = 2; let z = 3;"
        sdg = build_sdg(source)
        for i in range(3):
            node = sdg.get_node("__main__", i)
            assert node is not None
            assert node.index == i

    def test_get_nonexistent_node(self):
        source = "let x = 1;"
        sdg = build_sdg(source)
        node = sdg.get_node("__main__", 99)
        assert node is None

    def test_get_node_in_function(self):
        source = """
fn f(x) {
    return x + 1;
}
"""
        sdg = build_sdg(source)
        node = sdg.get_node("f", 0)
        assert node is not None
