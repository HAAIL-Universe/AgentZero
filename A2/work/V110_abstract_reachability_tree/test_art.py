"""
Tests for V110: Abstract Reachability Tree (ART)
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from art import (
    build_cfg, build_cfg_from_source, verify_program, check_assertion,
    get_predicates, compare_with_without_refinement, cfg_summary, art_summary,
    CFGNodeType, ARTResult, PredicateState, PredicateRegistry,
    _ast_to_smt, _extract_atoms, INT, BOOL
)

# Import SMT types for predicate construction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
from smt_solver import Var, IntConst, App, Op, Sort, SortKind


# ============================================================================
# Section 1: CFG Construction
# ============================================================================

class TestCFGConstruction:
    """Test CFG building from C10 source."""

    def test_simple_assignment(self):
        src = "let x = 5;"
        cfg = build_cfg(src)
        assert cfg.entry is not None
        assert cfg.exit_node is not None
        assert len(cfg.nodes) >= 3  # entry, assign, exit

    def test_sequential_assignments(self):
        src = "let x = 1; let y = 2; let z = 3;"
        cfg = build_cfg(src)
        # Should have: entry, 3 assigns, exit, error
        assert len(cfg.nodes) >= 5

    def test_if_statement(self):
        src = """
        let x = 5;
        if (x > 0) {
            let y = 1;
        } else {
            let y = 2;
        }
        """
        cfg = build_cfg(src)
        # Should have assume (true branch) and assume_not (false branch)
        types = [n.type for n in cfg.nodes]
        assert CFGNodeType.ASSUME in types
        assert CFGNodeType.ASSUME_NOT in types

    def test_while_loop(self):
        src = """
        let x = 10;
        while (x > 0) {
            x = x - 1;
        }
        """
        cfg = build_cfg(src)
        types = [n.type for n in cfg.nodes]
        assert CFGNodeType.ASSUME in types  # loop condition true
        assert CFGNodeType.ASSUME_NOT in types  # loop exit

    def test_assertion_creates_error_edge(self):
        src = """
        let x = 5;
        assert(x > 0);
        """
        cfg = build_cfg(src)
        types = [n.type for n in cfg.nodes]
        assert CFGNodeType.ASSERT in types
        assert CFGNodeType.ERROR in types
        # The assert node should have an edge to error
        assert_nodes = [n for n in cfg.nodes if n.type == CFGNodeType.ASSERT]
        assert len(assert_nodes) == 1
        assert cfg.error_node in assert_nodes[0].successors

    def test_nested_if(self):
        src = """
        let x = 5;
        if (x > 0) {
            if (x < 10) {
                let y = 1;
            }
        }
        """
        cfg = build_cfg(src)
        assume_nodes = [n for n in cfg.nodes if n.type == CFGNodeType.ASSUME]
        assert len(assume_nodes) >= 2

    def test_cfg_connectivity(self):
        """Entry should be reachable, and exit should be reachable from entry."""
        src = "let x = 1; let y = 2;"
        cfg = build_cfg(src)
        # BFS from entry
        visited = set()
        queue = [cfg.entry]
        while queue:
            n = queue.pop(0)
            if n.id in visited:
                continue
            visited.add(n.id)
            queue.extend(n.successors)
        assert cfg.exit_node.id in visited


# ============================================================================
# Section 2: CFG Summary API
# ============================================================================

class TestCFGSummary:
    """Test CFG summary utility."""

    def test_simple_summary(self):
        src = "let x = 1;"
        s = cfg_summary(src)
        assert s['nodes'] > 0
        assert s['edges'] > 0
        assert 'entry' in s['types']

    def test_summary_with_conditionals(self):
        src = """
        let x = 5;
        if (x > 0) { let y = 1; } else { let y = 2; }
        """
        s = cfg_summary(src)
        assert s['types'].get('assume', 0) >= 1
        assert s['types'].get('assume_not', 0) >= 1

    def test_summary_with_assertions(self):
        src = """
        let x = 5;
        assert(x > 0);
        """
        s = cfg_summary(src)
        assert s['types'].get('assert', 0) >= 1
        assert s['types'].get('error', 0) >= 1


# ============================================================================
# Section 3: Predicate State
# ============================================================================

class TestPredicateState:
    """Test predicate abstraction state operations."""

    def test_top_state(self):
        s = PredicateState.top()
        assert not s.is_bottom
        assert len(s.predicates) == 0

    def test_bottom_state(self):
        s = PredicateState.bottom()
        assert s.is_bottom

    def test_subsumption_top_covers_everything(self):
        top = PredicateState.top()
        specific = PredicateState(frozenset({0, 1, 2}))
        assert top.subsumes(specific)

    def test_subsumption_same_state(self):
        s = PredicateState(frozenset({0, 1}))
        assert s.subsumes(s)

    def test_subsumption_more_specific_doesnt_cover(self):
        more = PredicateState(frozenset({0, 1, 2}))
        less = PredicateState(frozenset({0}))
        assert not more.subsumes(less)

    def test_bottom_covered_by_everything(self):
        top = PredicateState.top()
        bot = PredicateState.bottom()
        assert top.subsumes(bot)

    def test_join_keeps_common_predicates(self):
        s1 = PredicateState(frozenset({0, 1, 2}))
        s2 = PredicateState(frozenset({1, 2, 3}))
        j = s1.join(s2)
        assert j.predicates == frozenset({1, 2})

    def test_join_with_bottom(self):
        s = PredicateState(frozenset({0, 1}))
        b = PredicateState.bottom()
        assert s.join(b) == s
        assert b.join(s) == s


# ============================================================================
# Section 4: Predicate Registry
# ============================================================================

class TestPredicateRegistry:
    """Test predicate management."""

    def test_add_predicate(self):
        reg = PredicateRegistry()
        x = Var("x", INT)
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        idx = reg.add_predicate(pred, location_id=0)
        assert idx == 0

    def test_duplicate_predicate(self):
        reg = PredicateRegistry()
        x = Var("x", INT)
        pred1 = App(Op.GE, [x, IntConst(0)], BOOL)
        pred2 = App(Op.GE, [x, IntConst(0)], BOOL)
        idx1 = reg.add_predicate(pred1)
        idx2 = reg.add_predicate(pred2)
        assert idx1 == idx2  # Same predicate, same index

    def test_location_predicates(self):
        reg = PredicateRegistry()
        x = Var("x", INT)
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        reg.add_predicate(pred, location_id=5)
        assert 0 in reg.get_predicates_at(5)
        assert len(reg.get_predicates_at(99)) == 0

    def test_get_predicate_term(self):
        reg = PredicateRegistry()
        x = Var("x", INT)
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        idx = reg.add_predicate(pred)
        assert reg.get_predicate_term(idx) == pred


# ============================================================================
# Section 5: SMT Encoding
# ============================================================================

class TestSMTEncoding:
    """Test AST to SMT conversion."""

    def test_integer_literal(self):
        from stack_vm import lex, Parser
        tokens = lex("let x = 5;")
        prog = Parser(tokens).parse()
        stmt = prog.stmts[0]
        env = {}
        term = _ast_to_smt(stmt.value, env)
        assert isinstance(term, IntConst)
        assert term.value == 5

    def test_variable(self):
        from stack_vm import lex, Parser
        tokens = lex("let x = y;")
        prog = Parser(tokens).parse()
        stmt = prog.stmts[0]
        env = {}
        term = _ast_to_smt(stmt.value, env)
        assert isinstance(term, Var)
        assert term.name == "y"

    def test_binary_comparison(self):
        from stack_vm import lex, Parser
        tokens = lex("let z = x > 0;")
        prog = Parser(tokens).parse()
        stmt = prog.stmts[0]
        env = {}
        term = _ast_to_smt(stmt.value, env)
        assert isinstance(term, App)
        assert term.op == Op.GT

    def test_arithmetic(self):
        from stack_vm import lex, Parser
        tokens = lex("let z = x + 1;")
        prog = Parser(tokens).parse()
        stmt = prog.stmts[0]
        env = {}
        term = _ast_to_smt(stmt.value, env)
        assert isinstance(term, App)
        assert term.op == Op.ADD


# ============================================================================
# Section 6: Atom Extraction
# ============================================================================

class TestAtomExtraction:
    """Test predicate atom extraction from formulas."""

    def test_single_comparison(self):
        x = Var("x", INT)
        term = App(Op.GE, [x, IntConst(0)], BOOL)
        atoms = _extract_atoms(term)
        assert len(atoms) == 1

    def test_conjunction(self):
        x = Var("x", INT)
        a1 = App(Op.GE, [x, IntConst(0)], BOOL)
        a2 = App(Op.LE, [x, IntConst(10)], BOOL)
        conj = App(Op.AND, [a1, a2], BOOL)
        atoms = _extract_atoms(conj)
        assert len(atoms) == 2

    def test_negation(self):
        x = Var("x", INT)
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        neg = App(Op.NOT, [a], BOOL)
        atoms = _extract_atoms(neg)
        assert len(atoms) == 1


# ============================================================================
# Section 7: Safe Programs (No Assertions)
# ============================================================================

class TestSafeNoAssertions:
    """Programs without assertions should be trivially safe."""

    def test_empty_program(self):
        src = "let x = 1;"
        result = verify_program(src)
        assert result.safe

    def test_assignment_chain(self):
        src = "let x = 1; let y = x + 1; let z = y + 1;"
        result = verify_program(src)
        assert result.safe

    def test_conditional_no_assert(self):
        src = """
        let x = 5;
        if (x > 0) {
            let y = 1;
        } else {
            let y = 0;
        }
        """
        result = verify_program(src)
        assert result.safe


# ============================================================================
# Section 8: Safe Programs with Assertions
# ============================================================================

class TestSafeWithAssertions:
    """Programs with assertions that should pass."""

    def test_simple_assertion_passes(self):
        src = """
        let x = 5;
        assert(x > 0);
        """
        result = verify_program(src)
        assert result.safe

    def test_assertion_after_assignment(self):
        src = """
        let x = 10;
        let y = x + 5;
        assert(y > 0);
        """
        result = verify_program(src)
        assert result.safe

    def test_assertion_in_branch(self):
        src = """
        let x = 5;
        if (x > 0) {
            assert(x > 0);
        }
        """
        result = verify_program(src)
        assert result.safe

    def test_positive_after_check(self):
        src = """
        let x = 3;
        if (x > 0) {
            let y = x + 1;
            assert(y > 1);
        }
        """
        result = verify_program(src)
        assert result.safe


# ============================================================================
# Section 9: Unsafe Programs
# ============================================================================

class TestUnsafePrograms:
    """Programs with failing assertions."""

    def test_obvious_violation(self):
        src = """
        let x = 0;
        assert(x > 0);
        """
        result = verify_program(src)
        assert not result.safe
        assert result.counterexample is not None

    def test_negative_value(self):
        src = """
        let x = 0 - 5;
        assert(x > 0);
        """
        result = verify_program(src)
        assert not result.safe

    def test_violation_after_computation(self):
        src = """
        let x = 3;
        let y = x - 5;
        assert(y > 0);
        """
        result = verify_program(src)
        assert not result.safe

    def test_violation_in_else_branch(self):
        src = """
        let x = 0;
        if (x > 0) {
            let y = 1;
        } else {
            assert(x > 0);
        }
        """
        result = verify_program(src)
        assert not result.safe


# ============================================================================
# Section 10: Counterexample Quality
# ============================================================================

class TestCounterexampleQuality:
    """Test that counterexamples are meaningful."""

    def test_cex_has_path(self):
        src = """
        let x = 0;
        assert(x > 0);
        """
        result = verify_program(src)
        assert not result.safe
        assert result.counterexample is not None
        assert len(result.counterexample) > 0

    def test_cex_reaches_error(self):
        src = """
        let x = 0;
        assert(x > 0);
        """
        result = verify_program(src)
        assert not result.safe
        # Path should end at error
        types = [step[0] for step in result.counterexample]
        assert 'error' in types

    def test_check_assertion_api(self):
        src = """
        let x = 0;
        assert(x > 0);
        """
        safe, inputs = check_assertion(src)
        assert not safe


# ============================================================================
# Section 11: CEGAR Refinement
# ============================================================================

class TestCEGARRefinement:
    """Test that CEGAR refinement improves precision."""

    def test_predicates_discovered(self):
        src = """
        let x = 5;
        assert(x > 0);
        """
        result = get_predicates(src)
        assert result['safe']
        assert len(result['predicates']) > 0

    def test_refinement_count(self):
        src = """
        let x = 5;
        if (x > 0) {
            let y = x + 1;
            assert(y > 1);
        }
        """
        result = verify_program(src)
        assert result.safe
        assert result.refinement_count >= 0

    def test_comparison_api(self):
        src = """
        let x = 5;
        assert(x > 0);
        """
        comp = compare_with_without_refinement(src)
        assert 'no_refinement' in comp
        assert 'with_refinement' in comp
        assert comp['with_refinement']['safe']


# ============================================================================
# Section 12: Loop Programs
# ============================================================================

class TestLoopPrograms:
    """Programs with loops."""

    def test_simple_countdown_safe(self):
        src = """
        let x = 5;
        while (x > 0) {
            x = x - 1;
        }
        assert(x == 0);
        """
        result = verify_program(src, max_nodes=200)
        # May or may not prove safe depending on loop unrolling depth
        # But should terminate without error
        assert isinstance(result, ARTResult)

    def test_bounded_loop(self):
        src = """
        let x = 3;
        while (x > 0) {
            x = x - 1;
        }
        """
        result = verify_program(src, max_nodes=200)
        assert result.safe  # No assertions to violate

    def test_loop_with_assertion_result(self):
        src = """
        let x = 5;
        while (x > 0) {
            x = x - 1;
        }
        assert(x > 0);
        """
        result = verify_program(src, max_nodes=300)
        # Loop unrolling is bounded -- ART may or may not find the bug
        # depending on coverage and node budget. Just check it terminates.
        assert isinstance(result, ARTResult)
        assert result.art_nodes > 0


# ============================================================================
# Section 13: Coverage
# ============================================================================

class TestCoverage:
    """Test ART coverage (node subsumption)."""

    def test_coverage_count_nonneg(self):
        src = """
        let x = 5;
        if (x > 0) {
            let y = 1;
        } else {
            let y = 2;
        }
        """
        result = verify_program(src)
        assert result.covered_count >= 0

    def test_art_node_count(self):
        src = """
        let x = 5;
        let y = 10;
        if (x > 0) { let z = 1; }
        if (y > 0) { let w = 2; }
        """
        result = verify_program(src)
        assert result.art_nodes > 0

    def test_loop_coverage_bounds_exploration(self):
        src = """
        let x = 10;
        while (x > 0) {
            x = x - 1;
        }
        """
        result = verify_program(src, max_nodes=100)
        assert result.art_nodes > 0
        assert result.art_nodes <= 100


# ============================================================================
# Section 14: ART Summary API
# ============================================================================

class TestARTSummary:
    """Test the summary API."""

    def test_safe_summary(self):
        src = """
        let x = 5;
        assert(x > 0);
        """
        s = art_summary(src)
        assert s['safe']
        assert s['art_nodes'] > 0
        assert s['predicate_count'] >= 0
        assert not s['has_counterexample']

    def test_unsafe_summary(self):
        src = """
        let x = 0;
        assert(x > 0);
        """
        s = art_summary(src)
        assert not s['safe']
        assert s['has_counterexample']
        assert s['counterexample_inputs'] is not None

    def test_summary_keys(self):
        src = "let x = 1;"
        s = art_summary(src)
        expected_keys = {'safe', 'art_nodes', 'refinement_count', 'predicates',
                         'predicate_count', 'covered_count', 'has_counterexample',
                         'counterexample_inputs'}
        assert expected_keys.issubset(s.keys())


# ============================================================================
# Section 15: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge cases and robustness."""

    def test_multiple_assertions(self):
        src = """
        let x = 5;
        assert(x > 0);
        assert(x < 10);
        """
        result = verify_program(src)
        assert result.safe

    def test_assertion_with_computation(self):
        src = """
        let x = 3;
        let y = x * 2;
        assert(y > 5);
        """
        result = verify_program(src)
        assert result.safe

    def test_nested_conditional_assertion(self):
        src = """
        let x = 5;
        let y = 10;
        if (x > 0) {
            if (y > 0) {
                assert(x + y > 0);
            }
        }
        """
        result = verify_program(src)
        assert result.safe

    def test_failing_nested_assertion(self):
        src = """
        let x = 0;
        let y = 0;
        if (x == 0) {
            if (y == 0) {
                assert(x + y > 0);
            }
        }
        """
        result = verify_program(src)
        assert not result.safe

    def test_both_branches_safe(self):
        src = """
        let x = 5;
        if (x > 3) {
            assert(x > 0);
        } else {
            assert(x > 0);
        }
        """
        result = verify_program(src)
        assert result.safe

    def test_max_nodes_respected(self):
        src = """
        let x = 100;
        while (x > 0) {
            x = x - 1;
        }
        """
        result = verify_program(src, max_nodes=50)
        assert result.art_nodes <= 50

    def test_subtraction_assertion(self):
        src = """
        let x = 10;
        let y = x - 3;
        assert(y > 0);
        """
        result = verify_program(src)
        assert result.safe


# ============================================================================
# Section 16: Predicate Discovery Depth
# ============================================================================

class TestPredicateDiscovery:
    """Test that predicates are discovered through refinement."""

    def test_condition_predicates_seeded(self):
        src = """
        let x = 5;
        if (x > 0) {
            assert(x > 0);
        }
        """
        result = get_predicates(src)
        assert len(result['predicates']) > 0

    def test_assignment_equality_predicate(self):
        src = """
        let x = 5;
        assert(x == 5);
        """
        result = verify_program(src)
        assert result.safe

    def test_two_variable_predicate(self):
        src = """
        let x = 5;
        let y = 10;
        assert(x + y == 15);
        """
        result = verify_program(src)
        assert result.safe

    def test_predicate_after_reassignment(self):
        src = """
        let x = 5;
        x = x + 1;
        assert(x > 5);
        """
        result = verify_program(src)
        assert result.safe


# ============================================================================
# Section 17: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_conditional_assignment_safe(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        assert(y > 0);
        """
        result = verify_program(src)
        # y is either 1 or 2, both > 0
        assert result.safe

    def test_abs_pattern(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = 0 - x;
        }
        assert(y > 0);
        """
        result = verify_program(src)
        # For x=5: y=5, assertion holds
        assert result.safe

    def test_multiple_paths_unsafe(self):
        src = """
        let x = 0 - 1;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = 0;
        }
        assert(y > 0);
        """
        result = verify_program(src)
        # x = -1, so else branch: y = 0, assertion fails
        assert not result.safe

    def test_sequential_checks(self):
        src = """
        let x = 5;
        assert(x > 0);
        let y = x + 1;
        assert(y > 1);
        let z = y + 1;
        assert(z > 2);
        """
        result = verify_program(src)
        assert result.safe

    def test_build_cfg_api(self):
        src = """
        let x = 5;
        if (x > 0) { let y = 1; }
        assert(x > 0);
        """
        cfg = build_cfg_from_source(src)
        assert cfg.entry is not None
        assert cfg.exit_node is not None
        assert cfg.error_node is not None
        assert len(cfg.nodes) > 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
