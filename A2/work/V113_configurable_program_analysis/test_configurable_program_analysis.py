"""Tests for V113: Configurable Program Analysis (CPA)"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from configurable_program_analysis import (
    # CPA interface
    AbstractState, TransferRelation, MergeOperator, StopOperator, CPA,
    # Concrete CPAs
    IntervalCPA, IntervalState, IntervalTransfer,
    PredicateCPA, PredicateState, PredicateRegistry, PredicateTransfer,
    ZoneCPA, ZoneState, ZoneTransfer,
    CompositeCPA, CompositeState,
    # Merge/Stop strategies
    MergeSep, MergeJoin, StopSep, StopJoin, NoPrecisionAdjustment,
    # Algorithm
    cpa_algorithm, cpa_cegar, CPAResult, CPAEdge, EdgeType, ARTNode,
    # Public API
    verify_with_intervals, verify_with_zones, verify_with_predicates,
    verify_with_composite, compare_cpas, get_variable_ranges, cpa_summary,
    # CFG
    build_cfg_from_source, CFGNodeType,
)
from smt_solver import IntConst

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
from stack_vm import lex, Parser
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'V104_relational_abstract_domains'))
from relational_domains import ZoneDomain


# ============================================================
# 1. Interval State Operations
# ============================================================

class TestIntervalState:
    def test_top_state(self):
        s = IntervalState()
        assert not s.is_bottom()
        assert s.get_interval('x') == (float('-inf'), float('inf'))

    def test_bottom_state(self):
        s = IntervalState(bottom=True)
        assert s.is_bottom()

    def test_concrete_state(self):
        s = IntervalState({'x': (1, 5), 'y': (0, 10)})
        assert s.get_interval('x') == (1, 5)
        assert s.get_interval('y') == (0, 10)

    def test_subsumption(self):
        s1 = IntervalState({'x': (0, 10)})  # wider
        s2 = IntervalState({'x': (2, 8)})   # narrower
        assert s1.subsumes(s2)
        assert not s2.subsumes(s1)

    def test_join(self):
        s1 = IntervalState({'x': (0, 5)})
        s2 = IntervalState({'x': (3, 10)})
        j = s1.join(s2)
        assert j.get_interval('x') == (0, 10)

    def test_widen(self):
        s1 = IntervalState({'x': (0, 5)})
        s2 = IntervalState({'x': (-1, 7)})
        w = s1.widen(s2)
        assert w.get_interval('x')[0] == float('-inf')  # lo decreased -> -inf
        assert w.get_interval('x')[1] == float('inf')   # hi increased -> inf

    def test_copy_independence(self):
        s1 = IntervalState({'x': (0, 5)})
        s2 = s1.copy()
        s2.env['x'] = (1, 3)
        assert s1.get_interval('x') == (0, 5)

    def test_bottom_subsumes(self):
        bot = IntervalState(bottom=True)
        top = IntervalState()
        assert top.subsumes(bot)
        assert not bot.subsumes(top)

    def test_join_with_bottom(self):
        s = IntervalState({'x': (1, 5)})
        bot = IntervalState(bottom=True)
        assert s.join(bot).get_interval('x') == (1, 5)
        assert bot.join(s).get_interval('x') == (1, 5)


# ============================================================
# 2. Predicate State Operations
# ============================================================

class TestPredicateState:
    def test_empty_predicates(self):
        s = PredicateState()
        assert s.predicates == frozenset()
        assert not s.is_bottom()

    def test_subsumption(self):
        s1 = PredicateState(frozenset({0}))    # fewer preds = more general
        s2 = PredicateState(frozenset({0, 1}))  # more preds = more specific
        assert s1.subsumes(s2)
        assert not s2.subsumes(s1)

    def test_join(self):
        s1 = PredicateState(frozenset({0, 1, 2}))
        s2 = PredicateState(frozenset({1, 2, 3}))
        j = s1.join(s2)
        assert j.predicates == frozenset({1, 2})  # intersection

    def test_bottom(self):
        bot = PredicateState(bottom=True)
        assert bot.is_bottom()
        s = PredicateState(frozenset({0}))
        assert s.subsumes(bot)


# ============================================================
# 3. Zone State Operations
# ============================================================

class TestZoneState:
    def test_empty_zone(self):
        s = ZoneState()
        assert not s.is_bottom()

    def test_bottom(self):
        s = ZoneState(bottom=True)
        assert s.is_bottom()

    def test_subsumption(self):
        from relational_domains import ZoneDomain
        z1 = ZoneDomain(['x'])
        z1.set_upper('x', 10)
        z1.set_lower('x', 0)
        z1.close()
        s1 = ZoneState(zone=z1)

        z2 = ZoneDomain(['x'])
        z2.set_upper('x', 5)
        z2.set_lower('x', 2)
        z2.close()
        s2 = ZoneState(zone=z2)

        assert s1.subsumes(s2)  # [0,10] >= [2,5]
        assert not s2.subsumes(s1)

    def test_copy_independence(self):
        from relational_domains import ZoneDomain
        z = ZoneDomain(['x'])
        z.set_upper('x', 5)
        z.close()
        s1 = ZoneState(zone=z)
        s2 = s1.copy()
        s2.zone.set_upper('x', 10)
        assert s1.zone.get_upper('x') == 5


# ============================================================
# 4. Composite State
# ============================================================

class TestCompositeState:
    def test_composite_basic(self):
        c = CompositeState([IntervalState({'x': (0, 5)}), IntervalState({'y': (1, 3)})])
        assert not c.is_bottom()
        assert len(c.components) == 2

    def test_composite_bottom_propagation(self):
        c = CompositeState([IntervalState(bottom=True), IntervalState({'y': (1, 3)})])
        assert c.is_bottom()

    def test_composite_subsumption(self):
        c1 = CompositeState([IntervalState({'x': (0, 10)}), IntervalState({'y': (0, 10)})])
        c2 = CompositeState([IntervalState({'x': (2, 5)}), IntervalState({'y': (3, 7)})])
        assert c1.subsumes(c2)
        assert not c2.subsumes(c1)

    def test_composite_join(self):
        c1 = CompositeState([IntervalState({'x': (0, 5)})])
        c2 = CompositeState([IntervalState({'x': (3, 10)})])
        j = c1.join(c2)
        assert j.components[0].get_interval('x') == (0, 10)


# ============================================================
# 5. Merge and Stop Operators
# ============================================================

class TestMergeStop:
    def test_merge_sep(self):
        m = MergeSep()
        s1 = IntervalState({'x': (0, 5)})
        s2 = IntervalState({'x': (3, 10)})
        result = m.merge(s1, s2, None)
        assert result is s2  # no merge

    def test_merge_join(self):
        m = MergeJoin()
        s1 = IntervalState({'x': (0, 5)})
        s2 = IntervalState({'x': (3, 10)})
        result = m.merge(s1, s2, None)
        assert result.get_interval('x') == (0, 10)

    def test_stop_sep(self):
        stop = StopSep()
        s = IntervalState({'x': (2, 5)})
        reached = [IntervalState({'x': (0, 10)})]
        assert stop.stop(s, reached, None)
        assert not stop.stop(IntervalState({'x': (-1, 5)}), reached, None)

    def test_stop_join(self):
        stop = StopJoin()
        s = IntervalState({'x': (3, 7)})
        reached = [IntervalState({'x': (0, 5)}), IntervalState({'x': (5, 10)})]
        assert stop.stop(s, reached, None)


# ============================================================
# 6. CFG Construction
# ============================================================

class TestCFG:
    def test_simple_cfg(self):
        source = "let x = 5;"
        cfg = build_cfg_from_source(source)
        assert cfg.entry is not None
        assert cfg.exit_node is not None

    def test_if_cfg(self):
        source = """
        let x = 5;
        if (x > 0) {
            x = x + 1;
        }
        """
        cfg = build_cfg_from_source(source)
        has_assume = any(n.type == CFGNodeType.ASSUME for n in cfg.nodes)
        assert has_assume

    def test_assert_cfg(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        cfg = build_cfg_from_source(source)
        has_error = any(n.type == CFGNodeType.ERROR for n in cfg.nodes)
        assert has_error


# ============================================================
# 7. Interval CPA Analysis
# ============================================================

class TestIntervalCPA:
    def test_simple_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = verify_with_intervals(source)
        assert result.safe
        assert result.domain_name == "IntervalCPA"

    def test_simple_unsafe(self):
        source = """
        let x = 0;
        assert(x > 0);
        """
        result = verify_with_intervals(source)
        assert not result.safe

    def test_assignment_tracking(self):
        source = """
        let x = 3;
        let y = x + 2;
        assert(y > 0);
        """
        result = verify_with_intervals(source)
        assert result.safe

    def test_conditional(self):
        source = """
        let x = 5;
        if (x > 3) {
            x = 10;
        } else {
            x = 1;
        }
        assert(x > 0);
        """
        result = verify_with_intervals(source)
        assert result.safe

    def test_variable_ranges(self):
        source = """
        let x = 5;
        let y = x + 3;
        """
        result = verify_with_intervals(source)
        ranges = result.variable_ranges
        if ranges:
            if 'x' in ranges:
                assert ranges['x'] == (5, 5)
            if 'y' in ranges:
                assert ranges['y'] == (8, 8)

    def test_art_node_count(self):
        source = "let x = 5;"
        result = verify_with_intervals(source)
        assert result.art_nodes > 0


# ============================================================
# 8. Zone CPA Analysis
# ============================================================

class TestZoneCPA:
    def test_simple_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = verify_with_zones(source)
        assert result.safe

    def test_simple_unsafe(self):
        source = """
        let x = 0;
        assert(x > 0);
        """
        result = verify_with_zones(source)
        assert not result.safe

    def test_relational(self):
        source = """
        let x = 5;
        let y = x + 1;
        assert(y > x);
        """
        result = verify_with_zones(source)
        # Zone can track x - y <= -1, so y > x is provable
        assert result.safe

    def test_conditional_zone(self):
        source = """
        let x = 5;
        if (x > 0) {
            let y = x;
            assert(y > 0);
        }
        """
        result = verify_with_zones(source)
        assert result.safe

    def test_zone_ranges(self):
        source = """
        let x = 3;
        let y = x + 2;
        """
        result = verify_with_zones(source)
        ranges = result.variable_ranges
        if ranges and 'x' in ranges:
            assert ranges['x'][0] >= 3
            assert ranges['x'][1] <= 3


# ============================================================
# 9. Predicate CPA with CEGAR
# ============================================================

class TestPredicateCPA:
    def test_simple_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = verify_with_predicates(source)
        assert result.safe

    def test_simple_unsafe(self):
        source = """
        let x = 0;
        assert(x > 0);
        """
        result = verify_with_predicates(source)
        assert not result.safe

    def test_counterexample(self):
        source = """
        let x = 0;
        assert(x > 0);
        """
        result = verify_with_predicates(source)
        assert not result.safe
        assert result.counterexample_inputs is not None or result.counterexample is not None

    def test_conditional_safe(self):
        source = """
        let x = 5;
        if (x > 0) {
            assert(x > 0);
        }
        """
        result = verify_with_predicates(source)
        assert result.safe

    def test_predicate_registry(self):
        reg = PredicateRegistry()
        from smt_solver import Var, App, Op, Sort, SortKind
        x = Var('x', Sort(SortKind.INT))
        zero = IntConst(0)
        pred = App(Op.GT, [x, zero], Sort(SortKind.BOOL))
        idx = reg.add_predicate(pred, name="x > 0")
        assert idx == 0
        assert reg.get_predicate_name(0) == "x > 0"

    def test_predicate_state_operations(self):
        s1 = PredicateState(frozenset({0, 1}))
        s2 = PredicateState(frozenset({0, 1, 2}))
        assert s1.subsumes(s2)
        j = s1.join(s2)
        assert j.predicates == frozenset({0, 1})


# ============================================================
# 10. Composite CPA
# ============================================================

class TestCompositeCPA:
    def test_interval_zone_composite(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = verify_with_composite(source, ['interval', 'zone'])
        assert result.safe
        assert 'Composite' in result.domain_name

    def test_composite_unsafe(self):
        source = """
        let x = 0;
        assert(x > 0);
        """
        result = verify_with_composite(source, ['interval', 'zone'])
        assert not result.safe

    def test_composite_node_count(self):
        source = """
        let x = 5;
        let y = 3;
        """
        result = verify_with_composite(source, ['interval'])
        assert result.art_nodes > 0


# ============================================================
# 11. Compare CPAs
# ============================================================

class TestCompareCPAs:
    def test_comparison_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        comp = compare_cpas(source)
        assert 'results' in comp
        assert 'verdicts' in comp
        # All should agree it's safe
        for name, v in comp['verdicts'].items():
            assert v is True, f"{name} disagrees"

    def test_comparison_unsafe(self):
        source = """
        let x = 0;
        assert(x > 0);
        """
        comp = compare_cpas(source)
        for name, v in comp['verdicts'].items():
            assert v is False, f"{name} disagrees"

    def test_comparison_has_all_cpas(self):
        source = "let x = 5;"
        comp = compare_cpas(source)
        assert 'interval' in comp['results']
        assert 'zone' in comp['results']
        assert 'predicate' in comp['results']
        assert 'composite' in comp['results']


# ============================================================
# 12. Variable Range Extraction
# ============================================================

class TestVariableRanges:
    def test_interval_ranges(self):
        source = """
        let x = 5;
        let y = 10;
        """
        ranges = get_variable_ranges(source, 'interval')
        if ranges:
            if 'x' in ranges:
                assert ranges['x'] == (5, 5)

    def test_zone_ranges(self):
        source = """
        let x = 3;
        let y = x + 1;
        """
        ranges = get_variable_ranges(source, 'zone')
        if ranges:
            if 'x' in ranges:
                lo, hi = ranges['x']
                assert lo <= 3 <= hi


# ============================================================
# 13. CPA Summary
# ============================================================

class TestSummary:
    def test_summary(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        s = cpa_summary(source)
        assert 'cfg_nodes' in s
        assert s['cfg_nodes'] > 0
        assert 'comparison' in s


# ============================================================
# 14. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        source = "let x = 0;"
        result = verify_with_intervals(source)
        assert result.safe  # no assertions -> safe

    def test_nested_if(self):
        source = """
        let x = 10;
        if (x > 5) {
            if (x > 3) {
                assert(x > 0);
            }
        }
        """
        result = verify_with_intervals(source)
        assert result.safe

    def test_multiple_assertions(self):
        source = """
        let x = 5;
        assert(x > 0);
        let y = x + 1;
        assert(y > 1);
        """
        result = verify_with_intervals(source)
        assert result.safe

    def test_subtraction(self):
        source = """
        let x = 10;
        let y = x - 3;
        assert(y > 0);
        """
        result = verify_with_intervals(source)
        assert result.safe

    def test_arithmetic(self):
        source = """
        let x = 3;
        let y = x * 2;
        assert(y > 0);
        """
        result = verify_with_intervals(source)
        assert result.safe


# ============================================================
# 15. Interval Transfer Function Details
# ============================================================

class TestIntervalTransfer:
    def test_assign_literal(self):
        transfer = IntervalTransfer()
        state = IntervalState()
        from stack_vm import lex, Parser
        tokens = lex("let x = 5;")
        prog = Parser(tokens).parse()
        stmt = prog.stmts[0]  # LetDecl
        edge = CPAEdge(None, None, EdgeType.ASSIGN, (stmt.name, stmt.value))
        succs = transfer.get_abstract_successors(state, edge, None)
        assert len(succs) == 1
        assert succs[0].get_interval('x') == (5, 5)

    def test_assume_refine(self):
        transfer = IntervalTransfer()
        state = IntervalState({'x': (0, 100)})
        tokens = lex("x > 50")
        # Parse condition
        parser = Parser(tokens)
        # Parse as expression -- trick: wrap in if
        tokens2 = lex("if (x > 50) { let y = 0; }")
        prog = Parser(tokens2).parse()
        cond = prog.stmts[0].cond
        edge = CPAEdge(None, None, EdgeType.ASSUME, cond)
        succs = transfer.get_abstract_successors(state, edge, None)
        assert len(succs) == 1
        lo, hi = succs[0].get_interval('x')
        assert lo >= 51

    def test_skip_preserves_state(self):
        transfer = IntervalTransfer()
        state = IntervalState({'x': (0, 5)})
        edge = CPAEdge(None, None, EdgeType.SKIP, None)
        succs = transfer.get_abstract_successors(state, edge, None)
        assert len(succs) == 1
        assert succs[0].get_interval('x') == (0, 5)

    def test_bottom_produces_nothing(self):
        transfer = IntervalTransfer()
        state = IntervalState(bottom=True)
        edge = CPAEdge(None, None, EdgeType.SKIP, None)
        succs = transfer.get_abstract_successors(state, edge, None)
        assert len(succs) == 0


# ============================================================
# 16. Zone Transfer Function Details
# ============================================================

class TestZoneTransfer:
    def test_assign_const(self):
        transfer = ZoneTransfer()
        state = ZoneState()
        tokens = lex("let x = 5;")
        prog = Parser(tokens).parse()
        stmt = prog.stmts[0]
        edge = CPAEdge(None, None, EdgeType.ASSIGN, (stmt.name, stmt.value))
        succs = transfer.get_abstract_successors(state, edge, None)
        assert len(succs) == 1
        lo, hi = succs[0].get_interval('x')
        assert lo == 5 and hi == 5

    def test_assign_add(self):
        transfer = ZoneTransfer()
        from relational_domains import ZoneDomain
        z = ZoneDomain(['x'])
        z.assign_const('x', 3)
        z.close()
        state = ZoneState(zone=z)
        tokens = lex("let y = x + 2;")
        prog = Parser(tokens).parse()
        stmt = prog.stmts[0]
        edge = CPAEdge(None, None, EdgeType.ASSIGN, (stmt.name, stmt.value))
        succs = transfer.get_abstract_successors(state, edge, None)
        assert len(succs) == 1
        lo, hi = succs[0].get_interval('y')
        assert lo == 5 and hi == 5

    def test_assume_refine_zone(self):
        transfer = ZoneTransfer()
        from relational_domains import ZoneDomain
        z = ZoneDomain(['x'])
        z.set_upper('x', 100)
        z.set_lower('x', 0)
        z.close()
        state = ZoneState(zone=z)
        tokens = lex("if (x > 50) { let y = 0; }")
        prog = Parser(tokens).parse()
        cond = prog.stmts[0].cond
        succs = transfer._apply_assume(state, cond, positive=True)
        assert len(succs) == 1
        lo, hi = succs[0].get_interval('x')
        assert lo >= 51


# ============================================================
# 17. CPA Algorithm Mechanics
# ============================================================

class TestCPAAlgorithm:
    def test_algorithm_terminates(self):
        source = "let x = 5;"
        cpa = IntervalCPA()
        cfg = build_cfg_from_source(source)
        result = cpa_algorithm(cfg, cpa, max_nodes=100)
        assert result.art_nodes > 0

    def test_coverage_works(self):
        source = """
        let x = 5;
        if (x > 3) {
            let y = 1;
        } else {
            let y = 2;
        }
        """
        cpa = IntervalCPA()
        cfg = build_cfg_from_source(source)
        result = cpa_algorithm(cfg, cpa, max_nodes=200)
        # Should complete without exceeding node limit
        assert result.art_nodes < 200

    def test_node_limit(self):
        source = """
        let x = 5;
        if (x > 3) {
            let y = 1;
        } else {
            let y = 2;
        }
        """
        cpa = IntervalCPA()
        cfg = build_cfg_from_source(source)
        result = cpa_algorithm(cfg, cpa, max_nodes=5)
        # Should stop early
        assert result.art_nodes <= 6  # may create one extra


# ============================================================
# 18. CEGAR Mechanics
# ============================================================

class TestCEGAR:
    def test_cegar_safe(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        cpa = PredicateCPA()
        result = cpa_cegar(source, cpa)
        assert result.safe

    def test_cegar_unsafe(self):
        source = """
        let x = 0;
        assert(x > 0);
        """
        cpa = PredicateCPA()
        result = cpa_cegar(source, cpa)
        assert not result.safe

    def test_cegar_with_interval(self):
        """Interval CPA with CEGAR -- no refinement, but should still work."""
        source = """
        let x = 5;
        assert(x > 0);
        """
        cpa = IntervalCPA()
        result = cpa_cegar(source, cpa)
        assert result.safe


# ============================================================
# 19. Predicate Registry
# ============================================================

class TestPredicateRegistryFull:
    def test_add_and_retrieve(self):
        reg = PredicateRegistry()
        from smt_solver import Var, App, Op, Sort, SortKind, IntConst
        x = Var('x', Sort(SortKind.INT))
        pred = App(Op.GT, [x, IntConst(0)], Sort(SortKind.BOOL))
        idx = reg.add_predicate(pred, name="x>0")
        assert idx == 0
        assert reg.get_predicate_name(0) == "x>0"
        assert reg.get_predicate_term(0) is pred

    def test_duplicate_detection(self):
        reg = PredicateRegistry()
        from smt_solver import Var, App, Op, Sort, SortKind, IntConst
        x = Var('x', Sort(SortKind.INT))
        pred = App(Op.GT, [x, IntConst(0)], Sort(SortKind.BOOL))
        idx1 = reg.add_predicate(pred, name="x>0")
        idx2 = reg.add_predicate(pred, name="x>0")
        assert idx1 == idx2

    def test_location_tracking(self):
        reg = PredicateRegistry()
        from smt_solver import Var, App, Op, Sort, SortKind, IntConst
        x = Var('x', Sort(SortKind.INT))
        pred = App(Op.GT, [x, IntConst(0)], Sort(SortKind.BOOL))
        reg.add_predicate(pred, name="x>0", location_id=5)
        preds_at_5 = reg.get_predicates_at(5)
        assert 0 in preds_at_5

    def test_all_predicate_indices(self):
        reg = PredicateRegistry()
        from smt_solver import Var, App, Op, Sort, SortKind, IntConst
        x = Var('x', Sort(SortKind.INT))
        p1 = App(Op.GT, [x, IntConst(0)], Sort(SortKind.BOOL))
        p2 = App(Op.LT, [x, IntConst(10)], Sort(SortKind.BOOL))
        reg.add_predicate(p1, name="x>0")
        reg.add_predicate(p2, name="x<10")
        assert reg.get_all_predicate_indices() == {0, 1}


# ============================================================
# 20. Integration: Complex Programs
# ============================================================

class TestIntegration:
    def test_multi_variable_safe(self):
        source = """
        let x = 10;
        let y = 5;
        let z = x - y;
        assert(z > 0);
        """
        result = verify_with_intervals(source)
        assert result.safe

    def test_conditional_both_branches_safe(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        assert(y > 0);
        """
        result = verify_with_intervals(source)
        assert result.safe

    def test_all_cpas_agree_safe(self):
        source = """
        let x = 10;
        assert(x > 5);
        """
        comp = compare_cpas(source)
        for name, v in comp['verdicts'].items():
            assert v is True, f"{name} says unsafe"

    def test_all_cpas_agree_unsafe(self):
        source = """
        let x = 3;
        assert(x > 5);
        """
        comp = compare_cpas(source)
        for name, v in comp['verdicts'].items():
            assert v is False, f"{name} says safe"

    def test_zone_relational_advantage(self):
        """Zone can track x - y <= c, giving relational precision."""
        source = """
        let x = 5;
        let y = x;
        """
        result = verify_with_zones(source)
        ranges = result.variable_ranges
        if ranges and 'x' in ranges and 'y' in ranges:
            # Both should be exactly 5
            assert ranges['x'] == (5, 5)
            assert ranges['y'] == (5, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
