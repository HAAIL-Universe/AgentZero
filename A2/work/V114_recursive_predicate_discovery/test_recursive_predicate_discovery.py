"""Tests for V114: Recursive Predicate Discovery"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
CHALLENGES = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges')
sys.path.insert(0, os.path.join(CHALLENGES, 'C037_smt_solver'))

from recursive_predicate_discovery import (
    discover_predicates, discover_inductive_predicates, discover_and_verify,
    get_cfg, get_program_info, check_inductiveness, compare_discovery_strategies,
    predicate_summary, build_cfg, PredicateSource, PredicateDiscoveryEngine,
    CFGNodeType, Predicate, DiscoveryResult, InductivePredicate,
    _extract_program_constants, _extract_program_variables,
    _extract_conditions, _extract_assertions, _generate_template_predicates,
    _generate_interval_predicates, _generate_condition_predicates,
    _generate_assertion_predicates, _run_interval_analysis,
    _ast_to_smt, _collect_smt_vars, _score_predicates,
    _check_predicate_sufficient,
)
from smt_solver import Var, IntConst, App, Op, Sort, SortKind

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ============================================================
# 1. CFG Construction
# ============================================================

class TestCFGConstruction:
    def test_simple_assignment(self):
        cfg = build_cfg("let x = 5;")
        assert cfg.entry is not None
        assert cfg.exit is not None
        assert len(cfg.nodes) >= 3  # entry, assign, exit

    def test_if_statement(self):
        cfg = build_cfg("let x = 5; if (x > 0) { let y = 1; } else { let y = 2; }")
        edges = cfg.get_edges()
        # Should have assume and assume_not nodes
        ntypes = [cfg.nodes[e.source].ntype for e in edges]
        assert CFGNodeType.ASSUME in ntypes
        assert CFGNodeType.ASSUME_NOT in ntypes

    def test_while_loop(self):
        cfg = build_cfg("let x = 10; while (x > 0) { x = x - 1; }")
        headers = cfg.get_loop_headers()
        assert len(headers) >= 1

    def test_nested_if(self):
        src = "let x = 5; if (x > 0) { if (x < 10) { let y = 1; } }"
        cfg = build_cfg(src)
        assume_nodes = [n for n in cfg.nodes.values()
                        if n.ntype == CFGNodeType.ASSUME]
        assert len(assume_nodes) >= 2

    def test_sequential_assignments(self):
        cfg = build_cfg("let x = 1; let y = 2; let z = 3;")
        assign_nodes = [n for n in cfg.nodes.values()
                        if n.ntype == CFGNodeType.ASSIGN]
        assert len(assign_nodes) == 3

    def test_get_variables(self):
        cfg = build_cfg("let x = 5; let y = x + 1;")
        variables = cfg.get_variables()
        assert 'x' in variables
        assert 'y' in variables

    def test_cfg_edges(self):
        cfg = build_cfg("let x = 1; let y = 2;")
        edges = cfg.get_edges()
        assert len(edges) >= 2

    def test_assert_node(self):
        cfg = build_cfg("let x = 5; assert(x > 0);")
        assert_nodes = [n for n in cfg.nodes.values()
                        if n.ntype == CFGNodeType.ASSERT]
        assert len(assert_nodes) == 1


# ============================================================
# 2. Program Analysis Extraction
# ============================================================

class TestProgramExtraction:
    def test_extract_constants(self):
        consts = _extract_program_constants("let x = 5; let y = 10;")
        assert 5 in consts
        assert 10 in consts
        assert 0 in consts  # always included

    def test_extract_neighbor_constants(self):
        consts = _extract_program_constants("let x = 5;")
        assert 4 in consts  # 5-1
        assert 6 in consts  # 5+1

    def test_extract_variables(self):
        variables = _extract_program_variables("let x = 5; let y = x + 1;")
        assert 'x' in variables
        assert 'y' in variables

    def test_extract_conditions(self):
        conditions = _extract_conditions("let x = 5; if (x > 0) { let y = 1; }")
        assert len(conditions) >= 1

    def test_extract_while_conditions(self):
        conditions = _extract_conditions("let i = 0; while (i < 10) { i = i + 1; }")
        assert len(conditions) >= 1

    def test_extract_assertions(self):
        assertions = _extract_assertions("let x = 5; assert(x > 0);")
        assert len(assertions) >= 1

    def test_no_assertions(self):
        assertions = _extract_assertions("let x = 5;")
        assert len(assertions) == 0

    def test_get_program_info(self):
        info = get_program_info("let x = 5; if (x > 0) { assert(x >= 0); }")
        assert 'x' in info['variables']
        assert len(info['conditions']) >= 1
        assert len(info['assertions']) >= 1
        assert 5 in info['constants']


# ============================================================
# 3. Interval Analysis
# ============================================================

class TestIntervalAnalysis:
    def test_constant_assignment(self):
        intervals = _run_interval_analysis("let x = 5;")
        assert 'x' in intervals
        assert intervals['x'] == (5, 5)

    def test_arithmetic(self):
        intervals = _run_interval_analysis("let x = 5; let y = x + 3;")
        assert intervals.get('y') == (8, 8)

    def test_subtraction(self):
        intervals = _run_interval_analysis("let x = 10; let y = x - 3;")
        assert intervals.get('y') == (7, 7)

    def test_if_join(self):
        src = "let x = 5; if (x > 3) { x = 10; } else { x = 1; }"
        intervals = _run_interval_analysis(src)
        lo, hi = intervals.get('x', (None, None))
        # After join, x should be in [1, 10]
        assert lo is not None and lo <= 1
        assert hi is not None and hi >= 10

    def test_unknown_variable(self):
        intervals = _run_interval_analysis("let x = 5; let y = x;")
        assert intervals.get('y') == (5, 5)


# ============================================================
# 4. Template Predicate Generation
# ============================================================

class TestTemplateGeneration:
    def test_nonneg_templates(self):
        preds = _generate_template_predicates(['x', 'y'], {0, 1})
        terms_str = [str(p.term) for p in preds]
        # Should have x >= 0 and y >= 0
        assert any('x' in t and '>=' in t and '0' in t for t in terms_str)

    def test_upper_bound_templates(self):
        preds = _generate_template_predicates(['x'], {5, 10})
        terms_str = [str(p.term) for p in preds]
        assert any('<=' in t for t in terms_str)

    def test_var_comparison_templates(self):
        preds = _generate_template_predicates(['x', 'y'], {0})
        terms_str = [str(p.term) for p in preds]
        # Should have x < y or y < x
        assert any('<' in t and 'x' in t and 'y' in t for t in terms_str)

    def test_source_is_template(self):
        preds = _generate_template_predicates(['x'], {0})
        assert all(p.source == PredicateSource.TEMPLATE for p in preds)

    def test_empty_variables(self):
        preds = _generate_template_predicates([], {0, 1})
        assert len(preds) == 0

    def test_constant_limiting(self):
        # With many constants, should limit
        preds = _generate_template_predicates(['x'], set(range(100)))
        # Should not explode
        assert len(preds) < 500


# ============================================================
# 5. Interval Predicate Generation
# ============================================================

class TestIntervalPredicates:
    def test_generates_bounds(self):
        preds = _generate_interval_predicates("let x = 5;")
        assert len(preds) >= 1
        assert any(p.source == PredicateSource.INTERVAL for p in preds)

    def test_exact_value(self):
        preds = _generate_interval_predicates("let x = 5;")
        # Should have x >= 5, x <= 5, x == 5
        descs = [p.description for p in preds]
        assert any('exact' in d for d in descs)

    def test_multiple_variables(self):
        preds = _generate_interval_predicates("let x = 3; let y = 7;")
        var_mentions = set()
        for p in preds:
            if 'x' in p.description:
                var_mentions.add('x')
            if 'y' in p.description:
                var_mentions.add('y')
        assert 'x' in var_mentions
        assert 'y' in var_mentions


# ============================================================
# 6. Condition and Assertion Predicates
# ============================================================

class TestConditionPredicates:
    def test_if_condition(self):
        preds = _generate_condition_predicates(
            "let x = 5; if (x > 0) { let y = 1; }")
        assert len(preds) >= 1
        assert all(p.source == PredicateSource.CONDITION for p in preds)

    def test_while_condition(self):
        preds = _generate_condition_predicates(
            "let i = 0; while (i < 10) { i = i + 1; }")
        assert len(preds) >= 1

    def test_assertion_predicates(self):
        preds = _generate_assertion_predicates("let x = 5; assert(x > 0);")
        assert len(preds) >= 1
        assert all(p.source == PredicateSource.ASSERTION for p in preds)


# ============================================================
# 7. AST to SMT Conversion
# ============================================================

class TestASTtoSMT:
    def test_integer_literal(self):
        from stack_vm import lex, Parser
        program = Parser(lex("let x = 5;")).parse()
        expr = program.stmts[0].value
        variables = {}
        term = _ast_to_smt(expr, variables)
        assert isinstance(term, IntConst)
        assert term.value == 5

    def test_variable(self):
        from stack_vm import lex, Parser
        program = Parser(lex("let y = x;")).parse()
        expr = program.stmts[0].value
        variables = {}
        term = _ast_to_smt(expr, variables)
        assert isinstance(term, Var)
        assert term.name == 'x'

    def test_binary_op(self):
        from stack_vm import lex, Parser
        program = Parser(lex("let y = x + 1;")).parse()
        expr = program.stmts[0].value
        variables = {}
        term = _ast_to_smt(expr, variables)
        assert isinstance(term, App)
        assert term.op == Op.ADD

    def test_comparison(self):
        from stack_vm import lex, Parser
        program = Parser(lex("if (x > 0) { let y = 1; }")).parse()
        cond = program.stmts[0].cond
        variables = {}
        term = _ast_to_smt(cond, variables)
        assert isinstance(term, App)
        assert term.op == Op.GT

    def test_collect_vars(self):
        x = Var('x', INT)
        y = Var('y', INT)
        term = App(Op.ADD, [x, y], INT)
        vars_set = _collect_smt_vars(term)
        assert vars_set == {'x', 'y'}


# ============================================================
# 8. Inductiveness Checking
# ============================================================

class TestInductiveness:
    def test_preserved_predicate(self):
        src = "let x = 0; while (x < 10) { x = x + 1; }"
        x = Var('x', INT)
        # x >= 0 should be preserved (adding 1 to non-negative stays non-negative)
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        result = check_inductiveness(src, pred)
        assert result.preserved is True

    def test_not_preserved_predicate(self):
        src = "let x = 0; while (x < 10) { x = x + 1; }"
        x = Var('x', INT)
        # x == 0 is not preserved by x = x + 1
        pred = App(Op.EQ, [x, IntConst(0)], BOOL)
        result = check_inductiveness(src, pred)
        assert result.preserved is False

    def test_no_loop(self):
        src = "let x = 5;"
        x = Var('x', INT)
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        result = check_inductiveness(src, pred)
        assert result.preserved is True  # vacuously true

    def test_upper_bound_not_preserved(self):
        src = "let x = 0; while (x < 10) { x = x + 1; }"
        x = Var('x', INT)
        pred = App(Op.LE, [x, IntConst(5)], BOOL)
        result = check_inductiveness(src, pred)
        assert result.preserved is False


# ============================================================
# 9. Full Discovery Pipeline
# ============================================================

class TestDiscoveryPipeline:
    def test_simple_program(self):
        result = discover_predicates("let x = 5;")
        assert isinstance(result, DiscoveryResult)
        assert result.selected_count > 0
        assert result.total_candidates > 0

    def test_loop_program(self):
        src = "let i = 0; while (i < 10) { i = i + 1; }"
        result = discover_predicates(src)
        assert result.selected_count > 0
        assert len(result.source_counts) > 0

    def test_if_program(self):
        src = "let x = 5; if (x > 0) { let y = 1; } else { let y = -1; }"
        result = discover_predicates(src)
        # With no loop, templates are vacuously inductive, so inductive dominates
        assert result.selected_count > 0
        assert len(result.source_counts) >= 1

    def test_max_predicates_limit(self):
        src = "let x = 5; let y = 10; let z = x + y;"
        result = discover_predicates(src, max_predicates=5)
        assert result.selected_count <= 5

    def test_ranked_by_score(self):
        src = "let x = 5; if (x > 0) { assert(x >= 0); }"
        result = discover_predicates(src)
        scores = [p.score for p in result.ranked_predicates]
        assert scores == sorted(scores, reverse=True)

    def test_assertion_program(self):
        src = "let x = 5; assert(x > 0);"
        result = discover_predicates(src)
        has_assertion = any(p.source == PredicateSource.ASSERTION
                          for p in result.predicates)
        assert has_assertion

    def test_discovery_stats(self):
        src = "let x = 5; if (x > 0) { let y = 1; }"
        result = discover_predicates(src)
        assert 'conditions' in result.discovery_stats
        assert 'templates' in result.discovery_stats


# ============================================================
# 10. Inductive Predicate Discovery
# ============================================================

class TestInductiveDiscovery:
    def test_loop_finds_inductive(self):
        src = "let x = 0; while (x < 10) { x = x + 1; }"
        preds = discover_inductive_predicates(src)
        assert len(preds) > 0
        assert all(p.source == PredicateSource.INDUCTIVE for p in preds)

    def test_counter_nonneg(self):
        src = "let x = 0; while (x < 10) { x = x + 1; }"
        preds = discover_inductive_predicates(src)
        # x >= 0 should be among inductive predicates
        descs = [str(p.term) for p in preds]
        has_nonneg = any('x' in d and '>=' in d for d in descs)
        assert has_nonneg

    def test_no_loop_empty(self):
        src = "let x = 5; let y = 10;"
        preds = discover_inductive_predicates(src)
        # Without loops, everything is vacuously inductive, but we filter
        # to only return INDUCTIVE-source predicates
        assert all(p.source == PredicateSource.INDUCTIVE for p in preds)


# ============================================================
# 11. Discovery and Verification
# ============================================================

class TestDiscoverAndVerify:
    def test_simple_verification(self):
        src = "let x = 5; assert(x > 0);"
        result = discover_and_verify(src)
        assert 'discovery' in result
        assert 'verification' in result
        assert result['predicate_count'] > 0

    def test_sufficient_check(self):
        src = "let x = 5;"
        x = Var('x', INT)
        prop = App(Op.GE, [x, IntConst(0)], BOOL)
        result = discover_and_verify(src, property_term=prop)
        assert 'sufficient' in result

    def test_with_loop(self):
        src = "let x = 0; while (x < 10) { x = x + 1; } assert(x >= 0);"
        result = discover_and_verify(src)
        assert result['total_candidates'] > 0


# ============================================================
# 12. Predicate Scoring
# ============================================================

class TestPredicateScoring:
    def test_assertion_scores_high(self):
        x = Var('x', INT)
        p1 = Predicate(App(Op.GT, [x, IntConst(0)], BOOL),
                       PredicateSource.ASSERTION, description="assertion")
        p2 = Predicate(App(Op.GE, [x, IntConst(0)], BOOL),
                       PredicateSource.TEMPLATE, description="template")
        cfg = build_cfg("let x = 5;")
        scored = _score_predicates([p1, p2], cfg, ['x'])
        assert scored[0].source == PredicateSource.ASSERTION

    def test_deduplication(self):
        x = Var('x', INT)
        p1 = Predicate(App(Op.GT, [x, IntConst(0)], BOOL),
                       PredicateSource.TEMPLATE)
        p2 = Predicate(App(Op.GT, [x, IntConst(0)], BOOL),
                       PredicateSource.TEMPLATE)
        cfg = build_cfg("let x = 5;")
        scored = _score_predicates([p1, p2], cfg, ['x'])
        assert len(scored) == 1

    def test_condition_beats_template(self):
        x = Var('x', INT)
        p1 = Predicate(App(Op.GT, [x, IntConst(0)], BOOL),
                       PredicateSource.CONDITION)
        p2 = Predicate(App(Op.GE, [x, IntConst(0)], BOOL),
                       PredicateSource.TEMPLATE)
        cfg = build_cfg("let x = 5;")
        scored = _score_predicates([p1, p2], cfg, ['x'])
        assert scored[0].source == PredicateSource.CONDITION


# ============================================================
# 13. Strategy Comparison
# ============================================================

class TestStrategyComparison:
    def test_compare(self):
        src = "let x = 5; if (x > 0) { let y = 1; }"
        results = compare_discovery_strategies(src)
        assert 'templates_only' in results
        assert 'conditions_only' in results
        assert 'full' in results
        assert results['full']['total_candidates'] >= results['templates_only']['total_candidates']

    def test_templates_only(self):
        src = "let x = 5; let y = 10;"
        results = compare_discovery_strategies(src)
        assert results['templates_only']['total_candidates'] > 0


# ============================================================
# 14. Summary Output
# ============================================================

class TestSummary:
    def test_summary_format(self):
        src = "let x = 5; if (x > 0) { let y = 1; }"
        summary = predicate_summary(src)
        assert "Predicate Discovery Summary" in summary
        assert "Total candidates" in summary

    def test_summary_with_loop(self):
        src = "let i = 0; while (i < 10) { i = i + 1; }"
        summary = predicate_summary(src)
        assert "Top 10" in summary


# ============================================================
# 15. Engine Configuration
# ============================================================

class TestEngineConfig:
    def test_templates_only(self):
        engine = PredicateDiscoveryEngine(
            "let x = 5;",
            use_templates=True, use_intervals=False,
            use_conditions=False, use_assertions=False,
            use_interpolation=False, use_inductive=False
        )
        result = engine.discover()
        assert result.total_candidates > 0
        assert all(p.source == PredicateSource.TEMPLATE for p in result.predicates)

    def test_intervals_only(self):
        engine = PredicateDiscoveryEngine(
            "let x = 5;",
            use_templates=False, use_intervals=True,
            use_conditions=False, use_assertions=False,
            use_interpolation=False, use_inductive=False
        )
        result = engine.discover()
        assert all(p.source == PredicateSource.INTERVAL for p in result.predicates)

    def test_all_disabled(self):
        engine = PredicateDiscoveryEngine(
            "let x = 5;",
            use_templates=False, use_intervals=False,
            use_conditions=False, use_assertions=False,
            use_interpolation=False, use_inductive=False
        )
        result = engine.discover()
        assert result.total_candidates == 0


# ============================================================
# 16. get_cfg API
# ============================================================

class TestGetCFG:
    def test_returns_cfg(self):
        cfg = get_cfg("let x = 5;")
        assert cfg.entry is not None
        assert cfg.exit is not None

    def test_loop_headers(self):
        cfg = get_cfg("let i = 0; while (i < 10) { i = i + 1; }")
        headers = cfg.get_loop_headers()
        assert len(headers) >= 1


# ============================================================
# 17. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        # Empty program should still work
        result = discover_predicates("let x = 0;")
        assert isinstance(result, DiscoveryResult)

    def test_nested_loops(self):
        src = """
        let i = 0;
        while (i < 5) {
            let j = 0;
            while (j < 5) {
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = discover_predicates(src)
        assert result.selected_count > 0

    def test_multiple_assertions(self):
        src = "let x = 5; let y = 10; assert(x > 0); assert(y > 0);"
        result = discover_predicates(src)
        assertion_preds = [p for p in result.predicates
                          if p.source == PredicateSource.ASSERTION]
        assert len(assertion_preds) >= 2

    def test_negative_constants(self):
        src = "let x = -5; let y = x + 10;"
        consts = _extract_program_constants(src)
        assert 5 in consts  # from negation of -5
        assert 10 in consts

    def test_complex_expression(self):
        src = "let x = 5; let y = x * 2 + 1;"
        result = discover_predicates(src)
        assert result.selected_count > 0


# ============================================================
# 18. Predicate Sufficiency Checking
# ============================================================

class TestPredicateSufficiency:
    def test_trivially_sufficient(self):
        src = "let x = 5;"
        x = Var('x', INT)
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        result = _check_predicate_sufficient(src, [pred])
        assert result['predicate_count'] == 1

    def test_with_assertion(self):
        src = "let x = 5; assert(x > 0);"
        x = Var('x', INT)
        pred = App(Op.GT, [x, IntConst(0)], BOOL)
        result = _check_predicate_sufficient(src, [pred], pred)
        assert 'sufficient' in result

    def test_empty_predicates(self):
        src = "let x = 5;"
        result = _check_predicate_sufficient(src, [])
        assert result['predicate_count'] == 0


# ============================================================
# 19. Multi-Variable Programs
# ============================================================

class TestMultiVariable:
    def test_two_variables(self):
        src = "let x = 5; let y = 10; if (x < y) { let z = 1; }"
        result = discover_predicates(src)
        # Should have relational predicates (x < y, etc.)
        has_relational = any('x' in str(p.term) and 'y' in str(p.term)
                            for p in result.predicates)
        assert has_relational

    def test_sum_conservation(self):
        src = "let x = 5; let y = 5; x = x + 1; y = y - 1;"
        result = discover_predicates(src)
        # Template should generate x + y == 10
        assert result.total_candidates > 0

    def test_three_variables(self):
        src = "let a = 1; let b = 2; let c = 3;"
        result = discover_predicates(src)
        info = get_program_info(src)
        assert len(info['variables']) == 3


# ============================================================
# 20. Predicate Source Tracking
# ============================================================

class TestSourceTracking:
    def test_all_sources_labeled(self):
        src = "let x = 0; while (x < 10) { x = x + 1; } assert(x >= 0);"
        result = discover_predicates(src)
        sources = {p.source for p in result.predicates}
        # Should have multiple source types
        assert len(sources) >= 2

    def test_source_counts_match(self):
        src = "let x = 5; if (x > 0) { let y = 1; }"
        result = discover_predicates(src)
        total_from_counts = sum(result.source_counts.values())
        assert total_from_counts == result.selected_count

    def test_descriptions_present(self):
        src = "let x = 5; assert(x > 0);"
        result = discover_predicates(src)
        for p in result.predicates:
            assert p.description != "" or p.source == PredicateSource.TEMPLATE


# ============================================================
# 21. Loop Analysis
# ============================================================

class TestLoopAnalysis:
    def test_countdown_loop(self):
        src = "let x = 10; while (x > 0) { x = x - 1; }"
        result = discover_predicates(src)
        assert result.selected_count > 0

    def test_accumulator_loop(self):
        src = "let sum = 0; let i = 0; while (i < 5) { sum = sum + i; i = i + 1; }"
        result = discover_predicates(src)
        # Should find inductive predicates
        preds = discover_inductive_predicates(src)
        assert len(preds) > 0

    def test_double_loop(self):
        src = """
        let x = 0;
        while (x < 10) { x = x + 1; }
        let y = 0;
        while (y < 5) { y = y + 1; }
        """
        cfg = get_cfg(src)
        headers = cfg.get_loop_headers()
        assert len(headers) >= 2
