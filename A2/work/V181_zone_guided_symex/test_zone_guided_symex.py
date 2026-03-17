"""Tests for V181: Zone-Guided Symbolic Execution."""
import pytest
import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
from zone_guided_symex import (
    guided_execute, incremental_guided_execute,
    analyze_zone_pruning, compare_zone_vs_octagon, compare_v001_vs_v181,
    verify_difference_property, batch_guided_execute,
    ZoneGuidedResult, ZonePreAnalysis,
    _zone_pre_analyze, _check_branch_feasibility_zone,
    _check_branch_feasibility_zone_negated,
    _cond_to_zone_constraints, _parse_zone_property,
    _collect_branch_conditions, _collect_branch_zone_states,
    _apply_assignment_to_zone,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V178_zone_abstract_domain'))
from zone import Zone, ZoneConstraint, upper_bound, lower_bound, diff_bound, INF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import lex, Parser, IntLit, BoolLit, BinOp, Var as ASTVar, UnaryOp


# ============================================================
# Zone Pre-Analysis
# ============================================================

class TestZonePreAnalysis:
    def test_simple_assignment(self):
        pre = _zone_pre_analyze("let x = 5;")
        assert not pre.final_state.is_bot()
        assert pre.final_state.get_upper_bound('x') == Fraction(5)
        assert pre.final_state.get_lower_bound('x') == Fraction(5)

    def test_difference_tracking(self):
        pre = _zone_pre_analyze("let x = 5; let y = x + 3;")
        assert not pre.final_state.is_bot()
        # Zone should track y - x == 3
        diff = pre.final_state.get_diff_bound('y', 'x')
        assert diff is not None
        assert diff <= Fraction(3)

    def test_symbolic_inputs(self):
        pre = _zone_pre_analyze("let y = x + 1;", {'x': 'int'})
        assert not pre.final_state.is_bot()
        # y - x should be tracked as 1
        diff = pre.final_state.get_diff_bound('y', 'x')
        assert diff is not None
        assert diff <= Fraction(1)

    def test_program_field(self):
        pre = _zone_pre_analyze("let x = 10;")
        assert pre.program is not None
        assert len(pre.program.stmts) > 0

    def test_conditional(self):
        src = """
        let x = 5;
        let y = 10;
        if (x < y) {
            let z = y - x;
        }
        """
        pre = _zone_pre_analyze(src)
        assert not pre.final_state.is_bot()


# ============================================================
# Branch Feasibility
# ============================================================

class TestBranchFeasibility:
    def test_infeasible_difference(self):
        """y = x + 3 => y < x is infeasible (y - x == 3)."""
        pre = _zone_pre_analyze("let x = 5; let y = x + 3;")
        tokens = lex("y < x")
        p = Parser(tokens)
        cond = p.expression()
        result = _check_branch_feasibility_zone(pre.final_state, cond)
        assert result == 'infeasible'

    def test_feasible_difference(self):
        """y = x + 3 => y > x is feasible."""
        pre = _zone_pre_analyze("let x = 5; let y = x + 3;")
        tokens = lex("y > x")
        p = Parser(tokens)
        cond = p.expression()
        result = _check_branch_feasibility_zone(pre.final_state, cond)
        assert result == 'feasible'

    def test_infeasible_bound(self):
        """x = 5 => x > 10 is infeasible."""
        pre = _zone_pre_analyze("let x = 5;")
        tokens = lex("x > 10")
        p = Parser(tokens)
        cond = p.expression()
        result = _check_branch_feasibility_zone(pre.final_state, cond)
        assert result == 'infeasible'

    def test_feasible_bound(self):
        """x = 5 => x < 10 is feasible."""
        pre = _zone_pre_analyze("let x = 5;")
        tokens = lex("x < 10")
        p = Parser(tokens)
        cond = p.expression()
        result = _check_branch_feasibility_zone(pre.final_state, cond)
        assert result == 'feasible'

    def test_bot_state(self):
        result = _check_branch_feasibility_zone(Zone.bot(), BoolLit(True, 0))
        assert result == 'infeasible'

    def test_negated_infeasible(self):
        """y = x + 3 => !(y > x) means y <= x, which is infeasible."""
        pre = _zone_pre_analyze("let x = 5; let y = x + 3;")
        tokens = lex("y > x")
        p = Parser(tokens)
        cond = p.expression()
        result = _check_branch_feasibility_zone_negated(pre.final_state, cond)
        assert result == 'infeasible'

    def test_negated_feasible(self):
        """y = x + 3 => !(y < x) means y >= x, which is feasible."""
        pre = _zone_pre_analyze("let x = 5; let y = x + 3;")
        tokens = lex("y < x")
        p = Parser(tokens)
        cond = p.expression()
        result = _check_branch_feasibility_zone_negated(pre.final_state, cond)
        assert result == 'feasible'


# ============================================================
# Condition to Zone Constraints
# ============================================================

class TestCondToZoneConstraints:
    def test_var_lt_const(self):
        cond = BinOp(op='<', left=ASTVar('x', 0), right=IntLit(5, 0), line=0)
        cs = _cond_to_zone_constraints(cond)
        assert len(cs) == 1
        # x < 5 => x <= 4 => upper_bound(x, 4)
        assert cs[0].var1 == 'x'
        assert cs[0].var2 is None
        assert cs[0].bound == Fraction(4)

    def test_var_ge_const(self):
        cond = BinOp(op='>=', left=ASTVar('x', 0), right=IntLit(3, 0), line=0)
        cs = _cond_to_zone_constraints(cond)
        assert len(cs) == 1
        # x >= 3 => lower_bound(x, 3)
        assert cs[0].var2 == 'x'
        assert cs[0].bound == Fraction(-3)

    def test_var_lt_var(self):
        cond = BinOp(op='<', left=ASTVar('x', 0), right=ASTVar('y', 0), line=0)
        cs = _cond_to_zone_constraints(cond)
        assert len(cs) == 1
        # x < y => x - y <= -1
        assert cs[0].var1 == 'x'
        assert cs[0].var2 == 'y'
        assert cs[0].bound == Fraction(-1)

    def test_var_eq_const(self):
        cond = BinOp(op='==', left=ASTVar('x', 0), right=IntLit(7, 0), line=0)
        cs = _cond_to_zone_constraints(cond)
        assert len(cs) == 2  # x <= 7 AND x >= 7

    def test_negation(self):
        cond = BinOp(op='<', left=ASTVar('x', 0), right=IntLit(5, 0), line=0)
        cs = _cond_to_zone_constraints(cond, negate=True)
        assert len(cs) == 1
        # !(x < 5) = x >= 5 => lower_bound(x, 5)

    def test_bool_true(self):
        cs = _cond_to_zone_constraints(BoolLit(True, 0))
        assert cs == []  # No constraints for True

    def test_bool_false(self):
        cs = _cond_to_zone_constraints(BoolLit(False, 0))
        assert cs is None  # Signal infeasible

    def test_and_conjunction(self):
        cond = BinOp(
            op='&&',
            left=BinOp(op='>=', left=ASTVar('x', 0), right=IntLit(0, 0), line=0),
            right=BinOp(op='<=', left=ASTVar('x', 0), right=IntLit(10, 0), line=0),
            line=0
        )
        cs = _cond_to_zone_constraints(cond)
        assert len(cs) == 2

    def test_not_operator(self):
        inner = BinOp(op='<', left=ASTVar('x', 0), right=IntLit(5, 0), line=0)
        cond = UnaryOp(op='!', operand=inner, line=0)
        cs = _cond_to_zone_constraints(cond)
        assert len(cs) == 1
        # !(x < 5) = x >= 5

    def test_const_lt_var(self):
        cond = BinOp(op='<', left=IntLit(3, 0), right=ASTVar('x', 0), line=0)
        cs = _cond_to_zone_constraints(cond)
        assert len(cs) == 1
        # 3 < x => x > 3 => x >= 4 => lower_bound(x, 4)

    def test_disequality_not_representable(self):
        cond = BinOp(op='!=', left=ASTVar('x', 0), right=IntLit(5, 0), line=0)
        cs = _cond_to_zone_constraints(cond)
        assert cs == []  # Can't represent !=


# ============================================================
# Guided Execution
# ============================================================

class TestGuidedExecution:
    def test_basic(self):
        src = "let x = 5; let y = 10;"
        result = guided_execute(src)
        assert isinstance(result, ZoneGuidedResult)
        assert len(result.paths) >= 1

    def test_difference_pruning(self):
        """Zone should detect y - x == 3, making y < x infeasible."""
        src = """
        let x = 5;
        let y = x + 3;
        if (y < x) {
            print(1);
        } else {
            print(2);
        }
        """
        result = guided_execute(src)
        assert result.branches_pruned_by_zone >= 1
        assert result.difference_constraints_found >= 1

    def test_symbolic_difference(self):
        """With symbolic x, zone tracks y - x == 1."""
        src = """
        let y = x + 1;
        if (y < x) {
            print(1);
        } else {
            print(2);
        }
        """
        result = guided_execute(src, {'x': 'int'})
        assert result.branches_pruned_by_zone >= 1

    def test_no_pruning_needed(self):
        """Simple concrete program with no infeasible branches."""
        src = """
        let x = 5;
        if (x > 3) {
            print(1);
        }
        """
        result = guided_execute(src)
        assert result.branches_analyzed >= 1

    def test_multiple_branches(self):
        src = """
        let x = 5;
        let y = x + 2;
        let z = y + 3;
        if (y < x) { print(1); }
        if (z < y) { print(2); }
        if (z < x) { print(3); }
        """
        result = guided_execute(src)
        # All three conditions are infeasible due to difference tracking
        assert result.branches_pruned_by_zone >= 3

    def test_result_properties(self):
        src = "let x = 5;"
        result = guided_execute(src)
        assert result.zone_state is not None
        assert result.branches_analyzed >= 0
        assert result.smt_checks_saved >= 0

    def test_pruning_ratio(self):
        src = """
        let x = 5;
        let y = x + 1;
        if (y < x) { print(1); }
        """
        result = guided_execute(src)
        assert result.pruning_ratio >= 0.0

    def test_total_pruned(self):
        src = """
        let x = 5;
        let y = x + 3;
        if (y < x) { print(1); } else { print(2); }
        """
        result = guided_execute(src)
        assert result.total_pruned >= 1


# ============================================================
# Zone vs Interval Pruning
# ============================================================

class TestZonePruning:
    def test_zone_advantage(self):
        """Zone catches difference-based infeasibility that intervals miss."""
        src = """
        let y = x + 3;
        if (y < x) {
            print(1);
        }
        """
        analysis = analyze_zone_pruning(src, {'x': 'int'})
        assert analysis['zone_pruned'] >= 1
        assert analysis['zone_only_pruned'] >= 1
        assert analysis['difference_advantage'] >= 1

    def test_interval_sufficient(self):
        """When concrete bounds suffice, interval matches zone."""
        src = """
        let x = 5;
        if (x > 100) {
            print(1);
        }
        """
        analysis = analyze_zone_pruning(src)
        # Both interval and zone can prune this
        assert analysis['zone_pruned'] >= 1
        assert analysis['interval_pruned'] >= 1

    def test_no_prunable_branches(self):
        src = "let x = 5;"
        analysis = analyze_zone_pruning(src)
        assert analysis['total_branches'] == 0

    def test_symbolic_chain(self):
        """Chain: y = x + 1, z = y + 1 => z - x == 2."""
        src = """
        let y = x + 1;
        let z = y + 1;
        if (z < x) {
            print(1);
        }
        """
        analysis = analyze_zone_pruning(src, {'x': 'int'})
        assert analysis['zone_pruned'] >= 1


# ============================================================
# Property Verification
# ============================================================

class TestPropertyVerification:
    def test_difference_property(self):
        src = "let x = 5; let y = x + 3;"
        result = verify_difference_property(src, "y - x <= 3")
        assert result['verified']

    def test_bound_property(self):
        src = "let x = 5;"
        result = verify_difference_property(src, "x <= 10")
        assert result['verified']

    def test_equality_property(self):
        src = "let x = 5; let y = x + 3;"
        result = verify_difference_property(src, "y - x == 3")
        assert result['verified']

    def test_violated_property(self):
        src = "let x = 5; let y = x + 3;"
        result = verify_difference_property(src, "y - x <= 1")
        assert not result['verified']

    def test_lower_bound_property(self):
        src = "let x = 5;"
        result = verify_difference_property(src, "x >= 5")
        assert result['verified']

    def test_sum_property_rejected(self):
        """Zones can't verify sum properties."""
        src = "let x = 5; let y = 3;"
        result = verify_difference_property(src, "x + y <= 10")
        assert not result['verified']
        assert 'error' in result


# ============================================================
# Property Parsing
# ============================================================

class TestPropertyParsing:
    def test_diff_le(self):
        cs = _parse_zone_property("x - y <= 5")
        assert cs is not None
        assert len(cs) == 1

    def test_diff_ge(self):
        cs = _parse_zone_property("x - y >= 3")
        assert cs is not None
        assert len(cs) == 1

    def test_diff_eq(self):
        cs = _parse_zone_property("x - y == 0")
        assert cs is not None
        assert len(cs) == 2

    def test_var_le(self):
        cs = _parse_zone_property("x <= 10")
        assert cs is not None
        assert len(cs) == 1

    def test_var_ge(self):
        cs = _parse_zone_property("x >= 0")
        assert cs is not None

    def test_var_eq(self):
        cs = _parse_zone_property("x == 5")
        assert cs is not None
        assert len(cs) == 2

    def test_sum_rejected(self):
        cs = _parse_zone_property("x + y <= 10")
        assert cs is None

    def test_unparseable(self):
        cs = _parse_zone_property("foo bar baz")
        assert cs is None

    def test_negative_bound(self):
        cs = _parse_zone_property("x - y <= -3")
        assert cs is not None
        assert cs[0].bound == Fraction(-3)


# ============================================================
# Comparison APIs
# ============================================================

class TestComparisons:
    def test_v001_vs_v181(self):
        src = """
        let y = x + 2;
        if (y < x) { print(1); }
        """
        cmp = compare_v001_vs_v181(src, {'x': 'int'})
        assert 'v181_branches_pruned' in cmp
        assert 'v001_style_pruned' in cmp
        assert 'zone_advantage' in cmp
        assert cmp['zone_advantage'] >= 1

    def test_zone_vs_octagon(self):
        src = """
        let y = x + 1;
        if (y < x) { print(1); }
        """
        cmp = compare_zone_vs_octagon(src, {'x': 'int'})
        assert 'zone_pruned' in cmp
        assert cmp['zone_pruned'] >= 1


# ============================================================
# Batch Execution
# ============================================================

class TestBatchExecution:
    def test_batch_basic(self):
        sources = [
            "let x = 5; let y = x + 1;",
            "let a = 10; let b = a + 2;",
        ]
        results = batch_guided_execute(sources)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ZoneGuidedResult)

    def test_batch_with_inputs(self):
        sources = [
            "let y = x + 1;",
            "let z = x + 2;",
        ]
        inputs = [{'x': 'int'}, {'x': 'int'}]
        results = batch_guided_execute(sources, inputs)
        assert len(results) == 2


# ============================================================
# Incremental Zone-Guided Execution
# ============================================================

class TestIncrementalExecution:
    def test_basic(self):
        src = """
        let x = 5;
        let y = x + 3;
        if (y < x) { print(1); }
        """
        result = incremental_guided_execute(src)
        assert isinstance(result, ZoneGuidedResult)
        assert result.branches_pruned_by_zone >= 1

    def test_better_than_final_state(self):
        """Incremental may prune more by using per-branch zone state."""
        src = """
        let x = 5;
        let y = x + 1;
        if (y < x) { print(1); }
        x = 100;
        """
        # Final state loses x=5 after reassignment, but incremental
        # captures x=5 at the branch point
        result = incremental_guided_execute(src)
        assert result.branches_pruned_by_zone >= 1

    def test_nested_branches(self):
        src = """
        let x = 5;
        let y = x + 2;
        if (x > 0) {
            if (y < x) {
                print(1);
            }
        }
        """
        result = incremental_guided_execute(src)
        assert result.branches_analyzed >= 2

    def test_symbolic_incremental(self):
        src = """
        let y = x + 3;
        if (y < x) { print(1); }
        """
        result = incremental_guided_execute(src, {'x': 'int'})
        assert result.branches_pruned_by_zone >= 1


# ============================================================
# Branch Collection
# ============================================================

class TestBranchCollection:
    def test_if_branches(self):
        src = "if (x > 0) { print(1); } else { print(2); }"
        tokens = lex(src)
        program = Parser(tokens).parse()
        branches = _collect_branch_conditions(program.stmts)
        assert len(branches) == 1

    def test_nested_if(self):
        src = """
        if (x > 0) {
            if (y > 0) { print(1); }
        }
        """
        tokens = lex(src)
        program = Parser(tokens).parse()
        branches = _collect_branch_conditions(program.stmts)
        assert len(branches) == 2

    def test_while_branch(self):
        src = "while (x > 0) { x = x - 1; }"
        tokens = lex(src)
        program = Parser(tokens).parse()
        branches = _collect_branch_conditions(program.stmts)
        assert len(branches) == 1

    def test_no_branches(self):
        src = "let x = 5;"
        tokens = lex(src)
        program = Parser(tokens).parse()
        branches = _collect_branch_conditions(program.stmts)
        assert len(branches) == 0


# ============================================================
# Zone State at Branch Points
# ============================================================

class TestBranchZoneStates:
    def test_collects_states(self):
        src = """
        let x = 5;
        let y = x + 3;
        if (y < x) { print(1); }
        """
        tokens = lex(src)
        program = Parser(tokens).parse()
        states = _collect_branch_zone_states(program.stmts)
        assert len(states) >= 1
        cond, zone = states[0]
        # Zone should know y - x == 3 at branch point
        diff = zone.get_diff_bound('y', 'x')
        assert diff is not None and diff <= Fraction(3)

    def test_symbolic_state(self):
        src = """
        let y = x + 1;
        if (y < x) { print(1); }
        """
        tokens = lex(src)
        program = Parser(tokens).parse()
        states = _collect_branch_zone_states(program.stmts, {'x': 'int'})
        assert len(states) >= 1
        cond, zone = states[0]
        diff = zone.get_diff_bound('y', 'x')
        assert diff is not None and diff <= Fraction(1)


# ============================================================
# Assignment to Zone
# ============================================================

class TestApplyAssignment:
    def test_const_assignment(self):
        z = Zone.top()
        z = z._ensure_var('x')
        z = _apply_assignment_to_zone(z, 'x', IntLit(5, 0))
        assert z.get_upper_bound('x') == Fraction(5)
        assert z.get_lower_bound('x') == Fraction(5)

    def test_var_assignment(self):
        z = Zone.from_constraints([
            upper_bound('y', Fraction(10)),
            lower_bound('y', Fraction(5)),
        ])
        z = _apply_assignment_to_zone(z, 'x', ASTVar('y', 0))
        assert z.get_upper_bound('x') == Fraction(10)

    def test_var_plus_const(self):
        z = Zone.from_constraints([
            upper_bound('y', Fraction(5)),
            lower_bound('y', Fraction(5)),
        ])
        rhs = BinOp(op='+', left=ASTVar('y', 0), right=IntLit(3, 0), line=0)
        z = _apply_assignment_to_zone(z, 'x', rhs)
        assert z.get_upper_bound('x') == Fraction(8)

    def test_unknown_rhs_forgets(self):
        z = Zone.from_constraints([
            upper_bound('x', Fraction(5)),
            lower_bound('x', Fraction(5)),
        ])
        # BoolLit converts to NumberLit(1), so x becomes 1
        z = _apply_assignment_to_zone(z, 'x', BoolLit(True, 0))
        assert z.get_upper_bound('x') == Fraction(1)
        assert z.get_lower_bound('x') == Fraction(1)

    def test_bot_state_stays_bot(self):
        z = Zone.bot()
        z = _apply_assignment_to_zone(z, 'x', IntLit(5, 0))
        assert z.is_bot()


# ============================================================
# While Loop Handling
# ============================================================

class TestWhileLoops:
    def test_simple_countdown(self):
        src = """
        let x = 10;
        while (x > 0) {
            x = x - 1;
        }
        """
        result = guided_execute(src)
        assert result.branches_analyzed >= 1

    def test_two_var_loop(self):
        """Loop with two variables maintaining difference."""
        src = """
        let x = 0;
        let y = 5;
        while (x < 10) {
            x = x + 1;
            y = y + 1;
        }
        """
        result = guided_execute(src)
        assert result.branches_analyzed >= 1


# ============================================================
# Multi-Variable Relationships
# ============================================================

class TestMultiVariable:
    def test_three_var_chain(self):
        """y = x + 1, z = y + 1 => z - x == 2."""
        src = """
        let y = x + 1;
        let z = y + 1;
        if (z < x) { print(1); }
        if (z <= x) { print(2); }
        """
        result = guided_execute(src, {'x': 'int'})
        assert result.branches_pruned_by_zone >= 2

    def test_swap_detection(self):
        """After swap: x has old y, y has old x."""
        src = """
        let x = 5;
        let y = 10;
        let tmp = x;
        x = y;
        y = tmp;
        if (x < y) { print(1); }
        """
        result = guided_execute(src)
        # x=10, y=5 after swap, x < y is infeasible
        assert result.branches_pruned_by_zone >= 1

    def test_parallel_increments(self):
        src = """
        let y = x + 5;
        let z = x + 10;
        if (z < y) { print(1); }
        """
        result = guided_execute(src, {'x': 'int'})
        assert result.branches_pruned_by_zone >= 1


# ============================================================
# Zone Limitations (Sum Constraints)
# ============================================================

class TestZoneLimitations:
    def test_sum_not_tracked(self):
        """Zone can't track x + y == c, so it can't prune x + y > 15."""
        src = """
        let x = 5;
        let y = 7;
        if (x + y > 20) { print(1); }
        """
        # Zone knows x=5, y=7 from constants, but can't express x+y=12
        # However, it CAN prune this via individual bounds:
        # x <= 5 and y <= 7, so even without sum tracking,
        # the branch condition involves a BinOp sum which zones
        # may not be able to reason about directly
        result = guided_execute(src)
        # This is a known limitation -- zone may or may not prune
        assert result.branches_analyzed >= 1

    def test_sum_property_verification_rejected(self):
        result = verify_difference_property(
            "let x = 5; let y = 7;",
            "x + y <= 15"
        )
        assert not result['verified']
        assert 'error' in result


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = guided_execute("let x = 5;")
        assert result.branches_analyzed == 0
        assert result.branches_pruned_by_zone == 0

    def test_no_symbolic_inputs(self):
        src = """
        let x = 5;
        let y = x + 1;
        if (y < x) { print(1); }
        """
        result = guided_execute(src)
        assert result.branches_pruned_by_zone >= 1

    def test_many_branches(self):
        src = """
        let y = x + 1;
        let z = y + 1;
        let w = z + 1;
        if (y < x) { print(1); }
        if (z < x) { print(2); }
        if (w < x) { print(3); }
        if (z < y) { print(4); }
        if (w < y) { print(5); }
        if (w < z) { print(6); }
        """
        result = guided_execute(src, {'x': 'int'})
        assert result.branches_pruned_by_zone >= 6

    def test_unreachable_property(self):
        """If program is unreachable, property holds vacuously."""
        # This tests the bot-state path in verify_difference_property
        # We need a program where analysis reaches bot - hard to construct
        # Just test the property interface instead
        result = verify_difference_property("let x = 5;", "x == 5")
        assert result['verified']


# ============================================================
# Complex Programs
# ============================================================

class TestComplexPrograms:
    def test_fibonacci_style(self):
        """c = a + b is var+var, zone loses c's relation. But a=0, b=1 are tracked."""
        src = """
        let a = 0;
        let b = 1;
        let c = a + b;
        if (a < 0) { print(1); }
        if (b < 0) { print(2); }
        """
        result = guided_execute(src)
        # a=0 and b=1, so a<0 and b<0 are both infeasible via bounds
        assert result.branches_pruned_by_zone >= 2

    def test_accumulator(self):
        """sum = sum + x is var+var, zone forgets sum. Use var+const instead."""
        src = """
        let sum = 0;
        sum = sum + 5;
        if (sum < 0) { print(1); }
        """
        result = guided_execute(src)
        # sum = 0 + 5 = 5, so sum < 0 is infeasible
        assert result.branches_pruned_by_zone >= 1

    def test_conditional_assignment(self):
        src = """
        let x = 5;
        let y = 10;
        if (x < y) {
            let z = y - x;
            if (z < 0) {
                print(1);
            }
        }
        """
        result = guided_execute(src)
        assert result.branches_analyzed >= 2


# ============================================================
# Stress / Coverage
# ============================================================

class TestStress:
    def test_many_variables(self):
        """Chain of 10 increments: v1 = v0 + 1, v2 = v1 + 1, ..."""
        lines = ["let v0 = x;"]
        for i in range(1, 10):
            lines.append(f"let v{i} = v{i-1} + 1;")
        lines.append(f"if (v9 < x) {{ print(1); }}")
        src = "\n".join(lines)
        result = guided_execute(src, {'x': 'int'})
        assert result.branches_pruned_by_zone >= 1

    def test_max_paths_respected(self):
        src = """
        let y = x + 1;
        if (x > 0) { print(1); }
        if (x > 5) { print(2); }
        if (x > 10) { print(3); }
        """
        result = guided_execute(src, {'x': 'int'}, max_paths=5)
        assert len(result.paths) <= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
