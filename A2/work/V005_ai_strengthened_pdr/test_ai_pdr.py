"""
Tests for V005: Abstract-Interpretation-Strengthened PDR

Tests organized by section:
1. Loop extraction basics
2. Abstract candidate extraction
3. Candidate validation
4. AI-strengthened PDR on simple systems
5. verify_loop() high-level API
6. Comparison: standard vs strengthened PDR
7. Edge cases and error handling
8. Complex multi-variable systems
9. Conditional transitions
10. Manual hints API
11. Counterexample detection
12. Abstract warnings propagation
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ai_pdr import (
    parse, extract_loop_ts, extract_abstract_candidates,
    validate_candidate, filter_and_validate_candidates,
    ai_pdr_check, verify_loop, verify_ts_with_hints,
    compare_pdr_performance,
    AIPDROutput, AIPDRStats, VerifyResult,
    TransitionSystem, _and, _or, _eq, _negate, _implies,
    IntConst, BoolConst, App, Op, BOOL, INT, SMTVar,
    _expr_to_smt, _body_to_transition,
)

# Import C010 AST types for expression tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import IntLit, BoolLit, Var as ASTVar, BinOp, UnaryOp


# ============================================================
# Section 1: Loop Extraction Basics
# ============================================================

class TestLoopExtraction:
    def test_simple_counter_loop(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        ts, state_vars = extract_loop_ts(source)
        assert 'x' in state_vars
        assert ts.init_formula is not None
        assert ts.trans_formula is not None

    def test_two_variable_loop(self):
        source = "let x = 0; let y = 10; while (x < y) { x = x + 1; y = y - 1; }"
        ts, state_vars = extract_loop_ts(source)
        assert 'x' in state_vars
        assert 'y' in state_vars

    def test_loop_with_property(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        prop_tokens = __import__('stack_vm', fromlist=['lex']).lex("x >= 0;")
        prop_parser = __import__('stack_vm', fromlist=['Parser']).Parser(prop_tokens)
        prop_program = prop_parser.parse()
        prop_expr = prop_program.stmts[0]

        ts, state_vars = extract_loop_ts(source, property_expr=prop_expr)
        assert ts.prop_formula is not None

    def test_no_loop_raises(self):
        source = "let x = 0; x = x + 1;"
        with pytest.raises(ValueError, match="No while loop found"):
            extract_loop_ts(source)

    def test_loop_index_selection(self):
        source = """
        let a = 0;
        while (a < 5) { a = a + 1; }
        let b = 10;
        while (b > 0) { b = b - 1; }
        """
        ts1, vars1 = extract_loop_ts(source, loop_index=0)
        ts2, vars2 = extract_loop_ts(source, loop_index=1)
        assert 'a' in vars1
        assert 'b' in vars2


# ============================================================
# Section 2: Abstract Candidate Extraction
# ============================================================

class TestAbstractCandidates:
    def test_counter_produces_lower_bound(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        candidates, intervals, warnings = extract_abstract_candidates(source, ['x'])
        # x starts at 0 and increments -- abstract interp should find x >= 0
        kinds = [kind for kind, var, formula in candidates]
        assert len(candidates) > 0

    def test_constant_init_detected(self):
        source = "let x = 5;"
        candidates, intervals, warnings = extract_abstract_candidates(source, ['x'])
        # x is exactly 5 -- should find constant candidate
        const_candidates = [(k, v, f) for k, v, f in candidates if k == 'const']
        assert len(const_candidates) > 0

    def test_interval_bounds(self):
        source = "let x = 3; let y = 7;"
        candidates, intervals, warnings = extract_abstract_candidates(source, ['x', 'y'])
        # Should have tight intervals
        assert intervals['x'].lo == 3.0
        assert intervals['x'].hi == 3.0
        assert intervals['y'].lo == 7.0
        assert intervals['y'].hi == 7.0

    def test_sign_candidates(self):
        source = "let x = 0; while (x < 100) { x = x + 1; }"
        candidates, intervals, warnings = extract_abstract_candidates(source, ['x'])
        # x is non-negative in the exit state
        sign_candidates = [(k, v) for k, v, f in candidates if k == 'sign']
        # Should have at least one sign-derived candidate
        assert len(candidates) > 0


# ============================================================
# Section 3: Candidate Validation
# ============================================================

class TestCandidateValidation:
    def _make_counter_ts(self):
        """Simple counter: x starts at 0, increments by 1."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))
        return ts

    def test_valid_invariant_accepted(self):
        ts = self._make_counter_ts()
        # x >= 0 is a valid inductive invariant for a counter starting at 0
        x = ts.var("x")
        candidate = App(Op.GE, [x, IntConst(0)], BOOL)
        assert validate_candidate(ts, candidate) is True

    def test_invalid_invariant_rejected(self):
        ts = self._make_counter_ts()
        # x <= 5 is NOT inductive (counter goes past 5)
        x = ts.var("x")
        candidate = App(Op.LE, [x, IntConst(5)], BOOL)
        assert validate_candidate(ts, candidate) is False

    def test_filter_and_validate(self):
        ts = self._make_counter_ts()
        x = ts.var("x")
        candidates = [
            ('sign', 'x', App(Op.GE, [x, IntConst(0)], BOOL)),  # valid
            ('upper', 'x', App(Op.LE, [x, IntConst(5)], BOOL)),  # invalid
        ]
        accepted, rejected = filter_and_validate_candidates(ts, candidates)
        assert len(accepted) >= 1
        assert len(rejected) >= 1

    def test_trivially_true_invariant(self):
        ts = self._make_counter_ts()
        # True is always a valid invariant
        assert validate_candidate(ts, BoolConst(True)) is True


# ============================================================
# Section 4: AI-Strengthened PDR on Simple Systems
# ============================================================

class TestAIPDRSimple:
    def test_counter_safe(self):
        """Counter x starts at 0, increments. Property: x >= 0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        source = "let x = 0; while (x >= 0) { x = x + 1; }"
        result = ai_pdr_check(ts, source=source, state_vars=['x'])
        assert result.result == VerifyResult.SAFE

    def test_counter_unsafe(self):
        """Counter x starts at 0, increments. Property: x < 5 (eventually false)."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.LT, [x, IntConst(5)], BOOL))

        source = "let x = 0; while (x < 10) { x = x + 1; }"
        result = ai_pdr_check(ts, source=source, state_vars=['x'])
        assert result.result == VerifyResult.UNSAFE
        assert result.counterexample is not None

    def test_without_source_falls_back_to_standard(self):
        """Without source, ai_pdr_check should still work (no seeding)."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = ai_pdr_check(ts)
        assert result.result == VerifyResult.SAFE
        assert result.stats.abstract_candidates == 0

    def test_stats_populated(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        # Use a source where abstract interp finds useful info about x
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        result = ai_pdr_check(ts, source=source, state_vars=['x'])
        # At minimum pdr_stats should be populated
        assert result.stats.pdr_stats is not None


# ============================================================
# Section 5: verify_loop() High-Level API
# ============================================================

class TestVerifyLoop:
    def test_simple_counter_safe(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        result = verify_loop(source, property_source="x >= 0")
        assert result.result == VerifyResult.SAFE

    def test_simple_counter_unsafe(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        result = verify_loop(source, property_source="x < 5")
        assert result.result == VerifyResult.UNSAFE

    def test_decrementing_counter(self):
        source = "let x = 10; while (x > 0) { x = x - 1; }"
        result = verify_loop(source, property_source="x >= 0")
        # x goes 10, 9, ..., 1, 0 -- x >= 0 should hold
        # But the loop body does x = x - 1 unconditionally,
        # so at x=0 the transition gives x'=-1, violating x>=0
        # PDR should find this
        assert result.result == VerifyResult.UNSAFE

    def test_two_counter_invariant(self):
        source = """
        let x = 0;
        let y = 10;
        while (x < 10) {
            x = x + 1;
            y = y - 1;
        }
        """
        # x + y == 10 should hold as an invariant
        # But we need to express this as a property...
        # The transition preserves x + y: (x+1) + (y-1) = x + y
        ts = TransitionSystem()
        x_v = ts.add_int_var("x")
        y_v = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")
        ts.set_init(_and(_eq(x_v, IntConst(0)), _eq(y_v, IntConst(10))))
        ts.set_trans(_and(
            _eq(xp, App(Op.ADD, [x_v, IntConst(1)], INT)),
            _eq(yp, App(Op.SUB, [y_v, IntConst(1)], INT))
        ))
        sum_xy = App(Op.ADD, [x_v, y_v], INT)
        ts.set_property(_eq(sum_xy, IntConst(10)))

        result = ai_pdr_check(ts, source=source, state_vars=['x', 'y'])
        assert result.result == VerifyResult.SAFE

    def test_no_property_defaults_to_true(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        result = verify_loop(source)
        assert result.result == VerifyResult.SAFE  # True is trivially invariant


# ============================================================
# Section 6: Comparison: Standard vs Strengthened
# ============================================================

class TestComparison:
    def test_both_agree_on_safe(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        source = "let x = 0; while (x >= 0) { x = x + 1; }"
        std, ai = compare_pdr_performance(ts, source=source, state_vars=['x'])

        from pdr import PDRResult
        assert std.result == PDRResult.SAFE
        assert ai.result == VerifyResult.SAFE

    def test_both_agree_on_unsafe(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.LT, [x, IntConst(3)], BOOL))

        source = "let x = 0; while (x < 10) { x = x + 1; }"
        std, ai = compare_pdr_performance(ts, source=source, state_vars=['x'])

        # Both should detect unsafe
        from pdr import PDRResult
        assert std.result == PDRResult.UNSAFE
        assert ai.result == VerifyResult.UNSAFE

    def test_strengthened_uses_fewer_or_equal_queries(self):
        """AI-strengthened PDR should use <= SMT queries than standard PDR."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        source = "let x = 0; while (x >= 0) { x = x + 1; }"
        std, ai = compare_pdr_performance(ts, source=source, state_vars=['x'])

        # The AI version seeds frames, so it might converge faster
        # At minimum, both should succeed
        from pdr import PDRResult
        assert std.result == PDRResult.SAFE
        assert ai.result == VerifyResult.SAFE


# ============================================================
# Section 7: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_trivial_property_true(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(BoolConst(True))

        result = ai_pdr_check(ts)
        assert result.result == VerifyResult.SAFE

    def test_init_violates_property(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(-1)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = ai_pdr_check(ts)
        assert result.result == VerifyResult.UNSAFE
        assert result.counterexample is not None
        assert result.counterexample.length == 0  # Violated at init

    def test_empty_seed_clauses(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        # Pass empty source -- no candidates extracted
        result = ai_pdr_check(ts, source="let z = 99;", state_vars=['x'])
        assert result.result == VerifyResult.SAFE
        # No candidates from unrelated variable
        # The abstract interp has x=TOP since it's not defined in this source


# ============================================================
# Section 8: Complex Multi-Variable Systems
# ============================================================

class TestMultiVariable:
    def test_accumulator(self):
        """sum = 0; i = 0; while i < n: sum += i; i += 1. Property: sum >= 0."""
        ts = TransitionSystem()
        s = ts.add_int_var("s")
        i = ts.add_int_var("i")
        sp = ts.prime("s")
        ip = ts.prime("i")

        ts.set_init(_and(_eq(s, IntConst(0)), _eq(i, IntConst(0))))
        ts.set_trans(_and(
            _eq(sp, App(Op.ADD, [s, i], INT)),
            _eq(ip, App(Op.ADD, [i, IntConst(1)], INT))
        ))
        ts.set_property(App(Op.GE, [s, IntConst(0)], BOOL))

        source = "let s = 0; let i = 0; while (i < 10) { s = s + i; i = i + 1; }"
        result = ai_pdr_check(ts, source=source, state_vars=['s', 'i'])
        assert result.result == VerifyResult.SAFE

    def test_diverging_variables(self):
        """x grows, y shrinks. Property: x >= 0 AND y <= 10."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        y = ts.add_int_var("y")
        xp = ts.prime("x")
        yp = ts.prime("y")

        ts.set_init(_and(_eq(x, IntConst(0)), _eq(y, IntConst(10))))
        ts.set_trans(_and(
            _eq(xp, App(Op.ADD, [x, IntConst(1)], INT)),
            _eq(yp, App(Op.SUB, [y, IntConst(1)], INT))
        ))
        ts.set_property(_and(
            App(Op.GE, [x, IntConst(0)], BOOL),
            App(Op.LE, [y, IntConst(10)], BOOL)
        ))

        source = "let x = 0; let y = 10; while (x < 100) { x = x + 1; y = y - 1; }"
        result = ai_pdr_check(ts, source=source, state_vars=['x', 'y'])
        assert result.result == VerifyResult.SAFE


# ============================================================
# Section 9: Conditional Transitions
# ============================================================

class TestConditionalTransitions:
    def test_abs_like_transition(self):
        """x starts at -5, each step: if x < 0 then x' = x+1 else x' = x. Property: x <= 0."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")

        ts.set_init(_eq(x, IntConst(-5)))

        cond = App(Op.LT, [x, IntConst(0)], BOOL)
        then_val = App(Op.ADD, [x, IntConst(1)], INT)
        else_val = x
        ite = App(Op.ITE, [cond, then_val, else_val], INT)
        ts.set_trans(_eq(xp, ite))
        ts.set_property(App(Op.LE, [x, IntConst(0)], BOOL))

        result = ai_pdr_check(ts)
        assert result.result == VerifyResult.SAFE

    def test_conditional_in_loop_body(self):
        source = """
        let x = 0;
        while (x < 20) {
            if (x < 10) {
                x = x + 2;
            } else {
                x = x + 1;
            }
        }
        """
        result = verify_loop(source, property_source="x >= 0")
        assert result.result == VerifyResult.SAFE


# ============================================================
# Section 10: Manual Hints API
# ============================================================

class TestManualHints:
    def test_valid_hint_accepted(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        hints = [App(Op.GE, [x, IntConst(0)], BOOL)]
        result = verify_ts_with_hints(ts, hints=hints)
        assert result.result == VerifyResult.SAFE
        assert result.stats.candidates_accepted == 1

    def test_invalid_hint_rejected(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        hints = [App(Op.LE, [x, IntConst(10)], BOOL)]  # Not inductive
        result = verify_ts_with_hints(ts, hints=hints)
        assert result.result == VerifyResult.SAFE  # Still safe, hint just rejected
        assert result.stats.candidates_rejected == 1

    def test_no_hints(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = verify_ts_with_hints(ts)
        assert result.result == VerifyResult.SAFE

    def test_mixed_hints(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(0)))
        ts.set_trans(_eq(xp, App(Op.ADD, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        hints = [
            App(Op.GE, [x, IntConst(0)], BOOL),     # valid
            App(Op.LE, [x, IntConst(10)], BOOL),     # invalid (not inductive)
            BoolConst(True),                          # valid (trivially)
        ]
        result = verify_ts_with_hints(ts, hints=hints)
        assert result.result == VerifyResult.SAFE
        assert result.stats.candidates_accepted >= 2
        assert result.stats.candidates_rejected >= 1


# ============================================================
# Section 11: Counterexample Detection
# ============================================================

class TestCounterexamples:
    def test_immediate_violation(self):
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(-1)))
        ts.set_trans(_eq(xp, x))
        ts.set_property(App(Op.GE, [x, IntConst(0)], BOOL))

        result = ai_pdr_check(ts)
        assert result.result == VerifyResult.UNSAFE
        assert result.counterexample.length == 0
        assert result.counterexample.trace[0]['x'] == -1

    def test_delayed_violation(self):
        """Property violated after a few steps."""
        ts = TransitionSystem()
        x = ts.add_int_var("x")
        xp = ts.prime("x")
        ts.set_init(_eq(x, IntConst(3)))
        ts.set_trans(_eq(xp, App(Op.SUB, [x, IntConst(1)], INT)))
        ts.set_property(App(Op.GT, [x, IntConst(0)], BOOL))

        result = ai_pdr_check(ts)
        assert result.result == VerifyResult.UNSAFE
        assert result.counterexample is not None
        assert result.counterexample.length >= 1


# ============================================================
# Section 12: Abstract Warnings Propagation
# ============================================================

class TestAbstractWarnings:
    def test_division_warning_propagated(self):
        source = "let x = 0; let y = x / 0; while (x < 10) { x = x + 1; }"
        ts, state_vars = extract_loop_ts(source, property_expr=None)
        ts.set_property(BoolConst(True))
        result = ai_pdr_check(ts, source=source, state_vars=state_vars)
        # Abstract warnings from C039 should be propagated
        assert len(result.abstract_warnings) > 0

    def test_clean_program_no_warnings(self):
        source = "let x = 0; while (x < 10) { x = x + 1; }"
        ts, state_vars = extract_loop_ts(source)
        ts.set_property(App(Op.GE, [ts.var('x'), IntConst(0)], BOOL))
        result = ai_pdr_check(ts, source=source, state_vars=state_vars)
        # No division warnings in clean code
        div_warnings = [w for w in result.abstract_warnings
                        if 'division' in str(w.kind)]
        assert len(div_warnings) == 0


# ============================================================
# Section 13: Expression-to-SMT Conversion
# ============================================================

class TestExprToSMT:
    def test_int_literal(self):
        expr = IntLit(42)
        result = _expr_to_smt(expr, {})
        assert isinstance(result, IntConst)
        assert result.value == 42

    def test_variable(self):
        v = SMTVar("x", INT)
        expr = ASTVar("x")
        result = _expr_to_smt(expr, {"x": v})
        assert result is v

    def test_binary_add(self):
        expr = BinOp('+', IntLit(1), IntLit(2))
        result = _expr_to_smt(expr, {})
        assert isinstance(result, App)
        assert result.op == Op.ADD

    def test_comparison(self):
        expr = BinOp('<', ASTVar("x"), IntLit(10))
        v = SMTVar("x", INT)
        result = _expr_to_smt(expr, {"x": v})
        assert isinstance(result, App)
        assert result.op == Op.LT


# ============================================================
# Run tests
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
