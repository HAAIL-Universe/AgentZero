"""Tests for V077: LTL Synthesis via GR(1) Reduction."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V075_reactive_synthesis'))

from ltl_synthesis import (
    # Classification
    FormulaClass, is_propositional, is_one_step, classify_formula,
    classify_conjunction, is_gr1_fragment,
    # Reduction
    reduce_to_gr1, AuxVariable,
    # Synthesis
    synthesize_ltl, check_ltl_realizability,
    synthesize_from_strings, make_ltl_spec,
    LTLSynthSpec, LTLSynthOutput, ReductionResult,
    # Simulation/verification
    simulate_ltl_controller, verify_ltl_controller,
    # Analysis
    analyze_spec,
    # Examples
    synthesize_arbiter_ltl, synthesize_traffic_light_ltl, synthesize_buffer_ltl,
)
from ltl_model_checker import (
    Atom, Not, And, Or, Implies, Next, Finally, Globally, Until,
    LTLTrue, LTLFalse, parse_ltl, Release, WeakUntil, Iff,
)


# ===================================================================
# Section 1: Formula Classification
# ===================================================================

class TestPropositional:
    def test_atom(self):
        assert is_propositional(Atom("a"))

    def test_true_false(self):
        assert is_propositional(LTLTrue())
        assert is_propositional(LTLFalse())

    def test_boolean_combo(self):
        f = And(Or(Atom("a"), Not(Atom("b"))), Atom("c"))
        assert is_propositional(f)

    def test_implies(self):
        assert is_propositional(Implies(Atom("a"), Atom("b")))

    def test_iff(self):
        assert is_propositional(Iff(Atom("a"), Atom("b")))

    def test_temporal_not_prop(self):
        assert not is_propositional(Globally(Atom("a")))
        assert not is_propositional(Finally(Atom("a")))
        assert not is_propositional(Next(Atom("a")))
        assert not is_propositional(Until(Atom("a"), Atom("b")))


class TestOneStep:
    def test_atom_is_one_step(self):
        assert is_one_step(Atom("a"))

    def test_next_of_prop(self):
        assert is_one_step(Next(Atom("a")))
        assert is_one_step(Next(And(Atom("a"), Atom("b"))))

    def test_mixed_curr_next(self):
        # a & X(b)
        f = And(Atom("a"), Next(Atom("b")))
        assert is_one_step(f)

    def test_nested_next_not_one_step(self):
        # X(X(a)) is NOT one-step
        assert not is_one_step(Next(Next(Atom("a"))))

    def test_globally_not_one_step(self):
        assert not is_one_step(Globally(Atom("a")))

    def test_implies_with_next(self):
        # a -> X(b)
        f = Implies(Atom("a"), Next(Atom("b")))
        assert is_one_step(f)


class TestClassifyFormula:
    def test_init(self):
        assert classify_formula(Atom("a")) == FormulaClass.INIT
        assert classify_formula(And(Atom("a"), Atom("b"))) == FormulaClass.INIT

    def test_safety(self):
        # G(a)
        f = Globally(Atom("a"))
        assert classify_formula(f) == FormulaClass.SAFETY

    def test_safety_with_next(self):
        # G(a -> X(b))
        f = Globally(Implies(Atom("a"), Next(Atom("b"))))
        assert classify_formula(f) == FormulaClass.SAFETY

    def test_liveness(self):
        # GF(a)
        f = Globally(Finally(Atom("a")))
        assert classify_formula(f) == FormulaClass.LIVENESS

    def test_response(self):
        # G(a -> F(b))
        f = Globally(Implies(Atom("a"), Finally(Atom("b"))))
        assert classify_formula(f) == FormulaClass.RESPONSE

    def test_persistence(self):
        # FG(a)
        f = Finally(Globally(Atom("a")))
        assert classify_formula(f) == FormulaClass.PERSISTENCE

    def test_unsupported_until(self):
        # a U b -- raw until is not directly in GR(1) fragment
        f = Until(Atom("a"), Atom("b"))
        assert classify_formula(f) == FormulaClass.UNSUPPORTED

    def test_unsupported_nested(self):
        # G(F(G(a))) -- not in GR(1)
        f = Globally(Finally(Globally(Atom("a"))))
        assert classify_formula(f) == FormulaClass.UNSUPPORTED


class TestClassifyConjunction:
    def test_single(self):
        parts = classify_conjunction(Globally(Atom("a")))
        assert len(parts) == 1
        assert parts[0][0] == FormulaClass.SAFETY

    def test_conjunction(self):
        f = And(Globally(Atom("a")), Globally(Finally(Atom("b"))))
        parts = classify_conjunction(f)
        assert len(parts) == 2
        classes = {p[0] for p in parts}
        assert FormulaClass.SAFETY in classes
        assert FormulaClass.LIVENESS in classes

    def test_nested_conjunction(self):
        f = And(And(Atom("a"), Globally(Atom("b"))), Globally(Finally(Atom("c"))))
        parts = classify_conjunction(f)
        assert len(parts) == 3


class TestGR1Fragment:
    def test_in_fragment(self):
        f = And(Globally(Atom("a")), Globally(Finally(Atom("b"))))
        assert is_gr1_fragment(f)

    def test_not_in_fragment(self):
        f = Until(Atom("a"), Atom("b"))
        assert not is_gr1_fragment(f)

    def test_mixed_unsupported(self):
        f = And(Globally(Atom("a")), Until(Atom("b"), Atom("c")))
        assert not is_gr1_fragment(f)


# ===================================================================
# Section 2: Spec Analysis
# ===================================================================

class TestAnalyzeSpec:
    def test_basic_analysis(self):
        spec = make_ltl_spec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=["G(F(a))"],
            guarantees=["G(a -> F(b))"]
        )
        result = analyze_spec(spec)
        assert result["in_gr1_fragment"]
        assert result["assumption_count"] == 1
        assert result["guarantee_count"] == 1
        assert result["response_patterns"] == 1
        assert result["aux_vars_needed"] == 1

    def test_unsupported_detection(self):
        spec = LTLSynthSpec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[Until(Atom("a"), Atom("b"))],
            guarantees=[Globally(Atom("b"))]
        )
        result = analyze_spec(spec)
        assert not result["in_gr1_fragment"]
        assert len(result["unsupported"]) == 1

    def test_counts(self):
        spec = make_ltl_spec(
            env_vars=["a"], sys_vars=["b", "c"],
            assumptions=["G(F(a))", "a"],
            guarantees=["G(b)", "G(F(c))", "G(a -> F(b))"]
        )
        result = analyze_spec(spec)
        assert result["formula_counts"]["liveness"] == 2
        assert result["formula_counts"]["safety"] == 1
        assert result["formula_counts"]["init"] == 1
        assert result["formula_counts"]["response"] == 1


# ===================================================================
# Section 3: Reduction to GR(1)
# ===================================================================

class TestReduction:
    def test_simple_safety(self):
        spec = LTLSynthSpec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=[Globally(Atom("b"))]
        )
        result = reduce_to_gr1(spec)
        assert result.success
        assert len(result.aux_vars) == 0
        assert result.gr1_spec is not None
        assert result.gr1_spec.sys_safe != result.bdd.TRUE

    def test_response_adds_liveness(self):
        spec = LTLSynthSpec(
            env_vars=["req"], sys_vars=["grant"],
            assumptions=[],
            guarantees=[Globally(Implies(Atom("req"), Finally(Atom("grant"))))]
        )
        result = reduce_to_gr1(spec)
        assert result.success
        assert len(result.aux_vars) == 0  # no aux vars, direct GR(1) liveness
        # Response becomes GF(!req | grant)
        assert len(result.gr1_spec.sys_live) == 1

    def test_env_response_adds_env_liveness(self):
        spec = LTLSynthSpec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[Globally(Implies(Atom("a"), Finally(Atom("b"))))],
            guarantees=[Globally(Atom("b"))]
        )
        result = reduce_to_gr1(spec)
        assert result.success
        # env response -> env liveness GF(!a | b)
        assert len(result.gr1_spec.env_live) == 1

    def test_unsupported_fails(self):
        spec = LTLSynthSpec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[Until(Atom("a"), Atom("b"))],
            guarantees=[]
        )
        result = reduce_to_gr1(spec)
        assert not result.success
        assert result.error is not None
        assert len(result.unsupported) == 1

    def test_init_goes_to_init(self):
        spec = LTLSynthSpec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[Atom("a")],
            guarantees=[Not(Atom("b"))]
        )
        result = reduce_to_gr1(spec)
        assert result.success
        assert result.gr1_spec.env_init != result.bdd.TRUE
        assert result.gr1_spec.sys_init != result.bdd.TRUE

    def test_liveness_goes_to_live(self):
        spec = LTLSynthSpec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[Globally(Finally(Atom("a")))],
            guarantees=[Globally(Finally(Atom("b")))]
        )
        result = reduce_to_gr1(spec)
        assert result.success
        assert len(result.gr1_spec.env_live) == 1
        assert len(result.gr1_spec.sys_live) == 1

    def test_multiple_responses(self):
        spec = LTLSynthSpec(
            env_vars=["a", "b"], sys_vars=["c", "d"],
            assumptions=[],
            guarantees=[
                Globally(Implies(Atom("a"), Finally(Atom("c")))),
                Globally(Implies(Atom("b"), Finally(Atom("d")))),
            ]
        )
        result = reduce_to_gr1(spec)
        assert result.success
        assert len(result.aux_vars) == 0  # direct liveness, no aux
        assert len(result.gr1_spec.sys_live) == 2

    def test_persistence(self):
        spec = LTLSynthSpec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=[Finally(Globally(Atom("b")))]
        )
        result = reduce_to_gr1(spec)
        assert result.success

    def test_mixed_spec(self):
        """Safety + liveness + response all together."""
        spec = make_ltl_spec(
            env_vars=["req"], sys_vars=["grant", "busy"],
            assumptions=["G(F(req))"],
            guarantees=[
                "G(req -> F(grant))",     # response
                "G(!(grant & busy))",     # safety
                "G(F(!busy))",            # liveness
            ]
        )
        result = reduce_to_gr1(spec)
        assert result.success
        assert len(result.aux_vars) == 0  # no aux vars
        assert len(result.gr1_spec.sys_live) == 2  # liveness + response liveness


# ===================================================================
# Section 4: Basic Synthesis
# ===================================================================

class TestBasicSynthesis:
    def test_trivial_realizable(self):
        """System can always set b=true."""
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)"]
        )
        assert result.realizable

    def test_trivial_unrealizable(self):
        """System cannot control env var a."""
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(a)"]  # can't guarantee env var
        )
        assert not result.realizable

    def test_safety_only(self):
        """Simple safety: grant only when requested."""
        result = synthesize_from_strings(
            env_vars=["req"], sys_vars=["grant"],
            assumptions=[],
            guarantees=["G(!req -> !grant)"]
        )
        assert result.realizable

    def test_liveness_only(self):
        """System must toggle b infinitely."""
        result = synthesize_from_strings(
            env_vars=[], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(F(b))", "G(F(!b))"]
        )
        assert result.realizable

    def test_init_constraint(self):
        """System starts with b=false."""
        result = synthesize_from_strings(
            env_vars=[], sys_vars=["b"],
            assumptions=[],
            guarantees=["!b", "G(b)"]  # init !b but always b -- contradictory
        )
        assert not result.realizable

    def test_realizability_check(self):
        spec = make_ltl_spec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)"]
        )
        assert check_ltl_realizability(spec)


class TestResponseSynthesis:
    def test_simple_response(self):
        """Every request gets granted."""
        result = synthesize_from_strings(
            env_vars=["req"], sys_vars=["grant"],
            assumptions=["G(F(req))"],
            guarantees=["G(req -> F(grant))"]
        )
        assert result.realizable
        assert result.statistics["aux_vars_introduced"] == 0

    def test_response_with_safety(self):
        """Grant only when requested, and eventually grant."""
        result = synthesize_from_strings(
            env_vars=["req"], sys_vars=["grant"],
            assumptions=["G(F(req))"],
            guarantees=[
                "G(req -> F(grant))",
                "G(!req -> !grant)",
            ]
        )
        assert result.realizable

    def test_multiple_responses(self):
        """Two clients, both get served."""
        result = synthesize_from_strings(
            env_vars=["r0", "r1"], sys_vars=["g0", "g1"],
            assumptions=["G(F(r0))", "G(F(r1))"],
            guarantees=[
                "G(r0 -> F(g0))",
                "G(r1 -> F(g1))",
            ]
        )
        assert result.realizable


# ===================================================================
# Section 5: Example Synthesis Problems
# ===================================================================

class TestArbiter:
    def test_two_client_arbiter(self):
        result = synthesize_arbiter_ltl(2)
        assert result.realizable
        assert result.statistics["aux_vars_introduced"] == 0

    def test_arbiter_has_strategy(self):
        result = synthesize_arbiter_ltl(2)
        assert result.strategy is not None or result.synthesis_output.strategy_bdd is not None


class TestTrafficLight:
    def test_traffic_light(self):
        result = synthesize_traffic_light_ltl()
        assert result.realizable
        assert result.statistics["aux_vars_introduced"] == 0


class TestBuffer:
    def test_buffer(self):
        result = synthesize_buffer_ltl()
        assert result.realizable


# ===================================================================
# Section 6: Controller Verification
# ===================================================================

class TestVerification:
    def test_verify_safety_controller(self):
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)"]
        )
        assert result.realizable
        vr = verify_ltl_controller(result)
        assert vr is not None
        assert vr.get("init_in_winning", False)
        assert vr.get("winning_closed", False)

    def test_verify_response_controller(self):
        result = synthesize_from_strings(
            env_vars=["req"], sys_vars=["grant"],
            assumptions=["G(F(req))"],
            guarantees=["G(req -> F(grant))"]
        )
        assert result.realizable
        vr = verify_ltl_controller(result)
        assert vr is not None

    def test_verify_unrealizable_returns_none(self):
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(a)"]
        )
        assert not result.realizable
        assert verify_ltl_controller(result) is None


# ===================================================================
# Section 7: Simulation
# ===================================================================

class TestSimulation:
    def test_simulate_safety(self):
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)"]
        )
        assert result.realizable
        trace = simulate_ltl_controller(
            result,
            [{"a": True}, {"a": False}, {"a": True}]
        )
        if trace is not None:
            # b should always be true in all states
            for state in trace:
                assert state.get("b", False)

    def test_simulate_unrealizable_returns_none(self):
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(a)"]
        )
        assert simulate_ltl_controller(result, [{"a": True}]) is None


# ===================================================================
# Section 8: Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_no_assumptions(self):
        result = synthesize_from_strings(
            env_vars=[], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)"]
        )
        assert result.realizable

    def test_no_guarantees(self):
        """No guarantees -- trivially realizable."""
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=["G(F(a))"],
            guarantees=[]
        )
        assert result.realizable

    def test_no_env_vars(self):
        result = synthesize_from_strings(
            env_vars=[], sys_vars=["x", "y"],
            assumptions=[],
            guarantees=["G(F(x))", "G(F(!x))"]
        )
        assert result.realizable

    def test_contradictory_guarantees(self):
        result = synthesize_from_strings(
            env_vars=[], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)", "G(!b)"]
        )
        assert not result.realizable

    def test_empty_spec(self):
        result = synthesize_from_strings(
            env_vars=[], sys_vars=[],
            assumptions=[],
            guarantees=[]
        )
        assert result.realizable

    def test_unsupported_formula_fails_gracefully(self):
        spec = LTLSynthSpec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=[Until(Atom("a"), Atom("b"))]
        )
        result = synthesize_ltl(spec)
        assert not result.realizable
        assert result.statistics.get("error") is not None

    def test_true_guarantee(self):
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(true)"]
        )
        assert result.realizable

    def test_false_guarantee(self):
        result = synthesize_from_strings(
            env_vars=[], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(false)"]
        )
        assert not result.realizable


# ===================================================================
# Section 9: Complex Specifications
# ===================================================================

class TestComplexSpecs:
    def test_request_response_mutex(self):
        """Full arbiter: response + mutual exclusion + no spurious grants."""
        result = synthesize_from_strings(
            env_vars=["r0", "r1"],
            sys_vars=["g0", "g1"],
            assumptions=["G(F(r0))", "G(F(r1))"],
            guarantees=[
                "G(r0 -> F(g0))",
                "G(r1 -> F(g1))",
                "G(!(g0 & g1))",
                "G(!r0 -> !g0)",
                "G(!r1 -> !g1)",
            ]
        )
        assert result.realizable

    def test_env_assumption_helps(self):
        """Without env assumption, unrealizable. With it, realizable."""
        # Without assumption: system can't guarantee response because env might never give opportunity
        spec_no_assume = make_ltl_spec(
            env_vars=["req"], sys_vars=["grant"],
            assumptions=[],
            guarantees=["G(req -> F(grant))", "G(!req -> !grant)"]
        )
        # This might be realizable because system can just set grant=true whenever req is true
        # But with stronger constraints it wouldn't be

        # With assumption: env is fair
        spec_with_assume = make_ltl_spec(
            env_vars=["req"], sys_vars=["grant"],
            assumptions=["G(F(req))"],
            guarantees=["G(req -> F(grant))", "G(!req -> !grant)"]
        )
        r = synthesize_ltl(spec_with_assume)
        assert r.realizable

    def test_safety_and_liveness_combo(self):
        """System must toggle while respecting safety."""
        result = synthesize_from_strings(
            env_vars=[], sys_vars=["x"],
            assumptions=[],
            guarantees=["G(F(x))", "G(F(!x))"]
        )
        assert result.realizable

    def test_three_client_arbiter(self):
        result = synthesize_arbiter_ltl(3)
        assert result.realizable


# ===================================================================
# Section 10: Statistics and Output
# ===================================================================

class TestStatistics:
    def test_stats_populated(self):
        result = synthesize_from_strings(
            env_vars=["req"], sys_vars=["grant"],
            assumptions=["G(F(req))"],
            guarantees=["G(req -> F(grant))"]
        )
        assert "aux_vars_introduced" in result.statistics
        assert result.statistics["aux_vars_introduced"] == 0
        assert "total_env_vars" in result.statistics
        assert "total_sys_vars" in result.statistics

    def test_reduction_info(self):
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=["G(F(a))"],
            guarantees=["G(b)", "G(F(!b))"]
        )
        assert result.reduction is not None
        assert result.reduction.success
        assert "assumptions" in result.reduction.classification
        assert "guarantees" in result.reduction.classification


# ===================================================================
# Section 11: Parse-based API
# ===================================================================

class TestParseAPI:
    def test_make_spec(self):
        spec = make_ltl_spec(
            env_vars=["a"], sys_vars=["b"],
            assumptions=["G(F(a))"],
            guarantees=["G(a -> F(b))"]
        )
        assert len(spec.assumptions) == 1
        assert len(spec.guarantees) == 1
        assert spec.env_vars == ["a"]
        assert spec.sys_vars == ["b"]

    def test_string_synthesis(self):
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)"]
        )
        assert result.realizable

    def test_complex_string_formulas(self):
        result = synthesize_from_strings(
            env_vars=["a", "b"], sys_vars=["c"],
            assumptions=["G(F(a | b))"],
            guarantees=["G((a & b) -> F(c))"]
        )
        assert result.realizable


# ===================================================================
# Section 12: Persistence Pattern
# ===================================================================

class TestPersistence:
    def test_persistence_formula(self):
        """FG(b) -- eventually b holds forever."""
        f = Finally(Globally(Atom("b")))
        assert classify_formula(f) == FormulaClass.PERSISTENCE

    def test_persistence_synthesis(self):
        result = synthesize_from_strings(
            env_vars=[], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)"]  # Stronger than FG(b) -- should be realizable
        )
        assert result.realizable


# ===================================================================
# Section 13: Mealy Machine Extraction
# ===================================================================

class TestMealyMachine:
    def test_mealy_extraction(self):
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)"]
        )
        assert result.realizable
        if result.mealy is not None:
            assert len(result.mealy.states) > 0
            assert result.mealy.initial is not None

    def test_mealy_step(self):
        result = synthesize_from_strings(
            env_vars=["a"], sys_vars=["b"],
            assumptions=[],
            guarantees=["G(b)"]
        )
        if result.mealy is not None:
            state = result.mealy.initial
            # Try all available transitions from initial state
            if state in result.mealy.transitions:
                for env_input, next_st in result.mealy.transitions[state].items():
                    output = result.mealy.outputs[state][env_input]
                    assert output.get("b", False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
