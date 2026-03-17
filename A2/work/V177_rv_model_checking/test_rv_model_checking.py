"""Tests for V177: Runtime Verification + LTL Model Checking Composition."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V176_runtime_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V023_ltl_model_checking'))

import runtime_monitor as rv
import ltl_model_checker as mc
from rv_model_checking import (
    # Formula bridge
    rv_to_mc, mc_to_rv, formulas_equivalent, FormulaTranslationError,
    # Trace-to-model
    extract_model_from_traces, events_to_propositions, ExtractedModel, TraceState,
    # Dual verifier
    DualVerifier, DualResult, VerificationMode,
    # Counterexample-guided monitor
    counterexample_to_monitor, TargetedMonitor,
    # Specification mining
    mine_specifications, mine_and_verify, mine_response_patterns,
    mine_absence_patterns, mine_precedence_patterns, mine_existence_patterns,
    MinedProperty, PropertyPattern,
    # Trace conformance
    check_trace_conformance, ConformanceResult,
    # Integrated pipeline
    RVModelChecker,
    # Convenience
    verify_with_traces, mine_and_check,
)


# ============================================================
# Helper
# ============================================================

def E(name, **data):
    return rv.Event(name, data)

def make_trace(*names):
    return [rv.Event(n) for n in names]


# ============================================================
# Section 1: Formula Bridge Tests
# ============================================================

class TestFormulaTranslation:

    def test_atom(self):
        f = rv.Atom("p")
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.ATOM
        assert mc_f.name == "p"

    def test_true_false(self):
        assert rv_to_mc(rv.TrueF()).op == mc.LTLOp.TRUE
        assert rv_to_mc(rv.FalseF()).op == mc.LTLOp.FALSE

    def test_not(self):
        f = rv.Not(rv.Atom("p"))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.NOT
        assert mc_f.left.name == "p"

    def test_and(self):
        f = rv.And(rv.Atom("p"), rv.Atom("q"))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.AND
        assert mc_f.left.name == "p"
        assert mc_f.right.name == "q"

    def test_or(self):
        f = rv.Or(rv.Atom("p"), rv.Atom("q"))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.OR

    def test_implies(self):
        f = rv.Implies(rv.Atom("p"), rv.Atom("q"))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.IMPLIES

    def test_next(self):
        f = rv.Next(rv.Atom("p"))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.X

    def test_eventually(self):
        f = rv.Eventually(rv.Atom("p"))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.F

    def test_always(self):
        f = rv.Always(rv.Atom("p"))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.G

    def test_until(self):
        f = rv.Until(rv.Atom("p"), rv.Atom("q"))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.U

    def test_release(self):
        f = rv.Release(rv.Atom("p"), rv.Atom("q"))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.R

    def test_nested_formula(self):
        """G(p -> F(q)) -- response pattern."""
        f = rv.Always(rv.Implies(rv.Atom("p"), rv.Eventually(rv.Atom("q"))))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.G
        inner = mc_f.left
        assert inner.op == mc.LTLOp.IMPLIES
        assert inner.right.op == mc.LTLOp.F

    def test_deeply_nested(self):
        """G(p -> X(q && F(r)))"""
        f = rv.Always(rv.Implies(
            rv.Atom("p"),
            rv.Next(rv.And(rv.Atom("q"), rv.Eventually(rv.Atom("r"))))
        ))
        mc_f = rv_to_mc(f)
        assert mc_f.op == mc.LTLOp.G

    def test_past_time_raises(self):
        with pytest.raises(FormulaTranslationError, match="Past-time"):
            rv_to_mc(rv.Once(rv.Atom("p")))

    def test_historically_raises(self):
        with pytest.raises(FormulaTranslationError, match="Past-time"):
            rv_to_mc(rv.Historically(rv.Atom("p")))

    def test_since_raises(self):
        with pytest.raises(FormulaTranslationError, match="Past-time"):
            rv_to_mc(rv.Since(rv.Atom("p"), rv.Atom("q")))

    def test_previous_raises(self):
        with pytest.raises(FormulaTranslationError, match="Past-time"):
            rv_to_mc(rv.Previous(rv.Atom("p")))

    def test_bounded_eventually(self):
        f = rv.BoundedEventually(rv.Atom("p"), 3)
        mc_f = rv_to_mc(f)
        assert mc_f.op in (mc.LTLOp.OR, mc.LTLOp.ATOM)

    def test_bounded_always(self):
        f = rv.BoundedAlways(rv.Atom("p"), 2)
        mc_f = rv_to_mc(f)
        assert mc_f.op in (mc.LTLOp.AND, mc.LTLOp.ATOM)


class TestMCToRV:

    def test_atom(self):
        f = mc.Atom("p")
        rv_f = mc_to_rv(f)
        assert isinstance(rv_f, rv.Atom)
        assert rv_f.predicate == "p"

    def test_true_false(self):
        assert isinstance(mc_to_rv(mc.LTLTrue()), rv.TrueF)
        assert isinstance(mc_to_rv(mc.LTLFalse()), rv.FalseF)

    def test_not(self):
        rv_f = mc_to_rv(mc.Not(mc.Atom("p")))
        assert isinstance(rv_f, rv.Not)
        assert isinstance(rv_f.sub, rv.Atom)

    def test_boolean_ops(self):
        assert isinstance(mc_to_rv(mc.And(mc.Atom("p"), mc.Atom("q"))), rv.And)
        assert isinstance(mc_to_rv(mc.Or(mc.Atom("p"), mc.Atom("q"))), rv.Or)
        assert isinstance(mc_to_rv(mc.Implies(mc.Atom("p"), mc.Atom("q"))), rv.Implies)

    def test_temporal_ops(self):
        assert isinstance(mc_to_rv(mc.Next(mc.Atom("p"))), rv.Next)
        assert isinstance(mc_to_rv(mc.Finally(mc.Atom("p"))), rv.Eventually)
        assert isinstance(mc_to_rv(mc.Globally(mc.Atom("p"))), rv.Always)

    def test_until_release(self):
        assert isinstance(mc_to_rv(mc.Until(mc.Atom("p"), mc.Atom("q"))), rv.Until)
        assert isinstance(mc_to_rv(mc.Release(mc.Atom("p"), mc.Atom("q"))), rv.Release)

    def test_iff_expansion(self):
        rv_f = mc_to_rv(mc.Iff(mc.Atom("p"), mc.Atom("q")))
        assert isinstance(rv_f, rv.And)
        assert isinstance(rv_f.left, rv.Implies)
        assert isinstance(rv_f.right, rv.Implies)

    def test_weak_until_expansion(self):
        rv_f = mc_to_rv(mc.WeakUntil(mc.Atom("p"), mc.Atom("q")))
        assert isinstance(rv_f, rv.Or)
        assert isinstance(rv_f.left, rv.Until)
        assert isinstance(rv_f.right, rv.Always)


class TestRoundTrip:

    def test_atom_roundtrip(self):
        f = rv.Atom("p")
        assert formulas_equivalent(f, mc_to_rv(rv_to_mc(f)))

    def test_not_roundtrip(self):
        f = rv.Not(rv.Atom("p"))
        assert formulas_equivalent(f, mc_to_rv(rv_to_mc(f)))

    def test_and_roundtrip(self):
        f = rv.And(rv.Atom("p"), rv.Atom("q"))
        assert formulas_equivalent(f, mc_to_rv(rv_to_mc(f)))

    def test_temporal_roundtrip(self):
        f = rv.Always(rv.Implies(rv.Atom("p"), rv.Eventually(rv.Atom("q"))))
        assert formulas_equivalent(f, mc_to_rv(rv_to_mc(f)))

    def test_until_roundtrip(self):
        f = rv.Until(rv.Atom("p"), rv.Atom("q"))
        assert formulas_equivalent(f, mc_to_rv(rv_to_mc(f)))

    def test_complex_roundtrip(self):
        f = rv.Or(
            rv.And(rv.Atom("a"), rv.Next(rv.Atom("b"))),
            rv.Until(rv.Atom("c"), rv.Atom("d"))
        )
        assert formulas_equivalent(f, mc_to_rv(rv_to_mc(f)))


class TestFormulasEquivalent:

    def test_same_atom(self):
        assert formulas_equivalent(rv.Atom("p"), rv.Atom("p"))

    def test_different_atoms(self):
        assert not formulas_equivalent(rv.Atom("p"), rv.Atom("q"))

    def test_different_types(self):
        assert not formulas_equivalent(rv.Atom("p"), rv.TrueF())

    def test_true_true(self):
        assert formulas_equivalent(rv.TrueF(), rv.TrueF())

    def test_nested(self):
        f1 = rv.And(rv.Atom("a"), rv.Not(rv.Atom("b")))
        f2 = rv.And(rv.Atom("a"), rv.Not(rv.Atom("b")))
        assert formulas_equivalent(f1, f2)

    def test_nested_different(self):
        f1 = rv.And(rv.Atom("a"), rv.Not(rv.Atom("b")))
        f2 = rv.And(rv.Atom("a"), rv.Not(rv.Atom("c")))
        assert not formulas_equivalent(f1, f2)


# ============================================================
# Section 2: Trace-to-Model Extraction Tests
# ============================================================

class TestEventsToPropositions:

    def test_simple_event(self):
        props = events_to_propositions(rv.Event("click"))
        assert "click" in props

    def test_string_event(self):
        props = events_to_propositions("click")
        assert "click" in props

    def test_custom_extractors(self):
        extractors = {
            "high_value": lambda e: e.data.get("value", 0) > 100
        }
        props = events_to_propositions(
            rv.Event("purchase", {"value": 200}), extractors
        )
        assert "purchase" in props
        assert "high_value" in props

    def test_custom_extractor_false(self):
        extractors = {
            "high_value": lambda e: e.data.get("value", 0) > 100
        }
        props = events_to_propositions(
            rv.Event("purchase", {"value": 50}), extractors
        )
        assert "purchase" in props
        assert "high_value" not in props


class TestExtractModel:

    def test_single_trace(self):
        trace = make_trace("a", "b", "c")
        model = extract_model_from_traces([trace])
        assert set(model.state_vars) == {"a", "b", "c"}
        assert len(model.initial_states) == 1
        assert model.initial_states[0]["a"] is True
        assert len(model.transitions) == 2

    def test_multiple_traces(self):
        t1 = make_trace("a", "b")
        t2 = make_trace("a", "c")
        model = extract_model_from_traces([t1, t2])
        assert len(model.initial_states) == 1
        assert len(model.transitions) == 2

    def test_duplicate_transitions(self):
        t1 = make_trace("a", "b")
        t2 = make_trace("a", "b")
        model = extract_model_from_traces([t1, t2])
        assert len(model.transitions) == 1

    def test_empty_trace(self):
        model = extract_model_from_traces([[]])
        assert model.state_vars == []
        assert model.initial_states == []

    def test_model_stats(self):
        t1 = make_trace("a", "b", "c")
        t2 = make_trace("a", "b", "d")
        model = extract_model_from_traces([t1, t2])
        assert model.num_traces == 2
        assert model.num_events == 6

    def test_single_event_trace(self):
        trace = make_trace("a")
        model = extract_model_from_traces([trace])
        assert len(model.initial_states) == 1
        assert len(model.transitions) == 0


# ============================================================
# Section 3: Dual Verifier Tests
# ============================================================

class TestDualVerifierModelCheck:

    def test_simple_safety(self):
        """G(p || !p) -- tautology on alternating system."""
        verifier = DualVerifier(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": False}},
                {"condition": {"p": False}, "update": {"p": True}},
            ]
        )
        prop = rv.Always(rv.Or(rv.Atom("p"), rv.Not(rv.Atom("p"))))
        result = verifier.model_check(prop)
        assert result.holds

    def test_eventually_property(self):
        """F(!p) on system p=T -> p=F -> stay."""
        verifier = DualVerifier(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": False}},
                {"condition": {"p": False}, "update": {"p": False}},
            ]
        )
        prop = rv.Eventually(rv.Not(rv.Atom("p")))
        result = verifier.model_check(prop)
        assert result.holds

    def test_violated_property(self):
        """F(!p) on system where p always stays true."""
        verifier = DualVerifier(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": True}},
            ]
        )
        prop = rv.Eventually(rv.Not(rv.Atom("p")))
        result = verifier.model_check(prop)
        assert not result.holds

    def test_past_time_raises_in_mc(self):
        verifier = DualVerifier(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[{"condition": {"p": True}, "update": {"p": True}}]
        )
        with pytest.raises(FormulaTranslationError):
            verifier.model_check(rv.Once(rv.Atom("p")))


class TestDualVerifierMonitor:

    def test_monitor_satisfied(self):
        verifier = DualVerifier(state_vars=[], init_map=None, transitions=None)
        prop = rv.Eventually(rv.Atom("done"))
        trace = make_trace("start", "work", "done")
        verdicts, final = verifier.monitor_trace(prop, trace)
        assert final == rv.Verdict.TRUE

    def test_monitor_violated(self):
        verifier = DualVerifier(state_vars=[], init_map=None, transitions=None)
        prop = rv.Eventually(rv.Atom("done"))
        trace = make_trace("start", "work", "fail")
        verdicts, final = verifier.monitor_trace(prop, trace)
        assert final == rv.Verdict.FALSE

    def test_monitor_safety(self):
        verifier = DualVerifier(state_vars=[], init_map=None, transitions=None)
        prop = rv.Always(rv.Not(rv.Atom("error")))
        trace = make_trace("ok", "error", "ok")
        verdicts, final = verifier.monitor_trace(prop, trace)
        assert final == rv.Verdict.FALSE


class TestDualVerifierDualMode:

    def test_dual_consistent_satisfied(self):
        verifier = DualVerifier(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": False}},
                {"condition": {"p": False}, "update": {"p": True}},
            ]
        )
        prop = rv.Always(rv.Or(rv.Atom("p"), rv.Not(rv.Atom("p"))))
        trace = [rv.Event("p"), rv.Event("other")]
        result = verifier.verify(prop, trace, VerificationMode.DUAL)
        assert result.mc_result is not None
        assert result.rv_final is not None

    def test_monitor_only_mode(self):
        verifier = DualVerifier(state_vars=["p"], init_map={"p": True},
                                transitions=[])
        prop = rv.Eventually(rv.Atom("done"))
        trace = make_trace("done")
        result = verifier.verify(prop, trace, VerificationMode.MONITOR)
        assert result.mc_result is None
        assert result.rv_final is not None

    def test_mc_only_mode(self):
        verifier = DualVerifier(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": True}},
            ]
        )
        prop = rv.Always(rv.Atom("p"))
        result = verifier.verify(prop, None, VerificationMode.MODEL_CHECK)
        assert result.mc_result is not None
        assert result.rv_final is None

    def test_dual_no_trace(self):
        verifier = DualVerifier(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": True}},
            ]
        )
        prop = rv.Always(rv.Atom("p"))
        result = verifier.verify(prop, None, VerificationMode.DUAL)
        assert result.mc_result is not None
        assert result.rv_final is None


class TestDualVerifierFromModel:

    def test_from_extracted_model(self):
        traces = [make_trace("a", "b", "c"), make_trace("a", "b", "a")]
        model = extract_model_from_traces(traces)
        verifier = DualVerifier.from_extracted_model(model)
        assert verifier._has_model

    def test_verify_extracted_model(self):
        traces = [make_trace("a", "b"), make_trace("a", "c")]
        model = extract_model_from_traces(traces)
        verifier = DualVerifier.from_extracted_model(model)
        prop = rv.Always(rv.Or(rv.Atom("a"), rv.Not(rv.Atom("a"))))
        result = verifier.model_check(prop)
        assert result.holds


# ============================================================
# Section 4: Counterexample-Guided Monitor Tests
# ============================================================

class TestCounterexampleMonitor:

    def test_no_counterexample(self):
        mc_result = mc.LTLResult(holds=True)
        monitor = counterexample_to_monitor(
            rv.Always(rv.Atom("p")), mc_result, ["p"]
        )
        assert monitor is None

    def test_with_counterexample(self):
        mc_result = mc.LTLResult(
            holds=False,
            counterexample=([{"p": True}], [{"p": False}])
        )
        monitor = counterexample_to_monitor(
            rv.Always(rv.Atom("p")), mc_result, ["p"]
        )
        assert monitor is not None
        assert len(monitor.counterexample_prefix) == 1
        assert len(monitor.counterexample_cycle) == 1

    def test_targeted_monitor_check(self):
        mc_result = mc.LTLResult(
            holds=False,
            counterexample=([{"p": True}, {"p": False}], [{"p": False}])
        )
        monitor = counterexample_to_monitor(
            rv.Always(rv.Atom("p")), mc_result, ["p"]
        )
        assert monitor is not None
        trace = make_trace("p", "q")
        results = monitor.check_trace(trace)
        assert 'property_verdict' in results
        assert 'matches_counterexample' in results

    def test_matching_counterexample_pattern(self):
        mc_result = mc.LTLResult(
            holds=False,
            counterexample=([{"a": True, "b": False}], [{"a": False, "b": True}])
        )
        monitor = counterexample_to_monitor(
            rv.Eventually(rv.Atom("c")), mc_result, ["a", "b"]
        )
        assert monitor is not None
        assert len(monitor.prefix_patterns) > 0


# ============================================================
# Section 5: Specification Mining Tests
# ============================================================

class TestMineResponsePatterns:

    def test_clear_response(self):
        traces = [
            make_trace("a", "b", "a", "b"),
            make_trace("a", "b"),
        ]
        results = mine_response_patterns(traces)
        descs = [r.description for r in results]
        assert any("G(a -> F(b))" in d for d in descs)

    def test_no_response(self):
        traces = [make_trace("a", "c", "a", "c")]
        results = mine_response_patterns(traces)
        descs = [r.description for r in results]
        assert not any("G(a -> F(b))" in d for d in descs)

    def test_partial_response(self):
        traces = [make_trace("a", "b", "a", "c")]
        results = mine_response_patterns(traces)
        descs = [r.description for r in results]
        assert not any("G(a -> F(b))" in d for d in descs)

    def test_multiple_responses(self):
        traces = [
            make_trace("req", "ack", "done"),
            make_trace("req", "ack", "done"),
        ]
        results = mine_response_patterns(traces)
        assert len(results) >= 1


class TestMineAbsencePatterns:

    def test_absent_event(self):
        traces = [make_trace("a", "b"), make_trace("a", "c")]
        results = mine_absence_patterns(traces)
        descs = [r.description for r in results]
        assert any("G(!b)" in d for d in descs) or any("G(!c)" in d for d in descs)

    def test_no_absence(self):
        traces = [make_trace("a", "b"), make_trace("a", "b")]
        results = mine_absence_patterns(traces)
        assert len(results) == 0


class TestMinePrecedencePatterns:

    def test_clear_precedence(self):
        traces = [make_trace("a", "b"), make_trace("a", "c", "b")]
        results = mine_precedence_patterns(traces)
        descs = [r.description for r in results]
        assert any("a precedes b" in d for d in descs)

    def test_no_precedence(self):
        traces = [make_trace("b", "a"), make_trace("a", "b")]
        results = mine_precedence_patterns(traces)
        descs = [r.description for r in results]
        assert not any("a precedes b" in d for d in descs)


class TestMineExistencePatterns:

    def test_universal_event(self):
        traces = [make_trace("a", "b"), make_trace("a", "c"), make_trace("a", "d")]
        results = mine_existence_patterns(traces)
        descs = [r.description for r in results]
        assert any("F(a)" in d for d in descs)

    def test_rare_event(self):
        traces = [make_trace("a")] * 9 + [make_trace("a", "b")]
        results = mine_existence_patterns(traces)
        descs = [r.description for r in results]
        assert not any("F(b)" == d for d in descs)


class TestMineSpecifications:

    def test_combined_mining(self):
        traces = [
            make_trace("init", "request", "response", "done"),
            make_trace("init", "request", "response", "done"),
        ]
        results = mine_specifications(traces)
        assert len(results) > 0

    def test_filter_by_pattern(self):
        traces = [make_trace("a", "b"), make_trace("a", "b")]
        results = mine_specifications(traces, [PropertyPattern.RESPONSE])
        for r in results:
            assert r.pattern == PropertyPattern.RESPONSE

    def test_min_support(self):
        traces = [make_trace("a", "b")]
        results_1 = mine_specifications(traces, min_support=1)
        results_5 = mine_specifications(traces, min_support=5)
        assert len(results_1) >= len(results_5)


# ============================================================
# Section 6: Trace Conformance Tests
# ============================================================

class TestTraceConformance:

    def test_conforming_trace(self):
        model = ExtractedModel(
            state_vars=["a", "b"],
            initial_states=[{"a": True, "b": False}],
            transitions=[
                ({"a": True, "b": False}, {"a": False, "b": True}),
                ({"a": False, "b": True}, {"a": True, "b": False}),
            ]
        )
        trace = make_trace("a", "b", "a")
        result = check_trace_conformance(trace, model)
        assert result.conforms

    def test_empty_trace(self):
        model = ExtractedModel(
            state_vars=["a"],
            initial_states=[{"a": True}],
            transitions=[]
        )
        result = check_trace_conformance([], model)
        assert result.conforms


class TestConformanceWithExtractedModel:

    def test_conforming_after_extraction(self):
        trace = make_trace("login", "browse", "checkout", "logout")
        model = extract_model_from_traces([trace])
        result = check_trace_conformance(trace, model)
        assert result.conforms

    def test_new_trace_conformance(self):
        training = [make_trace("a", "b", "c"), make_trace("a", "c", "b")]
        model = extract_model_from_traces(training)
        test_trace = make_trace("a", "b")
        result = check_trace_conformance(test_trace, model)
        assert result.conforms


# ============================================================
# Section 7: Integrated Pipeline Tests
# ============================================================

class TestRVModelChecker:

    def test_add_trace(self):
        pipeline = RVModelChecker()
        pipeline.add_trace(make_trace("a", "b", "c"))
        assert pipeline.model is not None
        assert pipeline.model.num_traces == 1

    def test_add_multiple_traces(self):
        pipeline = RVModelChecker()
        pipeline.add_trace(make_trace("a", "b"))
        pipeline.add_trace(make_trace("a", "c"))
        assert pipeline.model.num_traces == 2

    def test_mine(self):
        pipeline = RVModelChecker()
        pipeline.add_trace(make_trace("req", "ack", "done"))
        pipeline.add_trace(make_trace("req", "ack", "done"))
        mined = pipeline.mine()
        assert isinstance(mined, list)

    def test_verify_property(self):
        pipeline = RVModelChecker()
        pipeline.add_trace(make_trace("a", "b"))
        pipeline.add_trace(make_trace("a", "c"))
        prop = rv.Always(rv.Or(rv.Atom("a"), rv.Not(rv.Atom("a"))))
        result = pipeline.verify_property(prop, make_trace("a", "b"))
        assert isinstance(result, DualResult)

    def test_set_explicit_model(self):
        pipeline = RVModelChecker()
        pipeline.set_model(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": False}},
                {"condition": {"p": False}, "update": {"p": True}},
            ]
        )
        prop = rv.Always(rv.Or(rv.Atom("p"), rv.Not(rv.Atom("p"))))
        result = pipeline.verify_property(prop)
        assert result.mc_satisfied

    def test_check_conformance(self):
        pipeline = RVModelChecker()
        pipeline.add_trace(make_trace("a", "b", "c"))
        result = pipeline.check_conformance(make_trace("a", "b"))
        assert isinstance(result, ConformanceResult)

    def test_full_pipeline(self):
        pipeline = RVModelChecker()
        pipeline.add_trace(make_trace("req", "ack", "done"))
        pipeline.add_trace(make_trace("req", "ack", "done"))
        summary = pipeline.full_pipeline()
        assert 'total_mined' in summary
        assert 'mc_confirmed' in summary

    def test_monitor_only_no_model(self):
        pipeline = RVModelChecker()
        prop = rv.Eventually(rv.Atom("done"))
        trace = make_trace("start", "done")
        result = pipeline.verify_property(prop, trace, VerificationMode.MONITOR)
        assert result.rv_final == rv.Verdict.TRUE

    def test_pipeline_mine_and_verify(self):
        pipeline = RVModelChecker()
        for _ in range(3):
            pipeline.add_trace(make_trace("init", "work", "complete"))
        mined = pipeline.mine()
        verified = pipeline.verify_mined()
        assert isinstance(verified, list)


class TestRVModelCheckerTargetedMonitor:

    def test_create_targeted_from_violation(self):
        pipeline = RVModelChecker()
        pipeline.set_model(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": True}},
            ]
        )
        prop = rv.Eventually(rv.Not(rv.Atom("p")))
        monitor = pipeline.create_targeted_monitor(prop)
        assert monitor is not None

    def test_no_targeted_for_satisfied(self):
        pipeline = RVModelChecker()
        pipeline.set_model(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": True}},
            ]
        )
        prop = rv.Always(rv.Atom("p"))
        monitor = pipeline.create_targeted_monitor(prop)
        assert monitor is None


# ============================================================
# Section 8: Convenience Function Tests
# ============================================================

class TestVerifyWithTraces:

    def test_simple(self):
        traces = [make_trace("a", "b"), make_trace("a", "c")]
        prop = rv.Always(rv.Or(rv.Atom("a"), rv.Not(rv.Atom("a"))))
        result = verify_with_traces(prop, traces)
        assert isinstance(result, DualResult)

    def test_monitor_only_fallback(self):
        traces = [make_trace("done")]
        prop = rv.Eventually(rv.Atom("done"))
        result = verify_with_traces(prop, traces)
        assert result.rv_final is not None


class TestMineAndCheck:

    def test_simple(self):
        traces = [
            make_trace("req", "ack", "done"),
            make_trace("req", "ack", "done"),
        ]
        result = mine_and_check(traces)
        assert 'total_mined' in result
        assert isinstance(result['total_mined'], int)

    def test_empty_traces(self):
        result = mine_and_check([[]])
        assert result['total_mined'] == 0


# ============================================================
# Section 9: End-to-End Integration Tests
# ============================================================

class TestEndToEnd:

    def test_request_response_system(self):
        pipeline = RVModelChecker()
        pipeline.set_model(
            state_vars=["req", "proc", "resp"],
            init_map={"req": True, "proc": False, "resp": False},
            transitions=[
                {"condition": {"req": True, "proc": False, "resp": False},
                 "update": {"req": False, "proc": True, "resp": False}},
                {"condition": {"req": False, "proc": True, "resp": False},
                 "update": {"req": False, "proc": False, "resp": True}},
                {"condition": {"req": False, "proc": False, "resp": True},
                 "update": {"req": True, "proc": False, "resp": False}},
            ]
        )
        prop = rv.Always(rv.Implies(
            rv.Atom("req"), rv.Eventually(rv.Atom("resp"))
        ))
        result = pipeline.verify_property(prop)
        assert result.mc_satisfied

    def test_safety_violation_detection(self):
        pipeline = RVModelChecker()
        pipeline.set_model(
            state_vars=["ok", "err"],
            init_map={"ok": True, "err": False},
            transitions=[
                {"condition": {"ok": True, "err": False},
                 "update": {"ok": True, "err": False}},
                {"condition": {"ok": True, "err": False},
                 "update": {"ok": False, "err": True}},
                {"condition": {"ok": False, "err": True},
                 "update": {"ok": False, "err": True}},
            ]
        )
        prop = rv.Always(rv.Not(rv.Atom("err")))
        result = pipeline.verify_property(prop)
        assert not result.mc_satisfied

    def test_trace_mining_to_verification(self):
        traces = [
            make_trace("req", "proc", "resp"),
            make_trace("req", "proc", "resp"),
            make_trace("req", "proc", "resp"),
        ]
        pipeline = RVModelChecker()
        for t in traces:
            pipeline.add_trace(t)
        mined = pipeline.mine(min_confidence=0.9)
        assert len(mined) > 0
        summary = pipeline.full_pipeline()
        assert summary['total_mined'] > 0

    def test_alternating_system(self):
        pipeline = RVModelChecker()
        pipeline.set_model(
            state_vars=["a", "b"],
            init_map={"a": True, "b": False},
            transitions=[
                {"condition": {"a": True, "b": False},
                 "update": {"a": False, "b": True}},
                {"condition": {"a": False, "b": True},
                 "update": {"a": True, "b": False}},
            ]
        )
        prop = rv.Always(rv.Eventually(rv.Atom("a")))
        result = pipeline.verify_property(prop)
        assert result.mc_satisfied

        prop = rv.Always(rv.Eventually(rv.Atom("b")))
        result = pipeline.verify_property(prop)
        assert result.mc_satisfied

    def test_dual_verification_consistency(self):
        pipeline = RVModelChecker()
        pipeline.set_model(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": False}},
                {"condition": {"p": False}, "update": {"p": True}},
            ]
        )
        prop = rv.Always(rv.Or(rv.Atom("p"), rv.Not(rv.Atom("p"))))
        trace = [rv.Event("p"), rv.Event("other"), rv.Event("p")]
        result = pipeline.verify_property(prop, trace, VerificationMode.DUAL)
        assert result.mc_result is not None
        assert result.rv_final is not None

    def test_counterexample_to_monitor_flow(self):
        pipeline = RVModelChecker()
        pipeline.set_model(
            state_vars=["p"],
            init_map={"p": True},
            transitions=[
                {"condition": {"p": True}, "update": {"p": True}},
            ]
        )
        prop = rv.Eventually(rv.Not(rv.Atom("p")))
        monitor = pipeline.create_targeted_monitor(prop)
        assert monitor is not None
        trace = make_trace("p", "p", "p")
        results = monitor.check_trace(trace)
        assert results['property_verdict'] == rv.Verdict.FALSE

    def test_formula_bridge_in_pipeline(self):
        mc_prop = mc.Globally(mc.Implies(mc.Atom("req"), mc.Finally(mc.Atom("resp"))))
        rv_prop = mc_to_rv(mc_prop)
        assert isinstance(rv_prop, rv.Always)

        mon = rv.FutureTimeMonitor(rv_prop)
        for evt in make_trace("req", "work", "resp"):
            mon.process(evt)
        assert mon.finalize() == rv.Verdict.TRUE

        mc_prop2 = rv_to_mc(rv_prop)
        assert mc_prop2.op == mc.LTLOp.G

    def test_multi_property_pipeline(self):
        pipeline = RVModelChecker()
        pipeline.set_model(
            state_vars=["idle", "busy", "done"],
            init_map={"idle": True, "busy": False, "done": False},
            transitions=[
                {"condition": {"idle": True, "busy": False, "done": False},
                 "update": {"idle": False, "busy": True, "done": False}},
                {"condition": {"idle": False, "busy": True, "done": False},
                 "update": {"idle": False, "busy": False, "done": True}},
                {"condition": {"idle": False, "busy": False, "done": True},
                 "update": {"idle": True, "busy": False, "done": False}},
            ]
        )

        p1 = rv.Always(rv.Implies(rv.Atom("idle"), rv.Eventually(rv.Atom("done"))))
        assert pipeline.verify_property(p1).mc_satisfied

        p2 = rv.Always(rv.Implies(rv.Atom("busy"), rv.Eventually(rv.Atom("done"))))
        assert pipeline.verify_property(p2).mc_satisfied

        p3 = rv.Always(rv.Eventually(rv.Atom("idle")))
        assert pipeline.verify_property(p3).mc_satisfied

    def test_conformance_integration(self):
        pipeline = RVModelChecker()
        pipeline.add_trace(make_trace("start", "process", "end"))
        pipeline.add_trace(make_trace("start", "process", "end"))
        r1 = pipeline.check_conformance(make_trace("start", "process", "end"))
        assert r1.conforms

    def test_mine_verify_monitor_flow(self):
        pipeline = RVModelChecker()
        for _ in range(3):
            pipeline.add_trace(make_trace("open", "read", "close"))

        mined = pipeline.mine(min_confidence=0.9)
        verified = pipeline.verify_mined()

        new_trace = make_trace("open", "read", "close")
        for prop in verified:
            result = pipeline.verify_property(
                prop.formula, new_trace, VerificationMode.MONITOR
            )
            assert result.rv_final is not None


# ============================================================
# Section 10: Edge Cases
# ============================================================

class TestEdgeCases:

    def test_single_event_trace(self):
        pipeline = RVModelChecker()
        pipeline.add_trace(make_trace("x"))
        assert pipeline.model.num_events == 1

    def test_empty_pipeline(self):
        pipeline = RVModelChecker()
        assert pipeline.model is None
        assert pipeline.traces == []

    def test_formula_translation_preserves_semantics(self):
        verifier = DualVerifier(
            state_vars=["p", "q"],
            init_map={"p": True, "q": False},
            transitions=[
                {"condition": {"p": True, "q": False},
                 "update": {"p": False, "q": True}},
                {"condition": {"p": False, "q": True},
                 "update": {"p": True, "q": False}},
            ]
        )
        prop = rv.Always(rv.Or(rv.Atom("p"), rv.Atom("q")))
        result = verifier.verify(prop, make_trace("p", "q", "p"), VerificationMode.DUAL)
        assert result.mc_result is not None

    def test_large_trace(self):
        large_trace = make_trace(*["a", "b"] * 50)
        pipeline = RVModelChecker()
        pipeline.add_trace(large_trace)
        prop = rv.Always(rv.Or(rv.Atom("a"), rv.Atom("b")))
        result = pipeline.verify_property(prop, large_trace, VerificationMode.MONITOR)
        assert result.rv_final is not None

    def test_many_propositions(self):
        names = [f"v{i}" for i in range(10)]
        trace = [rv.Event(n) for n in names]
        pipeline = RVModelChecker()
        pipeline.add_trace(trace)
        assert len(pipeline.model.state_vars) == 10

    def test_data_events_with_extractors(self):
        extractors = {
            "high_temp": lambda e: e.data.get("temp", 0) > 100,
            "low_temp": lambda e: e.data.get("temp", 0) < 0,
        }
        trace = [
            rv.Event("sensor", {"temp": 50}),
            rv.Event("sensor", {"temp": 150}),
            rv.Event("sensor", {"temp": -10}),
        ]
        pipeline = RVModelChecker()
        pipeline.add_trace(trace, extractors)
        assert "high_temp" in pipeline.model.state_vars
        assert "low_temp" in pipeline.model.state_vars

    def test_mine_with_no_patterns(self):
        traces = [make_trace("a"), make_trace("b"), make_trace("c")]
        results = mine_specifications(traces, min_support=1)
        assert isinstance(results, list)

    def test_verify_with_single_trace(self):
        trace = make_trace("a", "b", "c")
        prop = rv.Eventually(rv.Atom("c"))
        result = verify_with_traces(prop, [trace])
        assert result.rv_final is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
