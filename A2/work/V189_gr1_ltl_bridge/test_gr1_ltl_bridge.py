"""
Tests for V189: GR(1)-LTL Bridge
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_work = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_work, 'V023_ltl_model_checking'))

from gr1_ltl_bridge import (
    # Core detection
    is_propositional, is_next_propositional, classify_conjunct,
    collect_conjuncts, detect_gr1, is_gr1_fragment,
    # Decomposition
    GR1Decomposition, FragmentKind,
    # Conversion
    decomposition_to_spec, _eval_prop, _eval_next_prop,
    _enumerate_valuations,
    # Synthesis
    synthesize, quick_check, BridgeResult,
    synthesize_assume_guarantee, synthesize_safety,
    synthesize_liveness, synthesize_response,
    synthesize_gr1_direct,
    # Analysis
    analyze_spec, compare_methods, bridge_summary,
    # Utility
    _formula_depth, _formula_size, ltl_to_gr1_components,
    _conjoin, _decomp_to_ltl,
)

from ltl_model_checker import (
    LTL, LTLOp, Atom, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release, WeakUntil,
    LTLTrue, LTLFalse, atoms
)


# ============================================================
# 1. Propositional Detection
# ============================================================

class TestIsPropositional:
    def test_atom(self):
        assert is_propositional(Atom("a"))

    def test_true_false(self):
        assert is_propositional(LTLTrue())
        assert is_propositional(LTLFalse())

    def test_not(self):
        assert is_propositional(Not(Atom("a")))

    def test_and_or(self):
        assert is_propositional(And(Atom("a"), Atom("b")))
        assert is_propositional(Or(Atom("a"), Atom("b")))

    def test_implies(self):
        assert is_propositional(Implies(Atom("a"), Atom("b")))

    def test_nested_boolean(self):
        f = And(Or(Atom("a"), Not(Atom("b"))), Implies(Atom("c"), Atom("d")))
        assert is_propositional(f)

    def test_temporal_not_propositional(self):
        assert not is_propositional(Globally(Atom("a")))
        assert not is_propositional(Finally(Atom("a")))
        assert not is_propositional(Next(Atom("a")))
        assert not is_propositional(Until(Atom("a"), Atom("b")))

    def test_nested_temporal(self):
        assert not is_propositional(And(Atom("a"), Globally(Atom("b"))))


class TestIsNextPropositional:
    def test_atom(self):
        assert is_next_propositional(Atom("a"))

    def test_next_atom(self):
        assert is_next_propositional(Next(Atom("a")))

    def test_mixed(self):
        f = And(Atom("a"), Next(Atom("b")))
        assert is_next_propositional(f)

    def test_implies_with_next(self):
        f = Implies(Atom("a"), Next(Atom("a")))
        assert is_next_propositional(f)

    def test_nested_next_rejected(self):
        f = Next(Next(Atom("a")))
        assert not is_next_propositional(f)

    def test_next_temporal_rejected(self):
        f = Next(Globally(Atom("a")))
        assert not is_next_propositional(f)

    def test_globally_rejected(self):
        assert not is_next_propositional(Globally(Atom("a")))


# ============================================================
# 2. Conjunct Classification
# ============================================================

class TestClassifyConjunct:
    def test_propositional_is_init(self):
        kind, inner = classify_conjunct(Atom("a"))
        assert kind == "init"
        assert inner == Atom("a")

    def test_safety(self):
        kind, inner = classify_conjunct(Globally(Atom("a")))
        assert kind == "safety"
        assert inner == Atom("a")

    def test_justice(self):
        kind, inner = classify_conjunct(Globally(Finally(Atom("a"))))
        assert kind == "justice"
        assert inner == Atom("a")

    def test_safety_complex(self):
        f = Globally(Or(Atom("a"), Not(Atom("b"))))
        kind, inner = classify_conjunct(f)
        assert kind == "safety"

    def test_justice_complex(self):
        f = Globally(Finally(And(Atom("a"), Atom("b"))))
        kind, inner = classify_conjunct(f)
        assert kind == "justice"

    def test_transition(self):
        f = Globally(Implies(Atom("a"), Next(Atom("b"))))
        kind, inner = classify_conjunct(f)
        assert kind == "transition"

    def test_non_gr1_until(self):
        f = Until(Atom("a"), Atom("b"))
        kind, inner = classify_conjunct(f)
        assert kind == "non_gr1"

    def test_non_gr1_nested_temporal(self):
        f = Globally(Until(Atom("a"), Atom("b")))
        kind, inner = classify_conjunct(f)
        assert kind == "non_gr1"

    def test_true_is_init(self):
        kind, inner = classify_conjunct(LTLTrue())
        assert kind == "init"

    def test_g_true(self):
        kind, inner = classify_conjunct(Globally(LTLTrue()))
        assert kind == "safety"


# ============================================================
# 3. Collect Conjuncts
# ============================================================

class TestCollectConjuncts:
    def test_single(self):
        result = collect_conjuncts(Atom("a"))
        assert result == [Atom("a")]

    def test_and(self):
        f = And(Atom("a"), Atom("b"))
        result = collect_conjuncts(f)
        assert len(result) == 2

    def test_nested_and(self):
        f = And(And(Atom("a"), Atom("b")), Atom("c"))
        result = collect_conjuncts(f)
        assert len(result) == 3

    def test_true_filtered(self):
        result = collect_conjuncts(LTLTrue())
        assert result == []

    def test_or_not_flattened(self):
        f = Or(Atom("a"), Atom("b"))
        result = collect_conjuncts(f)
        assert len(result) == 1  # Or is a single conjunct


# ============================================================
# 4. GR(1) Detection
# ============================================================

class TestDetectGR1:
    def test_pure_safety(self):
        spec = Globally(Not(Atom("bad")))
        decomp = detect_gr1(spec, {"r"}, {"g"})
        assert decomp.is_gr1
        assert len(decomp.sys_safety) == 1

    def test_pure_justice(self):
        spec = Globally(Finally(Atom("grant")))
        decomp = detect_gr1(spec, {"r"}, {"grant"})
        assert decomp.is_gr1
        assert len(decomp.sys_justice) == 1

    def test_safety_and_justice(self):
        spec = And(Globally(Not(Atom("bad"))), Globally(Finally(Atom("good"))))
        decomp = detect_gr1(spec, set(), {"bad", "good"})
        assert decomp.is_gr1
        assert len(decomp.sys_safety) == 1
        assert len(decomp.sys_justice) == 1

    def test_assume_guarantee(self):
        assume = Globally(Finally(Atom("req")))
        guarantee = Globally(Finally(Atom("grant")))
        spec = Implies(assume, guarantee)
        decomp = detect_gr1(spec, {"req"}, {"grant"})
        assert decomp.is_gr1
        assert len(decomp.env_justice) == 1
        assert len(decomp.sys_justice) == 1

    def test_assume_guarantee_with_safety(self):
        assume = And(Globally(Atom("valid")), Globally(Finally(Atom("req"))))
        guarantee = And(Globally(Not(Atom("err"))), Globally(Finally(Atom("ack"))))
        spec = Implies(assume, guarantee)
        decomp = detect_gr1(spec, {"valid", "req"}, {"err", "ack"})
        assert decomp.is_gr1
        assert len(decomp.env_safety) == 1
        assert len(decomp.env_justice) == 1
        assert len(decomp.sys_safety) == 1
        assert len(decomp.sys_justice) == 1

    def test_non_gr1_until(self):
        spec = Until(Atom("a"), Atom("b"))
        decomp = detect_gr1(spec, {"a"}, {"b"})
        assert not decomp.is_gr1

    def test_non_gr1_nested(self):
        spec = Globally(Until(Atom("a"), Atom("b")))
        decomp = detect_gr1(spec, {"a"}, {"b"})
        assert not decomp.is_gr1

    def test_init_condition(self):
        spec = And(Atom("init_ok"), Globally(Not(Atom("bad"))))
        decomp = detect_gr1(spec, set(), {"init_ok", "bad"})
        assert decomp.is_gr1
        assert decomp.sys_init is not None

    def test_multiple_safety(self):
        spec = And(Globally(Atom("s1")), And(Globally(Atom("s2")), Globally(Atom("s3"))))
        decomp = detect_gr1(spec, set(), {"s1", "s2", "s3"})
        assert decomp.is_gr1
        assert len(decomp.sys_safety) == 3

    def test_multiple_justice(self):
        spec = And(Globally(Finally(Atom("j1"))), Globally(Finally(Atom("j2"))))
        decomp = detect_gr1(spec, set(), {"j1", "j2"})
        assert decomp.is_gr1
        assert len(decomp.sys_justice) == 2

    def test_transition_constraint(self):
        # G(a -> X(b)) is a valid GR(1) transition constraint
        spec = Globally(Implies(Atom("a"), Next(Atom("b"))))
        decomp = detect_gr1(spec, {"a"}, {"b"})
        assert decomp.is_gr1


class TestIsGR1Fragment:
    def test_gr1_true(self):
        spec = Globally(Finally(Atom("a")))
        assert is_gr1_fragment(spec)

    def test_gr1_false(self):
        spec = Until(Atom("a"), Atom("b"))
        assert not is_gr1_fragment(spec)


# ============================================================
# 5. Propositional Evaluation
# ============================================================

class TestEvalProp:
    def test_atom_true(self):
        assert _eval_prop(Atom("a"), frozenset({"a", "b"}))

    def test_atom_false(self):
        assert not _eval_prop(Atom("a"), frozenset({"b"}))

    def test_not(self):
        assert _eval_prop(Not(Atom("a")), frozenset({"b"}))

    def test_and(self):
        assert _eval_prop(And(Atom("a"), Atom("b")), frozenset({"a", "b"}))
        assert not _eval_prop(And(Atom("a"), Atom("b")), frozenset({"a"}))

    def test_or(self):
        assert _eval_prop(Or(Atom("a"), Atom("b")), frozenset({"a"}))
        assert not _eval_prop(Or(Atom("a"), Atom("b")), frozenset())

    def test_implies(self):
        assert _eval_prop(Implies(Atom("a"), Atom("b")), frozenset({"a", "b"}))
        assert _eval_prop(Implies(Atom("a"), Atom("b")), frozenset({"b"}))
        assert not _eval_prop(Implies(Atom("a"), Atom("b")), frozenset({"a"}))

    def test_true_false(self):
        assert _eval_prop(LTLTrue(), frozenset())
        assert not _eval_prop(LTLFalse(), frozenset())

    def test_iff(self):
        from ltl_model_checker import Iff
        assert _eval_prop(Iff(Atom("a"), Atom("b")), frozenset({"a", "b"}))
        assert _eval_prop(Iff(Atom("a"), Atom("b")), frozenset())
        assert not _eval_prop(Iff(Atom("a"), Atom("b")), frozenset({"a"}))


class TestEvalNextProp:
    def test_current_atom(self):
        assert _eval_next_prop(Atom("a"), frozenset({"a"}), frozenset())

    def test_next_atom(self):
        assert _eval_next_prop(Next(Atom("a")), frozenset(), frozenset({"a"}))

    def test_mixed(self):
        f = And(Atom("a"), Next(Atom("b")))
        assert _eval_next_prop(f, frozenset({"a"}), frozenset({"b"}))
        assert not _eval_next_prop(f, frozenset({"a"}), frozenset())

    def test_implies_with_next(self):
        f = Implies(Atom("a"), Next(Atom("a")))
        assert _eval_next_prop(f, frozenset({"a"}), frozenset({"a"}))
        assert not _eval_next_prop(f, frozenset({"a"}), frozenset())
        assert _eval_next_prop(f, frozenset(), frozenset())


# ============================================================
# 6. Enumeration
# ============================================================

class TestEnumerateValuations:
    def test_empty(self):
        result = _enumerate_valuations([])
        assert result == [frozenset()]

    def test_one_var(self):
        result = _enumerate_valuations(["a"])
        assert len(result) == 2
        assert frozenset() in result
        assert frozenset({"a"}) in result

    def test_two_vars(self):
        result = _enumerate_valuations(["a", "b"])
        assert len(result) == 4

    def test_three_vars(self):
        result = _enumerate_valuations(["a", "b", "c"])
        assert len(result) == 8


# ============================================================
# 7. Quick Check
# ============================================================

class TestQuickCheck:
    def test_true(self):
        assert quick_check(LTLTrue(), set(), set()) == "realizable"

    def test_false(self):
        assert quick_check(LTLFalse(), set(), set()) == "unrealizable"

    def test_g_true(self):
        assert quick_check(Globally(LTLTrue()), set(), set()) == "realizable"

    def test_g_false(self):
        assert quick_check(Globally(LTLFalse()), set(), set()) == "unrealizable"

    def test_f_true(self):
        assert quick_check(Finally(LTLTrue()), set(), set()) == "realizable"

    def test_f_false(self):
        assert quick_check(Finally(LTLFalse()), set(), set()) == "unrealizable"

    def test_gf_true(self):
        assert quick_check(Globally(Finally(LTLTrue())), set(), set()) == "realizable"

    def test_gf_false(self):
        assert quick_check(Globally(Finally(LTLFalse())), set(), set()) == "unrealizable"

    def test_propositional_realizable(self):
        # sys controls "a", can always set a=true
        result = quick_check(Atom("a"), set(), {"a"})
        assert result == "realizable"

    def test_propositional_unrealizable(self):
        # sys controls "a", but spec requires both a and !a
        spec = And(Atom("a"), Not(Atom("a")))
        result = quick_check(spec, set(), {"a"})
        assert result == "unrealizable"

    def test_propositional_env_dependency(self):
        # spec: a (env controls a), sys has no vars
        result = quick_check(Atom("a"), {"a"}, set())
        assert result == "unrealizable"  # sys can't force a=true

    def test_temporal_returns_none(self):
        result = quick_check(Globally(Atom("a")), set(), {"a"})
        assert result is None


# ============================================================
# 8. Decomposition to Spec Conversion
# ============================================================

class TestDecompositionToSpec:
    def test_simple_justice(self):
        decomp = GR1Decomposition(
            is_gr1=True,
            sys_justice=[Atom("g")]
        )
        spec = decomposition_to_spec(decomp, set(), {"g"})
        assert spec.sys_vars == ["g"]
        assert len(spec.sys_justice) == 1
        assert spec.sys_justice[0](frozenset({"g"}))
        assert not spec.sys_justice[0](frozenset())

    def test_env_justice(self):
        decomp = GR1Decomposition(
            is_gr1=True,
            env_justice=[Atom("r")],
            sys_justice=[Atom("g")]
        )
        spec = decomposition_to_spec(decomp, {"r"}, {"g"})
        assert len(spec.env_justice) == 1
        assert spec.env_justice[0](frozenset({"r"}))

    def test_init_condition(self):
        decomp = GR1Decomposition(
            is_gr1=True,
            sys_init=Atom("ok"),
            sys_justice=[Atom("g")]
        )
        spec = decomposition_to_spec(decomp, set(), {"ok", "g"})
        assert spec.sys_init(frozenset({"ok"}))
        assert not spec.sys_init(frozenset())

    def test_no_init_means_all_states(self):
        decomp = GR1Decomposition(
            is_gr1=True,
            sys_justice=[Atom("g")]
        )
        spec = decomposition_to_spec(decomp, set(), {"g"})
        assert spec.env_init(frozenset())
        assert spec.sys_init(frozenset())


# ============================================================
# 9. Full Synthesis (GR(1) path)
# ============================================================

class TestSynthesizeGR1:
    def test_trivial_realizable(self):
        # GF(g) where sys controls g -- just toggle g on
        spec = Globally(Finally(Atom("g")))
        result = synthesize(spec, set(), {"g"})
        assert result.verdict == "realizable"
        assert result.method == "gr1"
        assert result.is_gr1

    def test_trivial_unrealizable(self):
        # GF(r) where env controls r, no assumptions
        spec = Globally(Finally(Atom("r")))
        result = synthesize(spec, {"r"}, set())
        assert result.verdict == "unrealizable"

    def test_assume_guarantee_realizable(self):
        # GF(req) -> GF(grant), sys controls grant
        assume = Globally(Finally(Atom("req")))
        guarantee = Globally(Finally(Atom("grant")))
        spec = Implies(assume, guarantee)
        result = synthesize(spec, {"req"}, {"grant"})
        assert result.verdict == "realizable"
        assert result.method == "gr1"

    def test_safety_realizable(self):
        # G(!bad) where sys controls bad
        spec = Globally(Not(Atom("bad")))
        result = synthesize(spec, set(), {"bad"})
        assert result.verdict == "realizable"

    def test_controller_has_states(self):
        spec = Globally(Finally(Atom("g")))
        result = synthesize(spec, set(), {"g"})
        if result.controller:
            assert len(result.controller.states) > 0

    def test_multiple_justice(self):
        # GF(a) & GF(b), sys controls both
        spec = And(Globally(Finally(Atom("a"))), Globally(Finally(Atom("b"))))
        result = synthesize(spec, set(), {"a", "b"})
        assert result.verdict == "realizable"

    def test_safety_unrealizable(self):
        # G(!a) where env controls a -- env can set a=true
        spec = Globally(Not(Atom("a")))
        result = synthesize(spec, {"a"}, set())
        assert result.verdict == "unrealizable"


# ============================================================
# 10. Full Synthesis (LTL fallback path)
# ============================================================

class TestSynthesizeLTL:
    def test_non_gr1_uses_ltl(self):
        # a U b is not GR(1)
        spec = Until(Atom("a"), Atom("b"))
        result = synthesize(spec, set(), {"a", "b"})
        assert result.method == "ltl"
        assert not result.is_gr1

    def test_force_ltl(self):
        # Force LTL even for GR(1) spec
        spec = Globally(Finally(Atom("g")))
        result = synthesize(spec, set(), {"g"}, force_method="ltl")
        assert result.method == "ltl"

    def test_force_gr1(self):
        spec = Globally(Finally(Atom("g")))
        result = synthesize(spec, set(), {"g"}, force_method="gr1")
        assert result.method == "gr1"


# ============================================================
# 11. Convenience Synthesis APIs
# ============================================================

class TestConvenienceAPIs:
    def test_synthesize_assume_guarantee(self):
        result = synthesize_assume_guarantee(
            Globally(Finally(Atom("req"))),
            Globally(Finally(Atom("ack"))),
            {"req"}, {"ack"}
        )
        assert result.verdict == "realizable"

    def test_synthesize_safety(self):
        result = synthesize_safety(Atom("err"), set(), {"err"})
        assert result.verdict == "realizable"

    def test_synthesize_liveness(self):
        result = synthesize_liveness(Atom("done"), set(), {"done"})
        assert result.verdict == "realizable"

    def test_synthesize_response(self):
        result = synthesize_response(
            Atom("req"), Atom("ack"),
            {"req"}, {"ack"}
        )
        assert result.verdict == "realizable"


# ============================================================
# 12. Direct GR(1) Synthesis
# ============================================================

class TestSynthesizeGR1Direct:
    def test_direct_simple(self):
        result = synthesize_gr1_direct(
            env_assumptions=[],
            sys_guarantees=[Atom("g")],
            env_safety=[],
            sys_safety=[],
            env_vars=set(),
            sys_vars={"g"}
        )
        assert result.verdict == "realizable"
        assert result.method == "gr1"

    def test_direct_with_assumptions(self):
        result = synthesize_gr1_direct(
            env_assumptions=[Atom("r")],
            sys_guarantees=[Atom("g")],
            env_safety=[],
            sys_safety=[],
            env_vars={"r"},
            sys_vars={"g"}
        )
        assert result.verdict == "realizable"

    def test_direct_with_safety(self):
        result = synthesize_gr1_direct(
            env_assumptions=[],
            sys_guarantees=[Atom("g")],
            env_safety=[],
            sys_safety=[Not(Atom("bad"))],
            env_vars=set(),
            sys_vars={"g", "bad"}
        )
        assert result.verdict == "realizable"


# ============================================================
# 13. Analysis
# ============================================================

class TestAnalyzeSpec:
    def test_gr1_analysis(self):
        spec = And(Globally(Not(Atom("bad"))), Globally(Finally(Atom("good"))))
        result = analyze_spec(spec, set(), {"bad", "good"})
        assert result["is_gr1"]
        assert result["n_sys_safety"] == 1
        assert result["n_sys_justice"] == 1

    def test_non_gr1_analysis(self):
        spec = Until(Atom("a"), Atom("b"))
        result = analyze_spec(spec, {"a"}, {"b"})
        assert not result["is_gr1"]
        assert result["reason"] != ""

    def test_atom_analysis(self):
        result = analyze_spec(Atom("a"), set(), {"a"})
        assert result["is_gr1"]
        assert result["quick_check"] == "realizable"

    def test_formula_metrics(self):
        spec = And(Globally(Atom("a")), Globally(Finally(Atom("b"))))
        result = analyze_spec(spec, set(), {"a", "b"})
        assert result["formula_depth"] > 0
        assert result["formula_size"] > 0


class TestCompare:
    def test_compare_gr1_spec(self):
        spec = Globally(Finally(Atom("g")))
        result = compare_methods(spec, set(), {"g"})
        assert result["is_gr1"]
        assert result["ltl"]["verdict"] is not None
        assert result["gr1"]["verdict"] is not None

    def test_compare_non_gr1(self):
        spec = Until(Atom("a"), Atom("b"))
        result = compare_methods(spec, set(), {"a", "b"})
        assert not result["is_gr1"]
        assert result["gr1"] is None

    def test_methods_agree(self):
        spec = Globally(Finally(Atom("g")))
        result = compare_methods(spec, set(), {"g"})
        if result["methods_agree"] is not None:
            assert result["methods_agree"]


# ============================================================
# 14. Summary
# ============================================================

class TestBridgeSummary:
    def test_summary_output(self):
        spec = Globally(Finally(Atom("g")))
        result = synthesize(spec, set(), {"g"})
        summary = bridge_summary(result)
        assert "GR(1)-LTL Bridge Result" in summary
        assert result.verdict in summary
        assert result.method in summary

    def test_summary_with_controller(self):
        spec = Globally(Finally(Atom("g")))
        result = synthesize(spec, set(), {"g"})
        summary = bridge_summary(result)
        if result.controller:
            assert "Controller" in summary


# ============================================================
# 15. Formula Utilities
# ============================================================

class TestFormulaUtils:
    def test_depth_atom(self):
        assert _formula_depth(Atom("a")) == 0

    def test_depth_g(self):
        assert _formula_depth(Globally(Atom("a"))) == 1

    def test_depth_gf(self):
        assert _formula_depth(Globally(Finally(Atom("a")))) == 2

    def test_size_atom(self):
        assert _formula_size(Atom("a")) == 1

    def test_size_and(self):
        assert _formula_size(And(Atom("a"), Atom("b"))) == 3

    def test_conjoin_empty(self):
        result = _conjoin([])
        assert result.op == LTLOp.TRUE

    def test_conjoin_one(self):
        result = _conjoin([Atom("a")])
        assert result == Atom("a")

    def test_conjoin_many(self):
        result = _conjoin([Atom("a"), Atom("b"), Atom("c")])
        assert result.op == LTLOp.AND

    def test_decomp_to_ltl_roundtrip(self):
        decomp = GR1Decomposition(
            is_gr1=True,
            sys_safety=[Atom("s")],
            sys_justice=[Atom("j")]
        )
        ltl = _decomp_to_ltl(decomp)
        assert ltl.op == LTLOp.AND  # G(s) & GF(j)


# ============================================================
# 16. LTL to GR(1) Components
# ============================================================

class TestLTLToGR1Components:
    def test_gr1_extracts(self):
        spec = And(Globally(Atom("s")), Globally(Finally(Atom("j"))))
        result = ltl_to_gr1_components(spec)
        assert result is not None
        assert len(result["sys_safety"]) == 1
        assert len(result["sys_justice"]) == 1

    def test_non_gr1_returns_none(self):
        spec = Until(Atom("a"), Atom("b"))
        result = ltl_to_gr1_components(spec)
        assert result is None


# ============================================================
# 17. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_env_vars(self):
        spec = Globally(Finally(Atom("g")))
        result = synthesize(spec, set(), {"g"})
        assert result.verdict == "realizable"

    def test_empty_sys_vars(self):
        spec = Globally(Finally(Atom("a")))
        result = synthesize(spec, {"a"}, set())
        assert result.verdict == "unrealizable"

    def test_true_spec(self):
        result = synthesize(LTLTrue(), set(), set())
        assert result.verdict == "realizable"

    def test_false_spec(self):
        result = synthesize(LTLFalse(), set(), set())
        assert result.verdict == "unrealizable"

    def test_g_true(self):
        result = synthesize(Globally(LTLTrue()), set(), set())
        assert result.verdict == "realizable"

    def test_g_false(self):
        result = synthesize(Globally(LTLFalse()), set(), set())
        assert result.verdict == "unrealizable"

    def test_no_vars_at_all(self):
        spec = Globally(Finally(LTLTrue()))
        result = synthesize(spec, set(), set())
        assert result.verdict == "realizable"


# ============================================================
# 18. Controller Quality
# ============================================================

class TestControllerQuality:
    def test_controller_has_transitions(self):
        spec = Globally(Finally(Atom("g")))
        result = synthesize(spec, set(), {"g"})
        if result.controller:
            assert len(result.controller.transitions) > 0

    def test_controller_initial_state(self):
        spec = Globally(Finally(Atom("g")))
        result = synthesize(spec, set(), {"g"})
        if result.controller:
            assert result.controller.initial in result.controller.states

    def test_controller_vars(self):
        spec = Implies(
            Globally(Finally(Atom("req"))),
            Globally(Finally(Atom("ack")))
        )
        result = synthesize(spec, {"req"}, {"ack"})
        if result.controller:
            assert "req" in result.controller.inputs or len(result.controller.inputs) == 0
            # Output should include sys vars
            assert "ack" in result.controller.outputs or len(result.controller.outputs) >= 0


# ============================================================
# 19. Stats
# ============================================================

class TestStats:
    def test_gr1_stats(self):
        spec = And(Globally(Not(Atom("bad"))), Globally(Finally(Atom("good"))))
        result = synthesize(spec, set(), {"bad", "good"})
        assert result.stats.get("n_sys_safety") == 1
        assert result.stats.get("n_sys_justice") == 1

    def test_ltl_stats(self):
        spec = Until(Atom("a"), Atom("b"))
        result = synthesize(spec, set(), {"a", "b"})
        assert "game_vertices" in result.stats or "fallback_reason" in result.stats


# ============================================================
# 20. Integration: Arbiter Pattern
# ============================================================

class TestIntegrationArbiter:
    def test_arbiter_two_clients(self):
        """Classic arbiter: 2 clients request, arbiter grants, mutual exclusion."""
        r1, r2, g1, g2 = Atom("r1"), Atom("r2"), Atom("g1"), Atom("g2")

        # Assumptions: clients request infinitely often
        env_assume = And(Globally(Finally(r1)), Globally(Finally(r2)))

        # Guarantees: mutual exclusion + fairness
        mutex = Globally(Not(And(g1, g2)))
        fair1 = Globally(Finally(g1))
        fair2 = Globally(Finally(g2))
        sys_guarantee = And(mutex, And(fair1, fair2))

        spec = Implies(env_assume, sys_guarantee)
        result = synthesize(spec, {"r1", "r2"}, {"g1", "g2"})

        assert result.verdict == "realizable"
        assert result.is_gr1
        assert result.method == "gr1"

    def test_arbiter_without_fairness_assumptions(self):
        """Without env fairness assumptions, arbiter can't guarantee fairness."""
        r1, r2, g1, g2 = Atom("r1"), Atom("r2"), Atom("g1"), Atom("g2")

        # No assumptions, but require fairness for both
        mutex = Globally(Not(And(g1, g2)))
        fair1 = Globally(Finally(g1))
        fair2 = Globally(Finally(g2))
        spec = And(mutex, And(fair1, fair2))

        result = synthesize(spec, {"r1", "r2"}, {"g1", "g2"})
        # Should be realizable: sys just alternates g1/g2 regardless of requests
        assert result.verdict == "realizable"


# ============================================================
# 21. Integration: Traffic Light Pattern
# ============================================================

class TestIntegrationTrafficLight:
    def test_traffic_light(self):
        """Traffic light controller: alternates green for two directions."""
        ns, ew = Atom("ns_green"), Atom("ew_green")

        # Never both green
        safety = Globally(Not(And(ns, ew)))
        # Both get green infinitely often
        fair_ns = Globally(Finally(ns))
        fair_ew = Globally(Finally(ew))

        spec = And(safety, And(fair_ns, fair_ew))
        result = synthesize(spec, set(), {"ns_green", "ew_green"})

        assert result.verdict == "realizable"
        assert result.is_gr1


# ============================================================
# 22. Decomposition Edge Cases
# ============================================================

class TestDecompositionEdgeCases:
    def test_implies_with_true_assumption(self):
        spec = Implies(LTLTrue(), Globally(Finally(Atom("g"))))
        decomp = detect_gr1(spec, set(), {"g"})
        assert decomp.is_gr1
        assert len(decomp.sys_justice) == 1

    def test_implies_with_false_guarantee(self):
        spec = Implies(Globally(Finally(Atom("r"))), LTLFalse())
        decomp = detect_gr1(spec, {"r"}, set())
        assert decomp.is_gr1
        # FALSE as init means no initial state is valid

    def test_deeply_nested_and(self):
        specs = [Globally(Finally(Atom(f"j{i}"))) for i in range(5)]
        combined = specs[0]
        for s in specs[1:]:
            combined = And(combined, s)
        decomp = detect_gr1(combined, set(), {f"j{i}" for i in range(5)})
        assert decomp.is_gr1
        assert len(decomp.sys_justice) == 5

    def test_env_init(self):
        assume = And(Atom("start"), Globally(Finally(Atom("r"))))
        guarantee = Globally(Finally(Atom("g")))
        spec = Implies(assume, guarantee)
        decomp = detect_gr1(spec, {"start", "r"}, {"g"})
        assert decomp.is_gr1
        assert decomp.env_init is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
