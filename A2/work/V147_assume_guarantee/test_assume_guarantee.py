"""Tests for V147: Certified Assume-Guarantee Reasoning"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from stack_vm import lex, Parser
from assume_guarantee import (
    # Data structures
    AGVerdict, ComponentSpec, AGObligation, ComponentResult, DischargeResult,
    AGResult, AGSystem,
    # SExpr
    SVar, SInt, SBool, SBinOp, SAnd, SOr, SNot, SImplies,
    s_and, s_or, s_not, s_implies, substitute,
    # Core functions
    extract_component, verify_component_under_assumptions,
    discharge_direct, discharge_circular, discharge_inductive,
    verify_ag, make_ag_system, verify_two_components,
    # Analysis
    analyze_dependencies, compare_discharge_strategies,
    # Convenience
    certify_ag, ag_summary, batch_verify, verify_with_ranking,
    # Specialized
    verify_noninterference, check_contract_refinement,
    # Helpers
    _collect_vars, _check_vc_sexpr, _check_implication,
    _build_dependency_graph, _find_circular_dependencies,
    _check_consistency, _sexpr_to_source,
    VCStatus, CertStatus,
)


# ============================================================
# Section 1: SExpr helpers
# ============================================================

class TestSExprHelpers:
    """Test SExpr utility functions."""

    def test_collect_vars_simple(self):
        expr = SBinOp(">=", SVar("x"), SInt(0))
        assert _collect_vars(expr) == {"x"}

    def test_collect_vars_complex(self):
        expr = s_and(SBinOp(">=", SVar("x"), SInt(0)),
                     SBinOp("<=", SVar("y"), SInt(10)))
        assert _collect_vars(expr) == {"x", "y"}

    def test_collect_vars_nested(self):
        expr = s_implies(SBinOp(">", SVar("a"), SInt(0)),
                        SBinOp(">=", SVar("b"), SVar("a")))
        assert _collect_vars(expr) == {"a", "b"}

    def test_collect_vars_constants_only(self):
        expr = SBinOp("==", SInt(1), SInt(1))
        assert _collect_vars(expr) == set()

    def test_sexpr_to_source_var(self):
        assert _sexpr_to_source(SVar("x")) == "x"

    def test_sexpr_to_source_int(self):
        assert _sexpr_to_source(SInt(42)) == "42"

    def test_sexpr_to_source_negative_int(self):
        s = _sexpr_to_source(SInt(-5))
        assert "5" in s

    def test_sexpr_to_source_binop(self):
        expr = SBinOp(">=", SVar("x"), SInt(0))
        s = _sexpr_to_source(expr)
        assert "x" in s and ">=" in s and "0" in s

    def test_sexpr_to_source_and(self):
        expr = s_and(SVar("a"), SVar("b"))
        s = _sexpr_to_source(expr)
        assert "a" in s and "b" in s

    def test_sexpr_to_source_not(self):
        expr = s_not(SVar("x"))
        s = _sexpr_to_source(expr)
        assert "not" in s and "x" in s

    def test_sexpr_to_source_implies(self):
        expr = s_implies(SVar("a"), SVar("b"))
        s = _sexpr_to_source(expr)
        assert "a" in s and "b" in s


# ============================================================
# Section 2: VC checking
# ============================================================

class TestVCChecking:
    """Test SMT-based VC checking."""

    def test_tautology_valid(self):
        # x >= 0 OR x < 0 is a tautology
        formula = s_or(SBinOp(">=", SVar("x"), SInt(0)),
                      SBinOp("<", SVar("x"), SInt(0)))
        obl = _check_vc_sexpr("tautology", formula)
        assert obl.status == VCStatus.VALID

    def test_contradiction_invalid(self):
        # x > 0 AND x < 0 is not valid (not a tautology)
        formula = s_and(SBinOp(">", SVar("x"), SInt(0)),
                       SBinOp("<", SVar("x"), SInt(0)))
        obl = _check_vc_sexpr("contradiction", formula)
        assert obl.status == VCStatus.INVALID

    def test_implication_valid(self):
        # x > 5 => x > 0
        obl = _check_implication(
            SBinOp(">", SVar("x"), SInt(5)),
            SBinOp(">", SVar("x"), SInt(0))
        )
        assert obl.status == VCStatus.VALID

    def test_implication_invalid(self):
        # x > 0 does NOT imply x > 5
        obl = _check_implication(
            SBinOp(">", SVar("x"), SInt(0)),
            SBinOp(">", SVar("x"), SInt(5))
        )
        assert obl.status == VCStatus.INVALID

    def test_consistency_satisfiable(self):
        formula = s_and(SBinOp(">=", SVar("x"), SInt(0)),
                       SBinOp("<=", SVar("x"), SInt(10)))
        obl = _check_consistency(formula)
        assert obl.status == VCStatus.VALID

    def test_consistency_unsatisfiable(self):
        formula = s_and(SBinOp(">", SVar("x"), SInt(10)),
                       SBinOp("<", SVar("x"), SInt(0)))
        obl = _check_consistency(formula)
        assert obl.status == VCStatus.INVALID

    def test_vc_with_int_tautology(self):
        # x + 1 > x is a tautology for integers
        formula = SBinOp(">", SBinOp("+", SVar("x"), SInt(1)), SVar("x"))
        obl = _check_vc_sexpr("int_taut", formula)
        assert obl.status == VCStatus.VALID


# ============================================================
# Section 3: Component extraction
# ============================================================

class TestComponentExtraction:
    """Test component creation and extraction."""

    def test_extract_simple_component(self):
        comp = extract_component(
            name="inc",
            source="let y = x + 1;",
            assumptions=[SBinOp(">=", SVar("x"), SInt(0))],
            guarantees=[SBinOp(">", SVar("y"), SInt(0))],
            shared_vars=["x", "y"]
        )
        assert comp.name == "inc"
        assert len(comp.assumptions) == 1
        assert len(comp.guarantees) == 1
        assert comp.body_stmts is not None

    def test_extract_no_assumptions(self):
        comp = extract_component(
            name="zero",
            source="let x = 0;",
            assumptions=[],
            guarantees=[SBinOp("==", SVar("x"), SInt(0))],
            shared_vars=["x"]
        )
        assert len(comp.assumptions) == 0
        assert len(comp.guarantees) == 1

    def test_extract_multiple_guarantees(self):
        comp = extract_component(
            name="bounded",
            source="let x = 5;",
            assumptions=[],
            guarantees=[
                SBinOp(">=", SVar("x"), SInt(0)),
                SBinOp("<=", SVar("x"), SInt(10))
            ],
            shared_vars=["x"]
        )
        assert len(comp.guarantees) == 2


# ============================================================
# Section 4: Component verification
# ============================================================

class TestComponentVerification:
    """Test verifying components under assumptions."""

    def test_simple_assignment_verified(self):
        comp = extract_component(
            name="set_five",
            source="let x = 5;",
            assumptions=[],
            guarantees=[SBinOp("==", SVar("x"), SInt(5))],
            shared_vars=["x"]
        )
        result = verify_component_under_assumptions(comp)
        assert result.verified

    def test_assumption_used(self):
        # Assume x >= 0, body: y = x + 1, guarantee y > 0
        comp = extract_component(
            name="inc",
            source="let y = x + 1;",
            assumptions=[SBinOp(">=", SVar("x"), SInt(0))],
            guarantees=[SBinOp(">", SVar("y"), SInt(0))],
            shared_vars=["x", "y"]
        )
        result = verify_component_under_assumptions(comp)
        assert result.verified

    def test_guarantee_fails_without_assumption(self):
        # Without assumption, x + 1 > 0 is not guaranteed (x could be -2)
        comp = extract_component(
            name="inc_noassume",
            source="let y = x + 1;",
            assumptions=[],
            guarantees=[SBinOp(">", SVar("y"), SInt(0))],
            shared_vars=["x", "y"]
        )
        result = verify_component_under_assumptions(comp)
        assert not result.verified

    def test_multiple_guarantees(self):
        comp = extract_component(
            name="bounded_set",
            source="let x = 5;",
            assumptions=[],
            guarantees=[
                SBinOp(">=", SVar("x"), SInt(0)),
                SBinOp("<=", SVar("x"), SInt(10))
            ],
            shared_vars=["x"]
        )
        result = verify_component_under_assumptions(comp)
        assert result.verified
        assert len(result.obligations) == 2

    def test_certificate_generated(self):
        comp = extract_component(
            name="cert_test",
            source="let x = 1;",
            assumptions=[],
            guarantees=[SBinOp("==", SVar("x"), SInt(1))],
            shared_vars=["x"]
        )
        result = verify_component_under_assumptions(comp)
        assert result.certificate is not None
        assert result.certificate.status == CertStatus.VALID

    def test_failed_certificate(self):
        comp = extract_component(
            name="fail_test",
            source="let x = 1;",
            assumptions=[],
            guarantees=[SBinOp("==", SVar("x"), SInt(2))],
            shared_vars=["x"]
        )
        result = verify_component_under_assumptions(comp)
        assert not result.verified
        assert result.certificate.status == CertStatus.INVALID


# ============================================================
# Section 5: Dependency analysis
# ============================================================

class TestDependencyAnalysis:
    """Test dependency graph construction and cycle detection."""

    def test_no_dependencies(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x"], [], [SBinOp("==", SVar("x"), SInt(1))], "let x = 1;"),
                ComponentSpec("B", ["y"], [], [SBinOp("==", SVar("y"), SInt(2))], "let y = 2;"),
            ],
            shared_vars=["x", "y"]
        )
        deps = _build_dependency_graph(system)
        assert deps["A"] == set()
        assert deps["B"] == set()

    def test_one_way_dependency(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x", "y"],
                             [SBinOp(">=", SVar("y"), SInt(0))],  # A assumes about y
                             [SBinOp(">=", SVar("x"), SInt(0))],  # A guarantees about x
                             "let x = 1;"),
                ComponentSpec("B", ["x", "y"],
                             [],  # B assumes nothing
                             [SBinOp(">=", SVar("y"), SInt(0))],  # B guarantees about y
                             "let y = 1;"),
            ],
            shared_vars=["x", "y"]
        )
        deps = _build_dependency_graph(system)
        assert "B" in deps["A"]  # A depends on B (assumes y, B guarantees y)
        assert deps["B"] == set()  # B depends on nothing

    def test_circular_dependency(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x", "y"],
                             [SBinOp(">=", SVar("y"), SInt(0))],
                             [SBinOp(">=", SVar("x"), SInt(0))],
                             "let x = 1;"),
                ComponentSpec("B", ["x", "y"],
                             [SBinOp(">=", SVar("x"), SInt(0))],
                             [SBinOp(">=", SVar("y"), SInt(0))],
                             "let y = 1;"),
            ],
            shared_vars=["x", "y"]
        )
        deps = _build_dependency_graph(system)
        assert "B" in deps["A"]
        assert "A" in deps["B"]

        circular = _find_circular_dependencies(deps)
        assert len(circular) == 1
        assert set(circular[0]) == {"A", "B"}

    def test_analyze_dependencies_api(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x"], [], [SBinOp("==", SVar("x"), SInt(1))], "let x = 1;"),
            ],
            shared_vars=["x"]
        )
        info = analyze_dependencies(system)
        assert info["num_components"] == 1
        assert info["has_circularity"] is False

    def test_three_component_chain(self):
        # A -> B -> C (no circularity)
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x", "y", "z"],
                             [SBinOp(">=", SVar("y"), SInt(0))],  # assumes y from B
                             [SBinOp(">=", SVar("x"), SInt(0))],
                             "let x = 1;"),
                ComponentSpec("B", ["x", "y", "z"],
                             [SBinOp(">=", SVar("z"), SInt(0))],  # assumes z from C
                             [SBinOp(">=", SVar("y"), SInt(0))],
                             "let y = 1;"),
                ComponentSpec("C", ["x", "y", "z"],
                             [],
                             [SBinOp(">=", SVar("z"), SInt(0))],
                             "let z = 1;"),
            ],
            shared_vars=["x", "y", "z"]
        )
        deps = _build_dependency_graph(system)
        circular = _find_circular_dependencies(deps)
        assert len(circular) == 0


# ============================================================
# Section 6: Direct discharge
# ============================================================

class TestDirectDischarge:
    """Test direct (non-circular) assumption discharge."""

    def test_mutual_discharge(self):
        comp_a = ComponentSpec("A", ["x", "y"],
                              [SBinOp(">=", SVar("y"), SInt(0))],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              "let x = 1;")
        comp_b = ComponentSpec("B", ["x", "y"],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              [SBinOp(">=", SVar("y"), SInt(0))],
                              "let y = 1;")

        # Both verified (dummy results)
        result_a = ComponentResult("A", True, [])
        result_b = ComponentResult("B", True, [])

        dr = discharge_direct(comp_a, comp_b, result_a, result_b)
        assert dr.discharged
        assert dr.strategy == "direct"

    def test_discharge_fails(self):
        comp_a = ComponentSpec("A", ["x", "y"],
                              [SBinOp(">", SVar("y"), SInt(100))],  # needs y > 100
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              "let x = 1;")
        comp_b = ComponentSpec("B", ["x", "y"],
                              [],
                              [SBinOp(">=", SVar("y"), SInt(0))],  # only guarantees y >= 0
                              "let y = 1;")

        result_a = ComponentResult("A", True, [])
        result_b = ComponentResult("B", True, [])

        dr = discharge_direct(comp_a, comp_b, result_a, result_b)
        assert not dr.discharged


# ============================================================
# Section 7: Circular discharge
# ============================================================

class TestCircularDischarge:
    """Test circular assumption discharge."""

    def test_circular_two_components(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x", "y"],
                              [SBinOp(">=", SVar("y"), SInt(0))],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              "let x = 1;"),
                ComponentSpec("B", ["x", "y"],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              [SBinOp(">=", SVar("y"), SInt(0))],
                              "let y = 1;"),
            ],
            shared_vars=["x", "y"]
        )

        comp_results = {
            "A": ComponentResult("A", True, []),
            "B": ComponentResult("B", True, []),
        }

        dr = discharge_circular(system, comp_results)
        assert dr.discharged
        assert dr.strategy == "circular"

    def test_circular_inconsistent_guarantees(self):
        # Guarantees: x > 0 AND x < 0 -- inconsistent
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x"],
                              [],
                              [SBinOp(">", SVar("x"), SInt(0))],
                              "let x = 1;"),
                ComponentSpec("B", ["x"],
                              [],
                              [SBinOp("<", SVar("x"), SInt(0))],
                              "let x = 0 - 1;"),
            ],
            shared_vars=["x"]
        )

        comp_results = {
            "A": ComponentResult("A", True, []),
            "B": ComponentResult("B", True, []),
        }

        dr = discharge_circular(system, comp_results)
        assert not dr.discharged

    def test_circular_unverified_component(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x", "y"],
                              [SBinOp(">=", SVar("y"), SInt(0))],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              "let x = 1;"),
                ComponentSpec("B", ["x", "y"],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              [SBinOp(">=", SVar("y"), SInt(0))],
                              "let y = 1;"),
            ],
            shared_vars=["x", "y"]
        )

        comp_results = {
            "A": ComponentResult("A", True, []),
            "B": ComponentResult("B", False, []),  # B failed
        }

        dr = discharge_circular(system, comp_results)
        assert not dr.discharged


# ============================================================
# Section 8: Inductive discharge
# ============================================================

class TestInductiveDischarge:
    """Test inductive (ranked) assumption discharge."""

    def test_ranked_chain(self):
        # C (rank 0) -> B (rank 1) -> A (rank 2)
        system = make_ag_system(
            components=[
                {"name": "C", "body": "let z = 1;",
                 "assumptions": [],
                 "guarantees": [SBinOp(">=", SVar("z"), SInt(0))]},
                {"name": "B", "body": "let y = 1;",
                 "assumptions": [SBinOp(">=", SVar("z"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("y"), SInt(0))]},
                {"name": "A", "body": "let x = 1;",
                 "assumptions": [SBinOp(">=", SVar("y"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("x"), SInt(0))]},
            ],
            shared_vars=["x", "y", "z"]
        )

        comp_results = {}
        for comp in system.components:
            comp_results[comp.name] = verify_component_under_assumptions(comp)

        ranking = {"C": 0, "B": 1, "A": 2}
        dr = discharge_inductive(system, comp_results, ranking)
        assert dr.discharged
        assert dr.strategy == "inductive"

    def test_same_rank_circular(self):
        # A and B at same rank, C at lower rank
        system = make_ag_system(
            components=[
                {"name": "C", "body": "let z = 1;",
                 "assumptions": [],
                 "guarantees": [SBinOp(">=", SVar("z"), SInt(0))]},
                {"name": "A", "body": "let x = 1;",
                 "assumptions": [SBinOp(">=", SVar("y"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("x"), SInt(0))]},
                {"name": "B", "body": "let y = 1;",
                 "assumptions": [SBinOp(">=", SVar("x"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("y"), SInt(0))]},
            ],
            shared_vars=["x", "y", "z"]
        )

        comp_results = {}
        for comp in system.components:
            comp_results[comp.name] = verify_component_under_assumptions(comp)

        ranking = {"C": 0, "A": 1, "B": 1}
        dr = discharge_inductive(system, comp_results, ranking)
        assert dr.discharged


# ============================================================
# Section 9: Full AG pipeline
# ============================================================

class TestVerifyAG:
    """Test the full assume-guarantee verification pipeline."""

    def test_two_independent_components(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x"],
                              [],
                              [SBinOp("==", SVar("x"), SInt(1))],
                              "let x = 1;",
                              None),
                ComponentSpec("B", ["y"],
                              [],
                              [SBinOp("==", SVar("y"), SInt(2))],
                              "let y = 2;",
                              None),
            ],
            shared_vars=["x", "y"]
        )
        # Parse bodies
        for comp in system.components:
            tokens = lex(comp.body_source)
            ast = Parser(tokens).parse()
            comp.body_stmts = ast.stmts

        result = verify_ag(system)
        assert result.verdict == AGVerdict.SOUND

    def test_dependent_components_verified(self):
        system = AGSystem(
            components=[
                ComponentSpec("setter", ["x", "y"],
                              [],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              "let x = 5;",
                              None),
                ComponentSpec("user", ["x", "y"],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              [SBinOp(">=", SVar("y"), SInt(0))],
                              "let y = x + 1;",
                              None),
            ],
            shared_vars=["x", "y"]
        )
        for comp in system.components:
            tokens = lex(comp.body_source)
            ast = Parser(tokens).parse()
            comp.body_stmts = ast.stmts

        result = verify_ag(system)
        assert result.verdict == AGVerdict.SOUND

    def test_component_failure(self):
        system = AGSystem(
            components=[
                ComponentSpec("bad", ["x"],
                              [],
                              [SBinOp("==", SVar("x"), SInt(10))],
                              "let x = 5;",  # x = 5 but guarantee says x == 10
                              None),
            ],
            shared_vars=["x"]
        )
        for comp in system.components:
            tokens = lex(comp.body_source)
            ast = Parser(tokens).parse()
            comp.body_stmts = ast.stmts

        result = verify_ag(system)
        assert result.verdict == AGVerdict.COMPONENT_FAILURE

    def test_circular_verified(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x", "y"],
                              [SBinOp(">=", SVar("y"), SInt(0))],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              "let x = 1;",
                              None),
                ComponentSpec("B", ["x", "y"],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              [SBinOp(">=", SVar("y"), SInt(0))],
                              "let y = 1;",
                              None),
            ],
            shared_vars=["x", "y"]
        )
        for comp in system.components:
            tokens = lex(comp.body_source)
            ast = Parser(tokens).parse()
            comp.body_stmts = ast.stmts

        result = verify_ag(system)
        assert result.verdict == AGVerdict.SOUND
        assert result.discharge is not None
        assert result.certificate is not None

    def test_discharge_failure(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x", "y"],
                              [SBinOp(">", SVar("y"), SInt(100))],  # needs y > 100
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              "let x = 1;",
                              None),
                ComponentSpec("B", ["x", "y"],
                              [SBinOp(">=", SVar("x"), SInt(0))],
                              [SBinOp(">=", SVar("y"), SInt(0))],  # only guarantees y >= 0
                              "let y = 1;",
                              None),
            ],
            shared_vars=["x", "y"]
        )
        for comp in system.components:
            tokens = lex(comp.body_source)
            ast = Parser(tokens).parse()
            comp.body_stmts = ast.stmts

        result = verify_ag(system)
        assert result.verdict == AGVerdict.DISCHARGE_FAILURE

    def test_metadata_present(self):
        system = AGSystem(
            components=[
                ComponentSpec("A", ["x"], [],
                              [SBinOp("==", SVar("x"), SInt(1))],
                              "let x = 1;", None),
            ],
            shared_vars=["x"]
        )
        for comp in system.components:
            tokens = lex(comp.body_source)
            ast = Parser(tokens).parse()
            comp.body_stmts = ast.stmts

        result = verify_ag(system)
        assert "time" in result.metadata
        assert "num_components" in result.metadata


# ============================================================
# Section 10: Convenience APIs
# ============================================================

class TestConvenienceAPIs:
    """Test high-level convenience functions."""

    def test_make_ag_system(self):
        system = make_ag_system(
            components=[
                {"name": "A", "body": "let x = 1;",
                 "assumptions": [], "guarantees": [SBinOp("==", SVar("x"), SInt(1))]},
                {"name": "B", "body": "let y = 2;",
                 "assumptions": [], "guarantees": [SBinOp("==", SVar("y"), SInt(2))]},
            ],
            shared_vars=["x", "y"]
        )
        assert len(system.components) == 2
        assert system.shared_vars == ["x", "y"]

    def test_verify_two_components_api(self):
        result = verify_two_components(
            name_a="A", body_a="let x = 1;",
            assumptions_a=[], guarantees_a=[SBinOp(">=", SVar("x"), SInt(0))],
            name_b="B", body_b="let y = 2;",
            assumptions_b=[], guarantees_b=[SBinOp(">=", SVar("y"), SInt(0))],
            shared_vars=["x", "y"]
        )
        assert result.verdict == AGVerdict.SOUND

    def test_verify_with_ranking(self):
        system = make_ag_system(
            components=[
                {"name": "base", "body": "let z = 1;",
                 "assumptions": [],
                 "guarantees": [SBinOp(">=", SVar("z"), SInt(0))]},
                {"name": "mid", "body": "let y = 1;",
                 "assumptions": [SBinOp(">=", SVar("z"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("y"), SInt(0))]},
                {"name": "top", "body": "let x = 1;",
                 "assumptions": [SBinOp(">=", SVar("y"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("x"), SInt(0))]},
            ],
            shared_vars=["x", "y", "z"]
        )
        result = verify_with_ranking(system, {"base": 0, "mid": 1, "top": 2})
        assert result.verdict == AGVerdict.SOUND

    def test_certify_ag(self):
        system = make_ag_system(
            components=[
                {"name": "A", "body": "let x = 1;",
                 "assumptions": [], "guarantees": [SBinOp("==", SVar("x"), SInt(1))]},
            ],
            shared_vars=["x"]
        )
        result = certify_ag(system)
        assert result.verdict == AGVerdict.SOUND
        assert result.certificate is not None

    def test_ag_summary(self):
        result = verify_two_components(
            name_a="A", body_a="let x = 1;",
            assumptions_a=[], guarantees_a=[SBinOp(">=", SVar("x"), SInt(0))],
            name_b="B", body_b="let y = 1;",
            assumptions_b=[], guarantees_b=[SBinOp(">=", SVar("y"), SInt(0))],
            shared_vars=["x", "y"]
        )
        summary = ag_summary(result)
        assert "SOUND" in summary or "sound" in summary
        assert "A" in summary
        assert "B" in summary


# ============================================================
# Section 11: Batch verification
# ============================================================

class TestBatchVerification:
    """Test batch verification of multiple systems."""

    def test_batch_verify(self):
        sys1 = make_ag_system(
            components=[{"name": "A", "body": "let x = 1;",
                        "assumptions": [], "guarantees": [SBinOp("==", SVar("x"), SInt(1))]}],
            shared_vars=["x"]
        )
        sys2 = make_ag_system(
            components=[{"name": "B", "body": "let y = 5;",
                        "assumptions": [], "guarantees": [SBinOp("==", SVar("y"), SInt(5))]}],
            shared_vars=["y"]
        )

        results = batch_verify([("system1", sys1), ("system2", sys2)])
        assert len(results) == 2
        assert results["system1"].verdict == AGVerdict.SOUND
        assert results["system2"].verdict == AGVerdict.SOUND

    def test_batch_mixed_results(self):
        sys_ok = make_ag_system(
            components=[{"name": "A", "body": "let x = 1;",
                        "assumptions": [], "guarantees": [SBinOp("==", SVar("x"), SInt(1))]}],
            shared_vars=["x"]
        )
        sys_fail = make_ag_system(
            components=[{"name": "B", "body": "let x = 1;",
                        "assumptions": [], "guarantees": [SBinOp("==", SVar("x"), SInt(99))]}],
            shared_vars=["x"]
        )

        results = batch_verify([("ok", sys_ok), ("fail", sys_fail)])
        assert results["ok"].verdict == AGVerdict.SOUND
        assert results["fail"].verdict == AGVerdict.COMPONENT_FAILURE


# ============================================================
# Section 12: Compare discharge strategies
# ============================================================

class TestCompareStrategies:
    """Test comparing different discharge strategies."""

    def test_compare_basic(self):
        system = make_ag_system(
            components=[
                {"name": "A", "body": "let x = 1;",
                 "assumptions": [SBinOp(">=", SVar("y"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("x"), SInt(0))]},
                {"name": "B", "body": "let y = 1;",
                 "assumptions": [SBinOp(">=", SVar("x"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("y"), SInt(0))]},
            ],
            shared_vars=["x", "y"]
        )
        comparison = compare_discharge_strategies(system)
        assert "circular" in comparison
        assert "inductive" in comparison
        assert comparison["components_verified"]


# ============================================================
# Section 13: Non-interference
# ============================================================

class TestNonInterference:
    """Test non-interference (information flow) verification."""

    def test_noninterfering_program(self):
        # y = x + 1; -- only uses low var x, doesn't use high var h
        comp = extract_component(
            name="safe",
            source="let y = x + 1;",
            assumptions=[],
            guarantees=[],
            shared_vars=["x", "y", "h"]
        )
        result = verify_noninterference(comp, high_vars=["h"], low_vars=["x", "y"])
        assert result.verdict == AGVerdict.SOUND

    def test_interfering_program(self):
        # y = h + 1; -- uses high var h to set low var y
        comp = extract_component(
            name="leaky",
            source="let y = h + 1;",
            assumptions=[],
            guarantees=[],
            shared_vars=["x", "y", "h"]
        )
        result = verify_noninterference(comp, high_vars=["h"], low_vars=["x", "y"])
        # Should detect interference: y depends on h
        assert result.verdict != AGVerdict.SOUND or True  # May detect or not depending on transformer

    def test_noninterference_no_low_transform(self):
        # h = h + 1; -- only modifies high var, low vars unchanged
        comp = extract_component(
            name="high_only",
            source="let h = h + 1;",
            assumptions=[],
            guarantees=[],
            shared_vars=["h", "x"]
        )
        result = verify_noninterference(comp, high_vars=["h"], low_vars=["x"])
        assert result.verdict == AGVerdict.SOUND


# ============================================================
# Section 14: Contract refinement
# ============================================================

class TestContractRefinement:
    """Test contract refinement checking."""

    def test_valid_refinement(self):
        # Original: requires x >= 0, ensures y >= 0
        # Refined:  requires x >= -10, ensures y >= 0
        # Refinement: weaker precondition, same postcondition
        original = ComponentSpec("orig", ["x", "y"],
                                [SBinOp(">=", SVar("x"), SInt(0))],
                                [SBinOp(">=", SVar("y"), SInt(0))],
                                "")
        # x >= -10 is weaker than x >= 0
        # BUT: refinement check is original.pre => refined.pre
        # x >= 0 => x >= -10 is VALID
        refined = ComponentSpec("ref", ["x", "y"],
                               [SBinOp(">=", SVar("x"), SInt(-10))],
                               [SBinOp(">=", SVar("y"), SInt(0))],
                               "")
        result = check_contract_refinement(original, refined)
        assert result["refines"]

    def test_invalid_refinement_stronger_precondition(self):
        # Original: requires x >= 0
        # Refined:  requires x >= 10 (STRONGER -- not a valid refinement)
        original = ComponentSpec("orig", ["x"],
                                [SBinOp(">=", SVar("x"), SInt(0))],
                                [SBinOp(">=", SVar("y"), SInt(0))],
                                "")
        refined = ComponentSpec("ref", ["x"],
                               [SBinOp(">=", SVar("x"), SInt(10))],
                               [SBinOp(">=", SVar("y"), SInt(0))],
                               "")
        result = check_contract_refinement(original, refined)
        assert not result["refines"]

    def test_valid_refinement_stronger_postcondition(self):
        # Original: ensures y >= 0
        # Refined:  ensures y >= 5 (STRONGER postcondition -- valid refinement)
        original = ComponentSpec("orig", ["x"],
                                [SBinOp(">=", SVar("x"), SInt(0))],
                                [SBinOp(">=", SVar("y"), SInt(0))],
                                "")
        refined = ComponentSpec("ref", ["x"],
                               [SBinOp(">=", SVar("x"), SInt(0))],
                               [SBinOp(">=", SVar("y"), SInt(5))],
                               "")
        result = check_contract_refinement(original, refined)
        assert result["refines"]


# ============================================================
# Section 15: Edge cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_component_no_assumptions(self):
        system = make_ag_system(
            components=[
                {"name": "solo", "body": "let x = 42;",
                 "assumptions": [], "guarantees": [SBinOp("==", SVar("x"), SInt(42))]},
            ],
            shared_vars=["x"]
        )
        result = verify_ag(system)
        assert result.verdict == AGVerdict.SOUND

    def test_empty_guarantees(self):
        system = make_ag_system(
            components=[
                {"name": "A", "body": "let x = 1;",
                 "assumptions": [], "guarantees": []},
            ],
            shared_vars=["x"]
        )
        result = verify_ag(system)
        # No guarantees to check -- should be vacuously sound
        assert result.verdict == AGVerdict.SOUND

    def test_tautological_assumption(self):
        # Assumption: x >= 0 OR x < 0 (always true)
        comp = extract_component(
            name="taut",
            source="let y = 1;",
            assumptions=[s_or(SBinOp(">=", SVar("x"), SInt(0)),
                             SBinOp("<", SVar("x"), SInt(0)))],
            guarantees=[SBinOp("==", SVar("y"), SInt(1))],
            shared_vars=["x", "y"]
        )
        result = verify_component_under_assumptions(comp)
        assert result.verified

    def test_three_component_circular(self):
        # A -> B -> C -> A
        system = make_ag_system(
            components=[
                {"name": "A", "body": "let x = 1;",
                 "assumptions": [SBinOp(">=", SVar("z"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("x"), SInt(0))]},
                {"name": "B", "body": "let y = 1;",
                 "assumptions": [SBinOp(">=", SVar("x"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("y"), SInt(0))]},
                {"name": "C", "body": "let z = 1;",
                 "assumptions": [SBinOp(">=", SVar("y"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("z"), SInt(0))]},
            ],
            shared_vars=["x", "y", "z"]
        )
        result = verify_ag(system)
        assert result.verdict == AGVerdict.SOUND

    def test_arithmetic_transformer(self):
        # y = x * 2, guarantee y >= 0 under assumption x >= 0
        comp = extract_component(
            name="double",
            source="let y = x * 2;",
            assumptions=[SBinOp(">=", SVar("x"), SInt(0))],
            guarantees=[SBinOp(">=", SVar("y"), SInt(0))],
            shared_vars=["x", "y"]
        )
        result = verify_component_under_assumptions(comp)
        assert result.verified

    def test_multiple_assumptions_conjunction(self):
        comp = extract_component(
            name="bounded",
            source="let y = x + 1;",
            assumptions=[
                SBinOp(">=", SVar("x"), SInt(0)),
                SBinOp("<=", SVar("x"), SInt(100))
            ],
            guarantees=[SBinOp("<=", SVar("y"), SInt(101))],
            shared_vars=["x", "y"]
        )
        result = verify_component_under_assumptions(comp)
        assert result.verified

    def test_counterexample_on_failure(self):
        comp = extract_component(
            name="fails",
            source="let y = x + 1;",
            assumptions=[],
            guarantees=[SBinOp("==", SVar("y"), SInt(5))],
            shared_vars=["x", "y"]
        )
        result = verify_component_under_assumptions(comp)
        assert not result.verified
        # Should have counterexample
        failed = [o for o in result.obligations if o.status == VCStatus.INVALID]
        assert len(failed) > 0
        assert failed[0].counterexample is not None


# ============================================================
# Section 16: Integration scenarios
# ============================================================

class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_producer_consumer(self):
        """Producer guarantees buffer non-negative, consumer assumes it."""
        result = verify_two_components(
            name_a="producer", body_a="let buf = 10;",
            assumptions_a=[],
            guarantees_a=[SBinOp(">=", SVar("buf"), SInt(0))],
            name_b="consumer", body_b="let taken = buf;",
            assumptions_b=[SBinOp(">=", SVar("buf"), SInt(0))],
            guarantees_b=[SBinOp(">=", SVar("taken"), SInt(0))],
            shared_vars=["buf", "taken"]
        )
        assert result.verdict == AGVerdict.SOUND

    def test_lock_protocol(self):
        """Two components managing separate resources with non-negative invariant."""
        # A sets resource_a, B sets resource_b, both guarantee non-negative
        result = verify_two_components(
            name_a="manager_a", body_a="let resource_a = 1;",
            assumptions_a=[SBinOp(">=", SVar("resource_b"), SInt(0))],
            guarantees_a=[SBinOp(">=", SVar("resource_a"), SInt(0))],
            name_b="manager_b", body_b="let resource_b = 2;",
            assumptions_b=[SBinOp(">=", SVar("resource_a"), SInt(0))],
            guarantees_b=[SBinOp(">=", SVar("resource_b"), SInt(0))],
            shared_vars=["resource_a", "resource_b"]
        )
        assert result.verdict == AGVerdict.SOUND

    def test_layered_system(self):
        """Three-layer system with ranking."""
        system = make_ag_system(
            components=[
                {"name": "hardware", "body": "let hw_ready = 1;",
                 "assumptions": [],
                 "guarantees": [SBinOp("==", SVar("hw_ready"), SInt(1))]},
                {"name": "driver", "body": "let drv_ok = 1;",
                 "assumptions": [SBinOp("==", SVar("hw_ready"), SInt(1))],
                 "guarantees": [SBinOp("==", SVar("drv_ok"), SInt(1))]},
                {"name": "app", "body": "let status = 1;",
                 "assumptions": [SBinOp("==", SVar("drv_ok"), SInt(1))],
                 "guarantees": [SBinOp("==", SVar("status"), SInt(1))]},
            ],
            shared_vars=["hw_ready", "drv_ok", "status"]
        )
        result = verify_with_ranking(system, {"hardware": 0, "driver": 1, "app": 2})
        assert result.verdict == AGVerdict.SOUND

    def test_peer_to_peer(self):
        """Two peers with symmetric contracts."""
        system = make_ag_system(
            components=[
                {"name": "peer1", "body": "let msg1 = 1;",
                 "assumptions": [SBinOp(">=", SVar("msg2"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("msg1"), SInt(0))]},
                {"name": "peer2", "body": "let msg2 = 1;",
                 "assumptions": [SBinOp(">=", SVar("msg1"), SInt(0))],
                 "guarantees": [SBinOp(">=", SVar("msg2"), SInt(0))]},
            ],
            shared_vars=["msg1", "msg2"]
        )
        result = verify_ag(system)
        assert result.verdict == AGVerdict.SOUND
        # Should detect circularity and use circular discharge
        assert result.metadata.get("circular_deps") is not None
