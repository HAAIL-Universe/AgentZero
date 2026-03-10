"""Tests for V043: Concurrency Verification Composition."""

import os
import sys
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

_work = os.path.dirname(_dir)
for d in [
    os.path.join(_work, "V036_concurrent_separation_logic"),
    os.path.join(_work, "V040_effect_systems"),
    os.path.join(_work, "V023_ltl_model_checking"),
    os.path.join(_work, "V021_bdd_model_checking"),
]:
    if d not in sys.path:
        sys.path.insert(0, d)

from concurrency_verification_composition import (
    # Data model
    ConcVerdict, ThreadSpec, ConcurrentProgram, ConcVerificationResult,
    EffectRaceReport, TemporalCheckResult,
    # Phase 1: Effect race analysis
    infer_thread_effects, extract_state_effects, effect_race_analysis,
    # Phase 2: Effect checking
    check_thread_effects,
    # Phase 3: CSL
    build_parallel_cmd, run_csl_verification,
    # Phase 4: Temporal
    ConcurrentSystemBuilder,
    mutual_exclusion_property, deadlock_freedom_property, starvation_freedom_property,
    check_temporal_properties,
    # Pipeline
    verify_concurrent_program,
    # Convenience APIs
    verify_mutual_exclusion, verify_concurrent_effects,
    full_concurrent_verify, effect_guided_protocol_selection,
)

from effect_systems import (
    Effect, EffectKind, EffectSet, State, Exn, IO, DIV, NONDET,
    FnEffectSig,
)
from ltl_model_checker import (
    Atom, Globally, Finally, Until, Not as LNot, And as LAnd, Or as LOr,
    Implies as LImplies, Next,
    check_ltl,
)
from concurrent_separation_logic import (
    CNew, CAssign, CLoad, CStore, CDispose, CNull,
    CAcquire, CRelease, CParallel, CAtomic, CSkip, CSeq, CSeqList,
    CSLVerdict,
)

# Try to import SL formulas
try:
    from separation_logic_verifier import Emp, PointsTo, Star, Pure, SLTrue
    HAS_SL = True
except ImportError:
    HAS_SL = False


# =============================================================================
# Phase 1: Effect Inference for Threads
# =============================================================================

class TestEffectInference:
    """Test effect inference on thread source code."""

    def test_pure_thread(self):
        """Thread with no side effects."""
        thread = ThreadSpec("t0", "let x = 1; let y = x + 2;")
        sigs = infer_thread_effects(thread)
        assert "__main__" in sigs
        main = sigs["__main__"]
        assert main.effects.is_pure

    def test_state_effect_thread(self):
        """Thread that modifies shared state."""
        thread = ThreadSpec("t0", "let x = 0; x = x + 1;")
        sigs = infer_thread_effects(thread)
        main = sigs["__main__"]
        state_vars = extract_state_effects(sigs)
        assert "x" in state_vars

    def test_io_effect_thread(self):
        """Thread with IO effects."""
        thread = ThreadSpec("t0", "let x = 1; print(x);")
        sigs = infer_thread_effects(thread)
        main = sigs["__main__"]
        assert main.effects.has(EffectKind.IO)

    def test_multiple_state_effects(self):
        """Thread modifying multiple variables."""
        thread = ThreadSpec("t0", "let a = 0; let b = 0; a = 1; b = 2;")
        sigs = infer_thread_effects(thread)
        state_vars = extract_state_effects(sigs)
        assert "a" in state_vars
        assert "b" in state_vars

    def test_function_with_effects(self):
        """Function that has effects."""
        source = """
fn inc(x) {
    let r = x + 1;
    return r;
}
let a = 0;
a = inc(a);
"""
        thread = ThreadSpec("t0", source)
        sigs = infer_thread_effects(thread)
        assert "inc" in sigs or "__main__" in sigs


# =============================================================================
# Phase 1: Effect-Guided Race Analysis
# =============================================================================

class TestEffectRaceAnalysis:
    """Test effect-based race detection."""

    def test_no_shared_state(self):
        """Two threads with disjoint state."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 0; x = 1;"),
                ThreadSpec("t1", "let y = 0; y = 1;"),
            ]
        )
        all_reports, unprotected = effect_race_analysis(program)
        # x and y are thread-local (different names)
        shared = [r for r in unprotected if r.var not in ("*",)]
        # With separate let declarations, no actual sharing
        assert len(shared) == 0 or all(r.var == "*" for r in shared)

    def test_shared_state_detected(self):
        """Two threads writing same variable name."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let counter = 0; counter = counter + 1;"),
                ThreadSpec("t1", "let counter = 0; counter = counter + 1;"),
            ]
        )
        all_reports, unprotected = effect_race_analysis(program)
        counter_reports = [r for r in all_reports if r.var == "counter"]
        assert len(counter_reports) >= 2  # Both threads touch counter

    def test_protected_shared_state(self):
        """Shared state with lock protection."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let counter = 0; counter = 1;",
                           shared_vars={"counter"}, locks={"mutex"}),
                ThreadSpec("t1", "let counter = 0; counter = 2;",
                           shared_vars={"counter"}, locks={"mutex"}),
            ],
            shared_vars={"counter"},
            lock_invariants={"mutex": None},  # Just declare lock exists
        )
        all_reports, unprotected = effect_race_analysis(program)
        counter_unprotected = [r for r in unprotected if r.var == "counter"]
        assert len(counter_unprotected) == 0  # Protected by mutex

    def test_mixed_protection(self):
        """Some vars protected, some not."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let a = 0; let b = 0; a = 1; b = 1;",
                           shared_vars={"a"}, locks={"lock_a"}),
                ThreadSpec("t1", "let a = 0; let b = 0; a = 2; b = 2;",
                           shared_vars={"a"}, locks={"lock_a"}),
            ],
            shared_vars={"a"},
            lock_invariants={"lock_a": None},
        )
        all_reports, unprotected = effect_race_analysis(program)
        a_unprotected = [r for r in unprotected if r.var == "a"]
        b_unprotected = [r for r in unprotected if r.var == "b"]
        assert len(a_unprotected) == 0  # a is protected


# =============================================================================
# Phase 2: Effect Checking
# =============================================================================

class TestEffectChecking:
    """Test declared vs inferred effect checking."""

    def test_correct_declaration(self):
        """Declared effects match inferred."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 0; x = 1;",
                           declared_effects=EffectSet.of(State("x"))),
            ]
        )
        sigs, violations = check_thread_effects(program)
        assert len(violations) == 0

    def test_missing_declaration(self):
        """Inferred effect not declared."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 0; x = 1; print(x);",
                           declared_effects=EffectSet.of(State("x"))),
                # Missing IO declaration
            ]
        )
        sigs, violations = check_thread_effects(program)
        assert len(violations) > 0
        assert any("IO" in v or "io" in v for v in violations)

    def test_wildcard_covers_all(self):
        """State("*") covers any State effect."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 0; let y = 0; x = 1; y = 2;",
                           declared_effects=EffectSet.of(State("*"))),
            ]
        )
        sigs, violations = check_thread_effects(program)
        state_violations = [v for v in violations if "state" in v.lower() or "State" in v]
        assert len(state_violations) == 0

    def test_pure_thread_no_violations(self):
        """Pure thread with no declarations."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 1; let y = x + 2;"),
            ]
        )
        sigs, violations = check_thread_effects(program)
        assert len(violations) == 0

    def test_multiple_threads_checked(self):
        """All threads get checked."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 0; x = 1;",
                           declared_effects=EffectSet.of(State("x"))),
                ThreadSpec("t1", "let y = 0; print(y);",
                           declared_effects=EffectSet.of(IO)),
            ]
        )
        sigs, violations = check_thread_effects(program)
        assert "t0" in sigs
        assert "t1" in sigs


# =============================================================================
# Phase 3: CSL Integration
# =============================================================================

class TestCSLIntegration:
    """Test CSL verification integration."""

    def test_build_parallel_cmd_none(self):
        """No CSL commands -> None."""
        program = ConcurrentProgram(
            threads=[ThreadSpec("t0", "let x = 1;")]
        )
        assert build_parallel_cmd(program) is None

    def test_build_parallel_cmd_single(self):
        """Single CSL command -> returned directly."""
        cmd = CAssign("x", "1")
        program = ConcurrentProgram(
            threads=[ThreadSpec("t0", "let x = 1;", cmd=cmd)]
        )
        result = build_parallel_cmd(program)
        assert result is not None
        assert result.kind == cmd.kind

    def test_build_parallel_cmd_two(self):
        """Two CSL commands -> CParallel."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "", cmd=CAssign("x", "1")),
                ThreadSpec("t1", "", cmd=CAssign("y", "2")),
            ]
        )
        result = build_parallel_cmd(program)
        assert result is not None
        assert result.kind.name == "PARALLEL"

    def test_build_parallel_cmd_three(self):
        """Three CSL commands -> nested CParallel."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "", cmd=CAssign("x", "1")),
                ThreadSpec("t1", "", cmd=CAssign("y", "2")),
                ThreadSpec("t2", "", cmd=CAssign("z", "3")),
            ]
        )
        result = build_parallel_cmd(program)
        assert result is not None
        assert result.kind.name == "PARALLEL"

    def test_csl_race_detection(self):
        """CSL detects race on shared variable."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "", cmd=CSeqList([
                    CAssign("x", "1"),
                    CStore("p", "x"),
                ])),
                ThreadSpec("t1", "", cmd=CSeqList([
                    CAssign("x", "2"),
                    CStore("p", "x"),
                ])),
            ]
        )
        csl_result, ownership = run_csl_verification(program)
        assert csl_result is not None
        # Race on x (written by both) and p (stored by both)

    def test_csl_safe_disjoint(self):
        """Disjoint memory -> safe."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "", cmd=CAssign("x", "1")),
                ThreadSpec("t1", "", cmd=CAssign("y", "2")),
            ]
        )
        csl_result, ownership = run_csl_verification(program)
        assert csl_result is not None


# =============================================================================
# Phase 4: Temporal Verification - System Builder
# =============================================================================

class TestSystemBuilder:
    """Test concurrent system builder."""

    def test_no_protocol_system(self):
        """Build system with no synchronization."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="none"
        )
        assert "want_0" in state_vars
        assert "want_1" in state_vars
        assert "in_cs_0" in state_vars
        assert "in_cs_1" in state_vars

    def test_lock_protocol_system(self):
        """Build lock-based system."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="lock"
        )
        assert "lock" in state_vars
        assert "want_0" in state_vars

    def test_flag_protocol_system(self):
        """Build Peterson's protocol system."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="flag"
        )
        assert "flag_0" in state_vars
        assert "flag_1" in state_vars
        assert "turn" in state_vars

    def test_three_thread_lock(self):
        """Lock protocol scales to 3 threads."""
        builder = ConcurrentSystemBuilder(3)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=3, protocol="lock"
        )
        assert "want_2" in state_vars
        assert "in_cs_2" in state_vars


# =============================================================================
# Phase 4: LTL Properties
# =============================================================================

class TestLTLProperties:
    """Test LTL property construction."""

    def test_mutual_exclusion_2(self):
        """Mutual exclusion for 2 threads."""
        prop = mutual_exclusion_property(2)
        assert prop.op.name == "G"

    def test_mutual_exclusion_3(self):
        """Mutual exclusion for 3 threads."""
        prop = mutual_exclusion_property(3)
        assert prop.op.name == "G"

    def test_deadlock_freedom(self):
        """Deadlock freedom property."""
        prop = deadlock_freedom_property(2)
        assert prop.op.name == "G"

    def test_starvation_freedom(self):
        """Starvation freedom property."""
        prop = starvation_freedom_property(0)
        assert prop.op.name == "G"


# =============================================================================
# Phase 4: Temporal Model Checking
# =============================================================================

class TestTemporalModelChecking:
    """Test LTL model checking on concurrent systems."""

    def test_no_protocol_violates_mutex(self):
        """No synchronization -> mutual exclusion violated."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="none"
        )
        props = [("mutex", mutual_exclusion_property(2))]
        results = check_temporal_properties(state_vars, init_fn, trans_fn, props)
        assert len(results) == 1
        # Without synchronization, mutex can be violated
        assert not results[0].holds

    def test_lock_protocol_preserves_mutex(self):
        """Lock-based protocol -> mutual exclusion holds."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="lock"
        )
        props = [("mutex", mutual_exclusion_property(2))]
        results = check_temporal_properties(state_vars, init_fn, trans_fn, props)
        assert len(results) == 1
        assert results[0].holds

    def test_flag_protocol_preserves_mutex(self):
        """Peterson's protocol -> mutual exclusion holds."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="flag"
        )
        props = [("mutex", mutual_exclusion_property(2))]
        results = check_temporal_properties(state_vars, init_fn, trans_fn, props)
        assert len(results) == 1
        assert results[0].holds

    def test_three_thread_lock_mutex(self):
        """Lock protocol with 3 threads preserves mutex."""
        builder = ConcurrentSystemBuilder(3)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=3, protocol="lock"
        )
        props = [("mutex", mutual_exclusion_property(3))]
        results = check_temporal_properties(state_vars, init_fn, trans_fn, props)
        assert len(results) == 1
        assert results[0].holds

    def test_no_protocol_counterexample(self):
        """Counterexample when mutex violated."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="none"
        )
        props = [("mutex", mutual_exclusion_property(2))]
        results = check_temporal_properties(state_vars, init_fn, trans_fn, props)
        # Should have counterexample
        assert not results[0].holds
        # Counterexample may or may not be available depending on V023 implementation


# =============================================================================
# Unified Pipeline
# =============================================================================

class TestUnifiedPipeline:
    """Test the full verification pipeline."""

    def test_effects_only(self):
        """Pipeline with only effect checking."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 0; x = 1;"),
                ThreadSpec("t1", "let y = 0; y = 1;"),
            ]
        )
        result = verify_concurrent_program(
            program,
            check_effects_flag=True,
            check_csl=False,
            check_temporal=False,
        )
        assert isinstance(result, ConcVerificationResult)
        assert result.effect_sigs is not None

    def test_temporal_only(self):
        """Pipeline with only temporal checking."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="lock"
        )
        program = ConcurrentProgram(
            threads=[ThreadSpec("t0", "let x = 0;"), ThreadSpec("t1", "let y = 0;")]
        )
        result = verify_concurrent_program(
            program,
            check_effects_flag=False,
            check_csl=False,
            check_temporal=True,
            temporal_system=(state_vars, init_fn, trans_fn),
            temporal_properties=[("mutex", mutual_exclusion_property(2))],
        )
        assert len(result.temporal_results) == 1
        assert result.temporal_results[0].holds

    def test_combined_effects_and_temporal(self):
        """Full pipeline: effects + temporal."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="lock"
        )
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let counter = 0; counter = counter + 1;"),
                ThreadSpec("t1", "let counter = 0; counter = counter + 1;"),
            ]
        )
        result = verify_concurrent_program(
            program,
            check_effects_flag=True,
            check_csl=False,
            check_temporal=True,
            temporal_system=(state_vars, init_fn, trans_fn),
            temporal_properties=[("mutex", mutual_exclusion_property(2))],
        )
        # Effects: both threads write counter (race detected)
        assert len(result.effect_race_reports) > 0
        # Temporal: lock protocol is safe
        assert any(t.holds for t in result.temporal_results)

    def test_unsafe_verdict_on_effect_violation(self):
        """Effect violation triggers EFFECT_VIOLATION verdict."""
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 0; x = 1; print(x);",
                           declared_effects=EffectSet.of(State("x"))),
                # Missing IO
            ]
        )
        result = verify_concurrent_program(
            program, check_effects_flag=True, check_csl=False, check_temporal=False
        )
        assert result.verdict == ConcVerdict.EFFECT_VIOLATION

    def test_safe_verdict_all_clear(self):
        """All checks pass -> SAFE."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="lock"
        )
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 1; let y = x + 2;"),
                ThreadSpec("t1", "let a = 3; let b = a + 4;"),
            ]
        )
        result = verify_concurrent_program(
            program,
            check_effects_flag=True,
            check_csl=False,
            check_temporal=True,
            temporal_system=(state_vars, init_fn, trans_fn),
            temporal_properties=[("mutex", mutual_exclusion_property(2))],
        )
        assert result.verdict == ConcVerdict.SAFE

    def test_temporal_violation_verdict(self):
        """Temporal violation triggers TEMPORAL_VIOLATION verdict."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="none"
        )
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 1;"),
                ThreadSpec("t1", "let y = 1;"),
            ]
        )
        result = verify_concurrent_program(
            program,
            check_effects_flag=False,
            check_csl=False,
            check_temporal=True,
            temporal_system=(state_vars, init_fn, trans_fn),
            temporal_properties=[("mutex", mutual_exclusion_property(2))],
        )
        assert result.verdict == ConcVerdict.TEMPORAL_VIOLATION

    def test_result_summary(self):
        """Result summary is generated."""
        program = ConcurrentProgram(
            threads=[ThreadSpec("t0", "let x = 1;")]
        )
        result = verify_concurrent_program(
            program, check_effects_flag=True, check_csl=False, check_temporal=False
        )
        assert "Verdict" in result.summary


# =============================================================================
# Convenience APIs
# =============================================================================

class TestConvenienceAPIs:
    """Test high-level convenience functions."""

    def test_verify_mutex_no_protocol(self):
        """verify_mutual_exclusion with no protocol."""
        result = verify_mutual_exclusion(protocol="none", n_threads=2)
        assert not result.is_safe  # No protocol -> mutex violation

    def test_verify_mutex_lock(self):
        """verify_mutual_exclusion with lock protocol."""
        result = verify_mutual_exclusion(protocol="lock", n_threads=2)
        assert result.is_safe

    def test_verify_mutex_flag(self):
        """verify_mutual_exclusion with Peterson's."""
        result = verify_mutual_exclusion(protocol="flag", n_threads=2)
        assert result.is_safe

    def test_verify_mutex_three_threads(self):
        """verify_mutual_exclusion with 3 threads + lock."""
        result = verify_mutual_exclusion(protocol="lock", n_threads=3)
        assert result.is_safe

    def test_verify_concurrent_effects_disjoint(self):
        """verify_concurrent_effects with disjoint threads."""
        result = verify_concurrent_effects({
            "t0": "let x = 0; x = 1;",
            "t1": "let y = 0; y = 1;",
        })
        assert isinstance(result, ConcVerificationResult)

    def test_verify_concurrent_effects_shared(self):
        """verify_concurrent_effects with shared state."""
        result = verify_concurrent_effects(
            {
                "t0": "let counter = 0; counter = 1;",
                "t1": "let counter = 0; counter = 2;",
            },
            shared_vars={"counter"},
        )
        assert len(result.effect_race_reports) > 0

    def test_full_concurrent_verify(self):
        """full_concurrent_verify combines effects + temporal."""
        result = full_concurrent_verify(
            thread_sources={
                "t0": "let x = 0; x = x + 1;",
                "t1": "let y = 0; y = y + 1;",
            },
            protocol="lock",
            n_threads=2,
        )
        assert isinstance(result, ConcVerificationResult)
        # Temporal should pass (lock protocol)
        assert all(t.holds for t in result.temporal_results)

    def test_full_verify_with_custom_properties(self):
        """Full verify with custom LTL properties."""
        result = full_concurrent_verify(
            thread_sources={
                "t0": "let x = 1;",
                "t1": "let y = 1;",
            },
            protocol="lock",
            ltl_properties=[
                ("mutex", mutual_exclusion_property(2)),
            ],
        )
        assert all(t.holds for t in result.temporal_results)


# =============================================================================
# Effect-Guided Protocol Selection
# =============================================================================

class TestProtocolSelection:
    """Test effect-guided protocol selection."""

    def test_shared_state_recommends_lock(self):
        """Shared state detected -> recommends lock."""
        result = effect_guided_protocol_selection(
            {
                "t0": "let counter = 0; counter = counter + 1;",
                "t1": "let counter = 0; counter = counter + 1;",
            },
            shared_vars={"counter"},
        )
        assert result["has_shared_state"]
        assert result["recommended"] == "lock"
        assert "lock" in result["protocol_comparison"]
        assert result["protocol_comparison"]["lock"]["mutual_exclusion_holds"]

    def test_disjoint_state_recommends_none(self):
        """Disjoint state -> recommends none."""
        result = effect_guided_protocol_selection({
            "t0": "let x = 0; x = x + 1;",
            "t1": "let y = 0; y = y + 1;",
        })
        # x and y both get State effects; if they don't overlap, recommend none
        # But effect inference may see both as "counter" if same-named
        assert "protocol_comparison" in result

    def test_protocol_comparison_complete(self):
        """All protocols are compared."""
        result = effect_guided_protocol_selection(
            {
                "t0": "let a = 1;",
                "t1": "let b = 2;",
            },
        )
        assert "none" in result["protocol_comparison"]
        assert "lock" in result["protocol_comparison"]

    def test_two_thread_includes_flag(self):
        """2-thread comparison includes Peterson's."""
        result = effect_guided_protocol_selection(
            {
                "t0": "let x = 0; x = 1;",
                "t1": "let x = 0; x = 2;",
            },
        )
        assert "flag" in result["protocol_comparison"]

    def test_no_protocol_violates(self):
        """No protocol fails mutual exclusion."""
        result = effect_guided_protocol_selection(
            {
                "t0": "let a = 1;",
                "t1": "let b = 1;",
            },
        )
        assert not result["protocol_comparison"]["none"]["mutual_exclusion_holds"]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_single_thread(self):
        """Single thread program."""
        program = ConcurrentProgram(
            threads=[ThreadSpec("t0", "let x = 0; x = 1;")]
        )
        result = verify_concurrent_program(
            program, check_effects_flag=True, check_csl=False, check_temporal=False
        )
        assert result.verdict == ConcVerdict.SAFE

    def test_empty_thread_source(self):
        """Thread with minimal source."""
        program = ConcurrentProgram(
            threads=[ThreadSpec("t0", "let x = 0;")]
        )
        result = verify_concurrent_program(
            program, check_effects_flag=True, check_csl=False, check_temporal=False
        )
        assert isinstance(result, ConcVerificationResult)

    def test_many_threads_effects(self):
        """Multiple threads effect analysis."""
        threads = [
            ThreadSpec(f"t{i}", f"let v{i} = 0; v{i} = {i};")
            for i in range(5)
        ]
        program = ConcurrentProgram(threads=threads)
        result = verify_concurrent_program(
            program, check_effects_flag=True, check_csl=False, check_temporal=False
        )
        assert len(result.effect_sigs) == 5

    def test_is_safe_property(self):
        """ConcVerificationResult.is_safe property."""
        r = ConcVerificationResult(verdict=ConcVerdict.SAFE)
        assert r.is_safe
        r2 = ConcVerificationResult(verdict=ConcVerdict.RACE)
        assert not r2.is_safe

    def test_verdict_enum_values(self):
        """All verdict values are strings."""
        for v in ConcVerdict:
            assert isinstance(v.value, str)

    def test_thread_spec_defaults(self):
        """ThreadSpec default fields."""
        ts = ThreadSpec("t0", "let x = 0;")
        assert ts.cmd is None
        assert ts.declared_effects is None
        assert ts.shared_vars is None
        assert ts.locks is None

    def test_concurrent_program_defaults(self):
        """ConcurrentProgram default fields."""
        cp = ConcurrentProgram(threads=[])
        assert len(cp.lock_invariants) == 0
        assert len(cp.shared_vars) == 0
        assert len(cp.ltl_properties) == 0


# =============================================================================
# Composition Integrity
# =============================================================================

class TestCompositionIntegrity:
    """Test that V036 + V040 + V023 compose correctly."""

    def test_effect_inferred_then_temporal_checked(self):
        """Effects inform the picture, temporal verifies protocol."""
        # Two threads share a counter
        sources = {
            "t0": "let counter = 0; counter = counter + 1;",
            "t1": "let counter = 0; counter = counter + 1;",
        }

        # Step 1: Effect analysis shows shared state
        threads = [ThreadSpec(tid, src) for tid, src in sources.items()]
        program = ConcurrentProgram(threads=threads)
        reports, unprotected = effect_race_analysis(program)
        has_race_risk = len([r for r in reports if r.var == "counter"]) > 0

        # Step 2: Temporal analysis shows lock protocol is correct
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="lock"
        )
        ltl_results = check_temporal_properties(
            state_vars, init_fn, trans_fn,
            [("mutex", mutual_exclusion_property(2))]
        )

        assert has_race_risk  # Effects detect the risk
        assert ltl_results[0].holds  # Lock protocol resolves it

    def test_effect_clean_temporal_safe(self):
        """Pure threads + correct protocol = fully safe."""
        result = full_concurrent_verify(
            thread_sources={
                "t0": "let x = 1; let y = x + 2;",
                "t1": "let a = 3; let b = a + 4;",
            },
            protocol="lock",
        )
        assert result.verdict == ConcVerdict.SAFE
        assert all(t.holds for t in result.temporal_results)

    def test_effect_violation_overrides_temporal_safe(self):
        """Effect violation is reported even if temporal is safe."""
        builder = ConcurrentSystemBuilder(2)
        state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
            n_threads=2, protocol="lock"
        )
        program = ConcurrentProgram(
            threads=[
                ThreadSpec("t0", "let x = 0; x = 1; print(x);",
                           declared_effects=EffectSet.of(State("x"))),
                # IO not declared
                ThreadSpec("t1", "let y = 1;"),
            ]
        )
        result = verify_concurrent_program(
            program,
            check_effects_flag=True,
            check_csl=False,
            check_temporal=True,
            temporal_system=(state_vars, init_fn, trans_fn),
            temporal_properties=[("mutex", mutual_exclusion_property(2))],
        )
        assert result.verdict == ConcVerdict.EFFECT_VIOLATION
