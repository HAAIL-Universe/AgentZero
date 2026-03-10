"""Tests for V045: Concurrent Effect Refinement"""

import os
import sys
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from concurrent_effect_refinement import (
    # Data structures
    CERVerdict, ThreadContract, ContractCheckResult,
    EffectRefinementConsistency, LockInvariant, InvariantCheckResult,
    CERResult,
    # Phase 1
    check_thread_refinements,
    # Phase 2
    check_effect_refinement_consistency, _extract_refined_vars,
    # Phase 3
    check_cross_thread_contract, check_all_contracts,
    # Phase 4
    check_lock_invariants,
    # Phase 5
    effect_aware_subtype,
    # Pipeline
    verify_concurrent_refinements, _determine_verdict,
    # Convenience
    verify_thread_pair, verify_with_lock_protocol,
    infer_thread_contract, verify_effect_subtyping,
)

# V011 imports for test construction
sys.path.insert(0, os.path.join(_dir, '..', 'V011_refinement_types'))
from refinement_types import (
    RefinedType, RefinedFuncType, CheckResult, RefinementError,
    refined_int, unrefined, nat_type, pos_type, range_type, eq_type,
    check_subtype, SubtypeResult,
)

# V004 SExpr imports
sys.path.insert(0, os.path.join(_dir, '..', 'V004_verification_conditions'))
from vc_gen import SVar, SInt, SBool, SBinOp, SAnd, s_and

# V040 effect imports
sys.path.insert(0, os.path.join(_dir, '..', 'V040_effect_systems'))
from effect_systems import Effect, EffectKind, EffectSet, FnEffectSig


# ===================================================================
# Helpers
# ===================================================================

def _ge(var, val):
    """SExpr: var >= val"""
    return SBinOp(">=", SVar(var), SInt(val))

def _le(var, val):
    """SExpr: var <= val"""
    return SBinOp("<=", SVar(var), SInt(val))

def _gt(var, val):
    """SExpr: var > val"""
    return SBinOp(">", SVar(var), SInt(val))

def _lt(var, val):
    """SExpr: var < val"""
    return SBinOp("<", SVar(var), SInt(val))

def _eq(var, val):
    """SExpr: var == val"""
    return SBinOp("==", SVar(var), SInt(val))

def _neq(var, val):
    """SExpr: var != val"""
    return SBinOp("!=", SVar(var), SInt(val))

def state_effect(var):
    return Effect(EffectKind.STATE, var)

def pure_effects():
    return EffectSet.pure()

def state_effects(*vars):
    return EffectSet(frozenset(state_effect(v) for v in vars))


# ===================================================================
# Phase 1: Per-Thread Refinement Checking
# ===================================================================

class TestPerThreadRefinement:
    """Test refinement type checking on individual threads."""

    def test_thread_with_valid_refinements(self):
        """Thread source that satisfies its refinement spec."""
        source = """
        fn identity(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x;
        }
        """
        contract = ThreadContract(
            thread_id="t1",
            source=source,
        )
        result = check_thread_refinements(contract)
        assert result.ok

    def test_thread_with_invalid_refinements(self):
        """Thread source that violates its refinement spec."""
        source = """
        fn bad(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x;
        }
        """
        # x=0 satisfies precondition but result=0 violates postcondition
        contract = ThreadContract(
            thread_id="t1",
            source=source,
        )
        result = check_thread_refinements(contract)
        assert not result.ok

    def test_thread_with_explicit_specs(self):
        """Thread with externally provided specs."""
        source = """
        fn inc(x) {
            return x + 1;
        }
        """
        specs = {
            "inc": RefinedFuncType(
                params=[("x", nat_type())],
                ret=pos_type(),
            ),
        }
        contract = ThreadContract(
            thread_id="t1",
            source=source,
            specs=specs,
        )
        result = check_thread_refinements(contract)
        assert result.ok

    def test_thread_with_failing_explicit_specs(self):
        """Thread fails to meet externally provided specs."""
        source = """
        fn bad(x) {
            return x;
        }
        """
        # x >= 0 but return type requires v > 0 (x=0 violates)
        specs = {
            "bad": RefinedFuncType(
                params=[("x", nat_type())],
                ret=pos_type(),
            ),
        }
        contract = ThreadContract(
            thread_id="t1",
            source=source,
            specs=specs,
        )
        result = check_thread_refinements(contract)
        assert not result.ok

    def test_thread_error_handling(self):
        """Bad source gracefully produces an error result."""
        contract = ThreadContract(
            thread_id="t1",
            source="this is not valid C10 at all @@@@",
        )
        result = check_thread_refinements(contract)
        assert not result.ok


# ===================================================================
# Phase 2: Effect-Refinement Consistency
# ===================================================================

class TestEffectRefinementConsistency:
    """Test that effects and refinement predicates are consistent."""

    def test_consistent_effects_and_refinements(self):
        """Shared vars have both effects and refinements."""
        contract = ThreadContract(
            thread_id="t1",
            source="let x = 1; x = x + 1;",
            shared_vars={"x"},
            post_shared={"x": _ge("x", 0)},
        )
        # Manually provide sigs matching the thread's behavior
        sigs = {
            "__main__": FnEffectSig(
                name="__main__",
                params=[],
                ret=None,
                effects=state_effects("x"),
                body_effects=state_effects("x"),
                handled=pure_effects(),
            ),
        }
        result = check_effect_refinement_consistency(contract, sigs)
        assert result.consistent

    def test_unconstrained_effected_var(self):
        """Shared var with State effect but no refinement constraint."""
        contract = ThreadContract(
            thread_id="t1",
            source="let x = 1; x = 2; let y = 1; y = 2;",
            shared_vars={"x", "y"},
            post_shared={"x": _ge("x", 0)},
        )
        sigs = {
            "__main__": FnEffectSig(
                name="__main__",
                params=[],
                ret=None,
                effects=state_effects("x", "y"),
                body_effects=state_effects("x", "y"),
                handled=pure_effects(),
            ),
        }
        result = check_effect_refinement_consistency(contract, sigs)
        assert "y" in result.unconstrained_vars

    def test_pure_thread_consistency(self):
        """Thread with no effects and no refinement predicates is consistent."""
        contract = ThreadContract(
            thread_id="t1",
            source="let x = 1;",
            shared_vars=set(),
        )
        sigs = {}
        result = check_effect_refinement_consistency(contract, sigs)
        assert result.consistent

    def test_extract_refined_vars(self):
        """Extract variable names from refinement specs."""
        specs = {
            "f": RefinedFuncType(
                params=[("x", nat_type("x")), ("y", refined_int(_ge("y", 0), "y"))],
                ret=nat_type("v"),
            ),
        }
        vars_found = _extract_refined_vars(specs)
        assert "x" in vars_found
        assert "y" in vars_found


# ===================================================================
# Phase 3: Cross-Thread Contract Verification
# ===================================================================

class TestCrossThreadContracts:
    """Test producer-consumer contract checking."""

    def test_valid_contract_ge_implies_ge(self):
        """Producer posts x>=5, consumer requires x>=0. Should hold."""
        producer = ThreadContract(
            thread_id="producer",
            source="let x = 5;",
            shared_vars={"x"},
            post_shared={"x": _ge("x", 5)},
        )
        consumer = ThreadContract(
            thread_id="consumer",
            source="let y = x;",
            shared_vars={"x"},
            pre_shared={"x": _ge("x", 0)},
        )
        result = check_cross_thread_contract(producer, consumer, "x")
        assert result.holds

    def test_invalid_contract(self):
        """Producer posts x>=0, consumer requires x>5. Should fail."""
        producer = ThreadContract(
            thread_id="producer",
            source="let x = 0;",
            shared_vars={"x"},
            post_shared={"x": _ge("x", 0)},
        )
        consumer = ThreadContract(
            thread_id="consumer",
            source="let y = x;",
            shared_vars={"x"},
            pre_shared={"x": _gt("x", 5)},
        )
        result = check_cross_thread_contract(producer, consumer, "x")
        assert not result.holds
        assert result.counterexample is not None

    def test_trivial_contract_no_post(self):
        """No postcondition means trivially holds."""
        producer = ThreadContract(
            thread_id="producer",
            source="let x = 1;",
            shared_vars={"x"},
        )
        consumer = ThreadContract(
            thread_id="consumer",
            source="let y = x;",
            shared_vars={"x"},
            pre_shared={"x": _ge("x", 0)},
        )
        result = check_cross_thread_contract(producer, consumer, "x")
        assert result.holds
        assert "trivially" in result.reason

    def test_trivial_contract_no_pre(self):
        """No precondition means trivially holds."""
        producer = ThreadContract(
            thread_id="producer",
            source="let x = 1;",
            shared_vars={"x"},
            post_shared={"x": _ge("x", 0)},
        )
        consumer = ThreadContract(
            thread_id="consumer",
            source="let y = x;",
            shared_vars={"x"},
        )
        result = check_cross_thread_contract(producer, consumer, "x")
        assert result.holds

    def test_eq_implies_range(self):
        """Producer posts x==10, consumer requires 0<=x<=100. Should hold."""
        producer = ThreadContract(
            thread_id="producer",
            source="let x = 10;",
            shared_vars={"x"},
            post_shared={"x": _eq("x", 10)},
        )
        consumer = ThreadContract(
            thread_id="consumer",
            source="let y = x;",
            shared_vars={"x"},
            pre_shared={"x": s_and(_ge("x", 0), _le("x", 100))},
        )
        result = check_cross_thread_contract(producer, consumer, "x")
        assert result.holds

    def test_check_all_contracts_multiple_vars(self):
        """Check contracts across multiple shared variables."""
        t1 = ThreadContract(
            thread_id="t1",
            source="let x = 5; let y = 10;",
            shared_vars={"x", "y"},
            post_shared={"x": _ge("x", 0), "y": _ge("y", 0)},
        )
        t2 = ThreadContract(
            thread_id="t2",
            source="let a = x; let b = y;",
            shared_vars={"x", "y"},
            pre_shared={"x": _ge("x", 0), "y": _ge("y", 0)},
        )
        results = check_all_contracts([t1, t2])
        # t1->t2 for x and y = 2 checks
        # t2->t1 has no post/pre overlap, so no checks
        assert len(results) == 2
        assert all(r.holds for r in results)

    def test_check_all_contracts_bidirectional(self):
        """Both threads have pre and post on shared vars."""
        t1 = ThreadContract(
            thread_id="t1",
            source="let x = 5;",
            shared_vars={"x"},
            pre_shared={"x": _ge("x", 0)},
            post_shared={"x": _ge("x", 1)},
        )
        t2 = ThreadContract(
            thread_id="t2",
            source="let x = 10;",
            shared_vars={"x"},
            pre_shared={"x": _ge("x", 1)},
            post_shared={"x": _ge("x", 0)},
        )
        results = check_all_contracts([t1, t2])
        # t1->t2: post(x>=1) => pre(x>=1) -- holds
        # t2->t1: post(x>=0) => pre(x>=0) -- holds
        assert len(results) == 2
        assert all(r.holds for r in results)

    def test_contract_violation_detected(self):
        """Contract violation between threads."""
        t1 = ThreadContract(
            thread_id="t1",
            source="let x = 0;",
            shared_vars={"x"},
            post_shared={"x": _ge("x", 0)},  # x >= 0
        )
        t2 = ThreadContract(
            thread_id="t2",
            source="let y = x;",
            shared_vars={"x"},
            pre_shared={"x": _gt("x", 0)},  # x > 0 -- not implied by x >= 0!
        )
        results = check_all_contracts([t1, t2])
        assert any(not r.holds for r in results)


# ===================================================================
# Phase 4: Lock-Protected Invariants
# ===================================================================

class TestLockInvariants:
    """Test lock-protected invariant checking."""

    def test_invariant_preserved(self):
        """Thread postcondition preserves lock invariant."""
        contract = ThreadContract(
            thread_id="t1",
            source="let x = 5;",
            shared_vars={"x"},
            locks={"mutex"},
            post_shared={"x": _ge("x", 0)},
        )
        inv = LockInvariant(
            lock_name="mutex",
            variable="x",
            predicate=_ge("x", 0),
        )
        results = check_lock_invariants([contract], [inv])
        assert len(results) == 1
        assert results[0].preserved

    def test_invariant_violated(self):
        """Thread postcondition does not preserve invariant."""
        contract = ThreadContract(
            thread_id="t1",
            source="let x = 0;",
            shared_vars={"x"},
            locks={"mutex"},
            post_shared={"x": _ge("x", 0)},  # x >= 0
        )
        inv = LockInvariant(
            lock_name="mutex",
            variable="x",
            predicate=_gt("x", 0),  # x > 0 -- not preserved by x >= 0!
        )
        results = check_lock_invariants([contract], [inv])
        assert len(results) == 1
        assert not results[0].preserved

    def test_invariant_trivially_preserved_no_modify(self):
        """Thread that doesn't modify the variable trivially preserves invariant."""
        contract = ThreadContract(
            thread_id="t1",
            source="let y = 5;",
            shared_vars={"y"},
            locks={"mutex"},
            # No post_shared for "x"
        )
        inv = LockInvariant(
            lock_name="mutex",
            variable="x",
            predicate=_ge("x", 0),
        )
        results = check_lock_invariants([contract], [inv])
        assert len(results) == 1
        assert results[0].preserved

    def test_thread_without_lock_skipped(self):
        """Thread that doesn't use the lock is skipped."""
        contract = ThreadContract(
            thread_id="t1",
            source="let x = 5;",
            shared_vars={"x"},
            locks={"other_lock"},
            post_shared={"x": _ge("x", 0)},
        )
        inv = LockInvariant(
            lock_name="mutex",
            variable="x",
            predicate=_gt("x", 0),
        )
        results = check_lock_invariants([contract], [inv])
        assert len(results) == 0

    def test_multiple_invariants(self):
        """Multiple lock invariants checked."""
        contract = ThreadContract(
            thread_id="t1",
            source="let x = 5; let y = 10;",
            shared_vars={"x", "y"},
            locks={"mx", "my"},
            post_shared={"x": _ge("x", 0), "y": _ge("y", 5)},
        )
        invs = [
            LockInvariant(lock_name="mx", variable="x", predicate=_ge("x", 0)),
            LockInvariant(lock_name="my", variable="y", predicate=_ge("y", 0)),
        ]
        results = check_lock_invariants([contract], invs)
        assert len(results) == 2
        assert all(r.preserved for r in results)


# ===================================================================
# Phase 5: Effect-Aware Subtype Checking
# ===================================================================

class TestEffectAwareSubtyping:
    """Test combined refinement + effect subtyping."""

    def test_subtype_both_hold(self):
        """Both refinement and effect subtyping hold."""
        sub = nat_type()
        sup = refined_int(_ge("v", 0))
        sub_eff = pure_effects()
        sup_eff = state_effects("x")
        result = effect_aware_subtype(sub, sup, sub_eff, sup_eff)
        assert result.is_subtype

    def test_refinement_fails(self):
        """Refinement subtyping fails."""
        sub = refined_int(_ge("v", 0))  # v >= 0
        sup = pos_type()                 # v > 0
        sub_eff = pure_effects()
        sup_eff = pure_effects()
        result = effect_aware_subtype(sub, sup, sub_eff, sup_eff)
        assert not result.is_subtype
        assert "Refinement" in result.reason

    def test_effect_fails(self):
        """Effect subtyping fails (sub has extra effects)."""
        sub = nat_type()
        sup = nat_type()
        sub_eff = state_effects("x", "y")
        sup_eff = state_effects("x")
        result = effect_aware_subtype(sub, sup, sub_eff, sup_eff)
        assert not result.is_subtype
        assert "Effect" in result.reason

    def test_pure_subtypes_effectful(self):
        """Pure is a subtype of any effect set."""
        sub = nat_type()
        sup = nat_type()
        sub_eff = pure_effects()
        sup_eff = state_effects("x")
        result = effect_aware_subtype(sub, sup, sub_eff, sup_eff)
        assert result.is_subtype


# ===================================================================
# Unified Pipeline
# ===================================================================

class TestUnifiedPipeline:
    """Test the full verification pipeline."""

    def test_safe_producer_consumer(self):
        """Safe producer-consumer pattern."""
        producer = ThreadContract(
            thread_id="producer",
            source="""
            fn produce(x) {
                requires(x >= 0);
                ensures(result >= 0);
                return x + 1;
            }
            """,
            shared_vars={"x"},
            post_shared={"x": _ge("x", 1)},
        )
        consumer = ThreadContract(
            thread_id="consumer",
            source="""
            fn consume(x) {
                requires(x >= 0);
                ensures(result >= 0);
                return x;
            }
            """,
            shared_vars={"x"},
            pre_shared={"x": _ge("x", 0)},
        )
        result = verify_concurrent_refinements([producer, consumer])
        assert result.is_safe
        assert result.verdict == CERVerdict.SAFE

    def test_unsafe_contract_violation(self):
        """Contract violation detected in pipeline."""
        t1 = ThreadContract(
            thread_id="t1",
            source="""
            fn f(x) {
                requires(x >= 0);
                ensures(result >= 0);
                return x;
            }
            """,
            shared_vars={"x"},
            post_shared={"x": _ge("x", 0)},
        )
        t2 = ThreadContract(
            thread_id="t2",
            source="""
            fn g(x) {
                requires(x > 5);
                ensures(result > 5);
                return x;
            }
            """,
            shared_vars={"x"},
            pre_shared={"x": _gt("x", 5)},
        )
        result = verify_concurrent_refinements([t1, t2])
        assert result.verdict == CERVerdict.CONTRACT_VIOLATION

    def test_refinement_error_in_thread(self):
        """Pipeline catches refinement error in one thread."""
        t1 = ThreadContract(
            thread_id="t1",
            source="""
            fn bad(x) {
                requires(x >= 0);
                ensures(result > 0);
                return x;
            }
            """,
            shared_vars=set(),
        )
        result = verify_concurrent_refinements([t1])
        assert result.verdict == CERVerdict.REFINEMENT_ERROR

    def test_pipeline_with_lock_invariants(self):
        """Pipeline checks lock-protected invariants."""
        t1 = ThreadContract(
            thread_id="t1",
            source="let x = 5;",
            shared_vars={"x"},
            locks={"mutex"},
            post_shared={"x": _ge("x", 1)},
        )
        inv = LockInvariant(lock_name="mutex", variable="x", predicate=_ge("x", 0))
        result = verify_concurrent_refinements([t1], lock_invariants=[inv])
        assert result.is_safe

    def test_pipeline_with_lock_invariant_violation(self):
        """Pipeline detects lock invariant violation."""
        t1 = ThreadContract(
            thread_id="t1",
            source="let x = 5;",
            shared_vars={"x"},
            locks={"mutex"},
            post_shared={"x": _ge("x", 0)},  # x >= 0
        )
        inv = LockInvariant(lock_name="mutex", variable="x", predicate=_gt("x", 0))
        result = verify_concurrent_refinements([t1], lock_invariants=[inv])
        assert result.verdict == CERVerdict.CONTRACT_VIOLATION

    def test_three_threads(self):
        """Three threads with chain contracts: t1->t2->t3."""
        t1 = ThreadContract(
            thread_id="t1",
            source="let x = 10;",
            shared_vars={"x"},
            post_shared={"x": _ge("x", 10)},
        )
        t2 = ThreadContract(
            thread_id="t2",
            source="let x = x + 1;",
            shared_vars={"x"},
            pre_shared={"x": _ge("x", 5)},
            post_shared={"x": _ge("x", 6)},
        )
        t3 = ThreadContract(
            thread_id="t3",
            source="let y = x;",
            shared_vars={"x"},
            pre_shared={"x": _ge("x", 0)},
        )
        result = verify_concurrent_refinements([t1, t2, t3])
        assert result.is_safe

    def test_summary_format(self):
        """CERResult summary is a readable string."""
        t1 = ThreadContract(
            thread_id="t1",
            source="let x = 1;",
            shared_vars=set(),
        )
        result = verify_concurrent_refinements([t1])
        assert "Verdict:" in result.summary
        assert "Refinement:" in result.summary


# ===================================================================
# Convenience APIs
# ===================================================================

class TestConvenienceAPIs:
    """Test high-level convenience functions."""

    def test_verify_thread_pair_safe(self):
        """verify_thread_pair with compatible contracts."""
        result = verify_thread_pair(
            producer_source="let x = 5;",
            consumer_source="let y = x;",
            shared_vars={"x"},
            producer_post={"x": _ge("x", 5)},
            consumer_pre={"x": _ge("x", 0)},
        )
        assert result.is_safe

    def test_verify_thread_pair_unsafe(self):
        """verify_thread_pair with incompatible contracts."""
        result = verify_thread_pair(
            producer_source="let x = 0;",
            consumer_source="let y = x;",
            shared_vars={"x"},
            producer_post={"x": _ge("x", 0)},
            consumer_pre={"x": _gt("x", 0)},
        )
        assert not result.is_safe

    def test_verify_with_lock_protocol(self):
        """verify_with_lock_protocol convenience API."""
        contracts = [
            ThreadContract(
                thread_id="t1",
                source="let x = 5;",
                shared_vars={"x"},
                locks={"m"},
                post_shared={"x": _ge("x", 1)},
            ),
        ]
        invs = [LockInvariant(lock_name="m", variable="x", predicate=_ge("x", 0))]
        result = verify_with_lock_protocol(contracts, invs)
        assert result.is_safe

    def test_infer_thread_contract(self):
        """Infer a ThreadContract from source."""
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result >= 0);
            return x + 1;
        }
        """
        contract = infer_thread_contract("t1", source, shared_vars={"x"})
        assert contract.thread_id == "t1"
        assert contract.specs is not None
        assert "inc" in contract.specs

    def test_infer_thread_contract_no_specs(self):
        """Infer contract from source without annotations."""
        source = "let x = 5;"
        contract = infer_thread_contract("t1", source)
        assert contract.thread_id == "t1"
        # No annotations -> no specs
        assert contract.specs is None or len(contract.specs) == 0

    def test_verify_effect_subtyping_api(self):
        """verify_effect_subtyping convenience wrapper."""
        result = verify_effect_subtyping(
            nat_type(), nat_type(),
            pure_effects(), state_effects("x"),
        )
        assert result.is_subtype


# ===================================================================
# Edge Cases and Integration
# ===================================================================

class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_empty_contract_list(self):
        """No threads to verify."""
        result = verify_concurrent_refinements([])
        assert result.is_safe

    def test_single_thread_no_shared(self):
        """Single thread with no shared variables."""
        t = ThreadContract(
            thread_id="t1",
            source="""
            fn abs(x) {
                requires(x >= 0);
                ensures(result >= 0);
                return x;
            }
            """,
            shared_vars=set(),
        )
        result = verify_concurrent_refinements([t])
        assert result.is_safe

    def test_contract_check_result_fields(self):
        """ContractCheckResult has all expected fields."""
        producer = ThreadContract(
            thread_id="p",
            source="let x = 1;",
            shared_vars={"x"},
            post_shared={"x": _eq("x", 10)},
        )
        consumer = ThreadContract(
            thread_id="c",
            source="let y = x;",
            shared_vars={"x"},
            pre_shared={"x": _ge("x", 0)},
        )
        result = check_cross_thread_contract(producer, consumer, "x")
        assert result.producer_id == "p"
        assert result.consumer_id == "c"
        assert result.variable == "x"
        assert result.holds
        assert isinstance(result.reason, str)

    def test_range_contract(self):
        """Contract using range predicate (conjunction)."""
        producer = ThreadContract(
            thread_id="p",
            source="let x = 50;",
            shared_vars={"x"},
            post_shared={"x": s_and(_ge("x", 10), _le("x", 90))},
        )
        consumer = ThreadContract(
            thread_id="c",
            source="let y = x;",
            shared_vars={"x"},
            pre_shared={"x": s_and(_ge("x", 0), _le("x", 100))},
        )
        result = check_cross_thread_contract(producer, consumer, "x")
        assert result.holds

    def test_range_contract_too_tight(self):
        """Consumer range is tighter than producer range."""
        producer = ThreadContract(
            thread_id="p",
            source="let x = 50;",
            shared_vars={"x"},
            post_shared={"x": s_and(_ge("x", 0), _le("x", 100))},
        )
        consumer = ThreadContract(
            thread_id="c",
            source="let y = x;",
            shared_vars={"x"},
            pre_shared={"x": s_and(_ge("x", 10), _le("x", 90))},
        )
        result = check_cross_thread_contract(producer, consumer, "x")
        assert not result.holds

    def test_determine_verdict_all_safe(self):
        """Verdict is SAFE when all checks pass."""
        verdict = _determine_verdict(
            thread_refinements={"t1": CheckResult([], 1, 1, {})},
            contract_checks=[],
            consistency_checks=[],
            inv_results=[],
            conc_result=None,
            errors=[],
        )
        assert verdict == CERVerdict.SAFE

    def test_determine_verdict_refinement_error(self):
        """Verdict is REFINEMENT_ERROR on refinement failures."""
        verdict = _determine_verdict(
            thread_refinements={
                "t1": CheckResult(
                    [RefinementError("bad")], 0, 1, {}
                )
            },
            contract_checks=[],
            consistency_checks=[],
            inv_results=[],
            conc_result=None,
            errors=[],
        )
        assert verdict == CERVerdict.REFINEMENT_ERROR

    def test_determine_verdict_unknown_on_errors(self):
        """Verdict is UNKNOWN when there are errors."""
        verdict = _determine_verdict(
            thread_refinements={},
            contract_checks=[],
            consistency_checks=[],
            inv_results=[],
            conc_result=None,
            errors=["something failed"],
        )
        assert verdict == CERVerdict.UNKNOWN

    def test_with_concurrency_check(self):
        """Pipeline runs V043 concurrency check when requested."""
        t1 = ThreadContract(
            thread_id="t1",
            source="let x = 5;",
            shared_vars={"x"},
            post_shared={"x": _ge("x", 0)},
        )
        result = verify_concurrent_refinements(
            [t1],
            run_concurrency_check=True,
        )
        # Should complete without error (no declared effects mismatch)
        assert result.verdict in (CERVerdict.SAFE, CERVerdict.UNKNOWN)
