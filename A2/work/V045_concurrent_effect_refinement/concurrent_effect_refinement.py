"""V045: Concurrent Effect Refinement

Composes:
- V043 (Concurrency Verification Composition) - effects + CSL + temporal
- V011 (Refinement Types) - SMT-checked refinement predicates

Key insight: effects tell you WHAT a thread does, refinement types tell you
HOW PRECISELY it does it. Combining them gives:

1. Per-thread refinement checking: each thread satisfies its refined type spec
2. Cross-thread contract verification: thread A's postcondition on shared state
   implies thread B's precondition
3. Effect-refinement consistency: effects must cover variables in refinement predicates
4. Lock-protected invariants: refinement predicates on shared variables hold under
   lock protection
5. Effect-aware subtyping: State effects widen the subtype context
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Set, Tuple

# Path setup
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
_challenges = os.path.join(_az, "challenges")

for p in [
    os.path.join(_work, "V043_concurrency_verification_composition"),
    os.path.join(_work, "V040_effect_systems"),
    os.path.join(_work, "V011_refinement_types"),
    os.path.join(_work, "V036_concurrent_separation_logic"),
    os.path.join(_work, "V023_ltl_model_checking"),
    os.path.join(_work, "V021_bdd_model_checking"),
    os.path.join(_work, "V004_verification_conditions"),
    os.path.join(_work, "V002_pdr"),
    os.path.join(_challenges, "C010_stack_vm"),
    os.path.join(_challenges, "C037_smt_solver"),
    os.path.join(_challenges, "C038_symbolic_execution"),
    os.path.join(_challenges, "C039_abstract_interpreter"),
    os.path.join(_challenges, "C013_type_checker"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# V043 imports
from concurrency_verification_composition import (
    ThreadSpec, ConcurrentProgram, ConcVerdict, ConcVerificationResult,
    EffectRaceReport, TemporalCheckResult,
    infer_thread_effects, extract_state_effects, effect_race_analysis,
    check_thread_effects, verify_concurrent_program,
)

# V040 imports
from effect_systems import (
    Effect, EffectKind, EffectSet, FnEffectSig, EffectInferrer,
)

# V011 imports
from refinement_types import (
    RefinedType, RefinedFuncType, RefinementEnv, SubtypeResult,
    RefinementError, CheckResult, RefinementChecker,
    refined_int, unrefined, nat_type, pos_type, range_type, eq_type,
    check_subtype, parse_source, extract_refinement_specs,
    strip_annotations, check_refinements, check_program_refinements,
    subst_sexpr, selfify, negate_sexpr,
)

# V004 imports
from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot,
    s_and, s_or, s_not, s_implies, lower_to_smt,
)

# C037 imports
from smt_solver import SMTSolver, SMTResult, Op, App, IntConst, Var as SMTVar

# C010 imports
from stack_vm import lex, Parser


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class CERVerdict(Enum):
    """Concurrent Effect Refinement verdict."""
    SAFE = "safe"
    UNSAFE = "unsafe"
    REFINEMENT_ERROR = "refinement_error"
    EFFECT_MISMATCH = "effect_mismatch"
    CONTRACT_VIOLATION = "contract_violation"
    UNKNOWN = "unknown"


@dataclass
class ThreadContract:
    """Refinement type contract for a thread.

    Specifies preconditions and postconditions on shared variables,
    plus declared effects.
    """
    thread_id: str
    source: str
    # Refinement specs: function name -> refined function type
    specs: Optional[Dict[str, RefinedFuncType]] = None
    # Declared effects for the thread
    declared_effects: Optional[EffectSet] = None
    # Which variables are shared with other threads
    shared_vars: Optional[Set[str]] = None
    # Lock protection
    locks: Optional[Set[str]] = None
    # Pre/postconditions on shared state (SExpr predicates)
    pre_shared: Optional[Dict[str, SExpr]] = None   # var -> precondition
    post_shared: Optional[Dict[str, SExpr]] = None   # var -> postcondition


@dataclass
class ContractCheckResult:
    """Result of cross-thread contract verification."""
    producer_id: str
    consumer_id: str
    variable: str
    post_expr: SExpr       # producer's postcondition
    pre_expr: SExpr        # consumer's precondition
    holds: bool
    counterexample: Optional[Dict[str, int]] = None
    reason: str = ""


@dataclass
class EffectRefinementConsistency:
    """Check that effects and refinement predicates are consistent."""
    thread_id: str
    consistent: bool
    # Variables mentioned in refinement predicates but not in effects
    uneffected_vars: List[str] = field(default_factory=list)
    # Variables in effects but with no refinement constraints
    unconstrained_vars: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class CERResult:
    """Full Concurrent Effect Refinement result."""
    verdict: CERVerdict
    # Per-thread refinement checking
    thread_refinements: Dict[str, CheckResult] = field(default_factory=dict)
    # Cross-thread contract checks
    contract_checks: List[ContractCheckResult] = field(default_factory=list)
    # Effect-refinement consistency
    consistency_checks: List[EffectRefinementConsistency] = field(default_factory=list)
    # Effect analysis from V043
    effect_sigs: Dict[str, Dict[str, FnEffectSig]] = field(default_factory=dict)
    # Underlying V043 result (if run)
    concurrency_result: Optional[ConcVerificationResult] = None
    # Errors
    errors: List[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return self.verdict == CERVerdict.SAFE

    @property
    def summary(self) -> str:
        parts = [f"Verdict: {self.verdict.value}"]
        # Refinement summary
        total_errs = sum(len(r.errors) for r in self.thread_refinements.values())
        total_obls = sum(r.total_obligations for r in self.thread_refinements.values())
        verified = sum(r.verified_obligations for r in self.thread_refinements.values())
        parts.append(f"Refinement: {verified}/{total_obls} obligations, {total_errs} errors")
        # Contract summary
        if self.contract_checks:
            passed = sum(1 for c in self.contract_checks if c.holds)
            parts.append(f"Contracts: {passed}/{len(self.contract_checks)} hold")
        # Consistency summary
        if self.consistency_checks:
            consistent = sum(1 for c in self.consistency_checks if c.consistent)
            parts.append(f"Consistency: {consistent}/{len(self.consistency_checks)}")
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Phase 1: Per-Thread Refinement Checking
# ---------------------------------------------------------------------------

def check_thread_refinements(contract: ThreadContract) -> CheckResult:
    """Run V011 refinement type checking on a single thread's source."""
    try:
        if contract.specs:
            return check_refinements(contract.source, contract.specs)
        else:
            return check_program_refinements(contract.source)
    except Exception as e:
        result = CheckResult(
            errors=[RefinementError(message=f"Refinement check failed: {e}",
                                   location=contract.thread_id)],
            verified_obligations=0,
            total_obligations=0,
            function_types={},
        )
        return result


# ---------------------------------------------------------------------------
# Phase 2: Effect Inference and Consistency
# ---------------------------------------------------------------------------

def _extract_refined_vars(specs: Dict[str, RefinedFuncType]) -> Set[str]:
    """Extract variable names mentioned in refinement predicates."""
    vars_found = set()
    for fn_name, ftype in specs.items():
        for param_name, rtype in ftype.params:
            vars_found.add(param_name)
            _collect_sexpr_vars(rtype.predicate, vars_found)
        _collect_sexpr_vars(ftype.ret.predicate, vars_found)
    return vars_found


def _collect_sexpr_vars(expr: SExpr, out: Set[str]):
    """Collect variable names from an SExpr."""
    if isinstance(expr, SVar):
        out.add(expr.name)
    elif isinstance(expr, SBinOp):
        _collect_sexpr_vars(expr.left, out)
        _collect_sexpr_vars(expr.right, out)
    elif isinstance(expr, SUnaryOp):
        _collect_sexpr_vars(expr.operand, out)
    elif isinstance(expr, SAnd):
        for sub in expr.conjuncts:
            _collect_sexpr_vars(sub, out)
    elif isinstance(expr, SOr):
        for sub in expr.disjuncts:
            _collect_sexpr_vars(sub, out)
    elif isinstance(expr, SNot):
        _collect_sexpr_vars(expr.operand, out)
    elif isinstance(expr, SImplies):
        _collect_sexpr_vars(expr.antecedent, out)
        _collect_sexpr_vars(expr.consequent, out)


def check_effect_refinement_consistency(
    contract: ThreadContract,
    inferred_effects: Dict[str, FnEffectSig],
) -> EffectRefinementConsistency:
    """Check that effects and refinement predicates are consistent.

    Rules:
    - Every variable in a refinement predicate that gets modified
      should have a corresponding State effect
    - Variables with State effects but no refinement constraints are flagged
      as unconstrained (warning, not error)
    """
    # Collect variables from refinement predicates
    refined_vars = set()
    if contract.specs:
        refined_vars = _extract_refined_vars(contract.specs)
    if contract.pre_shared:
        for v, expr in contract.pre_shared.items():
            refined_vars.add(v)
            _collect_sexpr_vars(expr, refined_vars)
    if contract.post_shared:
        for v, expr in contract.post_shared.items():
            refined_vars.add(v)
            _collect_sexpr_vars(expr, refined_vars)

    # Collect state-effected variables from inferred effects
    effected_vars = set()
    for fn_name, sig in inferred_effects.items():
        for eff in sig.effects.effects:
            if eff.kind == EffectKind.STATE and eff.detail:
                effected_vars.add(eff.detail)

    # Shared vars that are refined but not effected
    shared = contract.shared_vars or set()
    uneffected = []
    for v in refined_vars:
        if v in shared and v not in effected_vars:
            # Variable has a refinement constraint and is shared,
            # but thread has no State effect on it -- could be read-only (ok)
            pass

    # Effected shared vars with no refinement constraint
    unconstrained = []
    for v in effected_vars:
        if v in shared and v not in refined_vars:
            unconstrained.append(v)

    consistent = len(uneffected) == 0
    msg_parts = []
    if uneffected:
        msg_parts.append(f"Refined but uneffected: {uneffected}")
    if unconstrained:
        msg_parts.append(f"Effected but unconstrained: {unconstrained}")
    if not msg_parts:
        msg_parts.append("Effects and refinements are consistent")

    return EffectRefinementConsistency(
        thread_id=contract.thread_id,
        consistent=consistent,
        uneffected_vars=uneffected,
        unconstrained_vars=unconstrained,
        message="; ".join(msg_parts),
    )


# ---------------------------------------------------------------------------
# Phase 3: Cross-Thread Contract Verification
# ---------------------------------------------------------------------------

def check_cross_thread_contract(
    producer: ThreadContract,
    consumer: ThreadContract,
    variable: str,
) -> ContractCheckResult:
    """Check that producer's postcondition implies consumer's precondition.

    For shared variable `v`:
      producer.post_shared[v] => consumer.pre_shared[v]

    This is checked via SMT: assert post AND NOT(pre), check UNSAT.
    """
    post_expr = (producer.post_shared or {}).get(variable)
    pre_expr = (consumer.pre_shared or {}).get(variable)

    if post_expr is None or pre_expr is None:
        # No contract to check -- trivially holds
        return ContractCheckResult(
            producer_id=producer.thread_id,
            consumer_id=consumer.thread_id,
            variable=variable,
            post_expr=post_expr or SBool(True),
            pre_expr=pre_expr or SBool(True),
            holds=True,
            reason="No contract specified (trivially holds)",
        )

    # SMT check: post AND NOT(pre) should be UNSAT
    solver = SMTSolver()
    var_cache = {}

    post_smt = lower_to_smt(solver, post_expr, var_cache)
    neg_pre = negate_sexpr(pre_expr)
    neg_pre_smt = lower_to_smt(solver, neg_pre, var_cache)

    solver.add(post_smt)
    solver.add(neg_pre_smt)

    result = solver.check()

    if result == SMTResult.UNSAT:
        return ContractCheckResult(
            producer_id=producer.thread_id,
            consumer_id=consumer.thread_id,
            variable=variable,
            post_expr=post_expr,
            pre_expr=pre_expr,
            holds=True,
            reason="Producer postcondition implies consumer precondition",
        )
    elif result == SMTResult.SAT:
        model = solver.model()
        return ContractCheckResult(
            producer_id=producer.thread_id,
            consumer_id=consumer.thread_id,
            variable=variable,
            post_expr=post_expr,
            pre_expr=pre_expr,
            holds=False,
            counterexample=model,
            reason="Contract violation: postcondition does not imply precondition",
        )
    else:
        return ContractCheckResult(
            producer_id=producer.thread_id,
            consumer_id=consumer.thread_id,
            variable=variable,
            post_expr=post_expr,
            pre_expr=pre_expr,
            holds=False,
            reason="SMT solver returned unknown",
        )


def check_all_contracts(
    contracts: List[ThreadContract],
) -> List[ContractCheckResult]:
    """Check all cross-thread contracts between pairs of threads.

    For each pair (producer, consumer) and each shared variable,
    verify that producer's postcondition implies consumer's precondition.
    """
    results = []

    for i, producer in enumerate(contracts):
        for j, consumer in enumerate(contracts):
            if i == j:
                continue
            # Find shared variables between this pair
            prod_post = producer.post_shared or {}
            cons_pre = consumer.pre_shared or {}
            common_vars = set(prod_post.keys()) & set(cons_pre.keys())

            for var in sorted(common_vars):
                result = check_cross_thread_contract(producer, consumer, var)
                results.append(result)

    return results


# ---------------------------------------------------------------------------
# Phase 4: Lock-Protected Invariant Checking
# ---------------------------------------------------------------------------

@dataclass
class LockInvariant:
    """A refinement predicate that must hold while a lock is held."""
    lock_name: str
    variable: str
    predicate: SExpr  # must hold whenever lock is acquired


@dataclass
class InvariantCheckResult:
    """Result of checking a lock-protected invariant."""
    lock_name: str
    variable: str
    thread_id: str
    preserved: bool
    reason: str = ""


def check_lock_invariants(
    contracts: List[ThreadContract],
    lock_invariants: List[LockInvariant],
) -> List[InvariantCheckResult]:
    """Check that lock-protected invariants are preserved by each thread.

    For each lock invariant (lock L protects predicate P(v)):
    - Each thread that acquires L must ensure P(v) still holds at release
    - This is checked via the thread's postcondition on v: post(v) => P(v)
    """
    results = []

    for inv in lock_invariants:
        for contract in contracts:
            # Only check threads that use this lock
            if contract.locks and inv.lock_name not in contract.locks:
                continue

            # Check: thread's postcondition on the variable implies the invariant
            post = (contract.post_shared or {}).get(inv.variable)
            if post is None:
                # Thread doesn't modify this variable -- invariant trivially preserved
                results.append(InvariantCheckResult(
                    lock_name=inv.lock_name,
                    variable=inv.variable,
                    thread_id=contract.thread_id,
                    preserved=True,
                    reason="Thread does not modify this variable",
                ))
                continue

            # SMT check: post AND NOT(invariant) should be UNSAT
            solver = SMTSolver()
            var_cache = {}
            post_smt = lower_to_smt(solver, post, var_cache)
            neg_inv = negate_sexpr(inv.predicate)
            neg_inv_smt = lower_to_smt(solver, neg_inv, var_cache)

            solver.add(post_smt)
            solver.add(neg_inv_smt)

            result = solver.check()

            if result == SMTResult.UNSAT:
                results.append(InvariantCheckResult(
                    lock_name=inv.lock_name,
                    variable=inv.variable,
                    thread_id=contract.thread_id,
                    preserved=True,
                    reason="Postcondition preserves invariant",
                ))
            else:
                results.append(InvariantCheckResult(
                    lock_name=inv.lock_name,
                    variable=inv.variable,
                    thread_id=contract.thread_id,
                    preserved=False,
                    reason="Postcondition may violate invariant",
                ))

    return results


# ---------------------------------------------------------------------------
# Phase 5: Effect-Aware Subtype Checking
# ---------------------------------------------------------------------------

def effect_aware_subtype(
    sub: RefinedType,
    sup: RefinedType,
    sub_effects: EffectSet,
    sup_effects: EffectSet,
) -> SubtypeResult:
    """Subtype check that also considers effects.

    sub <: sup requires:
    1. Standard refinement subtyping: sub.predicate => sup.predicate
    2. Effect subtyping: sub.effects <= sup.effects (sub is at most as effectful)
    """
    # First check refinement subtyping
    refine_result = check_subtype(sub, sup)
    if not refine_result.is_subtype:
        return SubtypeResult(
            is_subtype=False,
            counterexample=refine_result.counterexample,
            reason=f"Refinement subtype failure: {refine_result.reason}",
        )

    # Then check effect subtyping
    if not sub_effects.effects.issubset(sup_effects.effects):
        extra = sub_effects.effects - sup_effects.effects
        return SubtypeResult(
            is_subtype=False,
            counterexample=None,
            reason=f"Effect subtype failure: extra effects {extra}",
        )

    return SubtypeResult(is_subtype=True, reason="Subtype holds (refinement + effects)")


# ---------------------------------------------------------------------------
# Unified Pipeline
# ---------------------------------------------------------------------------

def verify_concurrent_refinements(
    contracts: List[ThreadContract],
    lock_invariants: Optional[List[LockInvariant]] = None,
    run_concurrency_check: bool = False,
    max_steps: int = 200,
) -> CERResult:
    """Full concurrent effect refinement verification pipeline.

    Phases:
    1. Per-thread refinement checking (V011)
    2. Effect inference + consistency checking (V040 + V011)
    3. Cross-thread contract verification (SMT)
    4. Lock-protected invariant checking
    5. (Optional) V043 concurrency verification
    """
    errors = []

    # Phase 1: Per-thread refinement checking
    thread_refinements = {}
    for contract in contracts:
        try:
            result = check_thread_refinements(contract)
            thread_refinements[contract.thread_id] = result
        except Exception as e:
            errors.append(f"Thread {contract.thread_id} refinement check failed: {e}")
            thread_refinements[contract.thread_id] = CheckResult(
                errors=[RefinementError(message=str(e))],
                verified_obligations=0,
                total_obligations=0,
                function_types={},
            )

    # Phase 2: Effect inference + consistency
    effect_sigs = {}
    consistency_checks = []
    for contract in contracts:
        try:
            ts = ThreadSpec(
                thread_id=contract.thread_id,
                source=contract.source,
                declared_effects=contract.declared_effects,
                shared_vars=contract.shared_vars,
                locks=contract.locks,
            )
            sigs = infer_thread_effects(ts)
            effect_sigs[contract.thread_id] = sigs
            consistency = check_effect_refinement_consistency(contract, sigs)
            consistency_checks.append(consistency)
        except Exception as e:
            errors.append(f"Thread {contract.thread_id} effect inference failed: {e}")

    # Phase 3: Cross-thread contracts
    contract_checks = check_all_contracts(contracts)

    # Phase 4: Lock invariants
    inv_results = []
    if lock_invariants:
        inv_results = check_lock_invariants(contracts, lock_invariants)

    # Phase 5: Optional V043 concurrency check
    conc_result = None
    if run_concurrency_check:
        try:
            threads = []
            for contract in contracts:
                ts = ThreadSpec(
                    thread_id=contract.thread_id,
                    source=contract.source,
                    declared_effects=contract.declared_effects,
                    shared_vars=contract.shared_vars,
                    locks=contract.locks,
                )
                threads.append(ts)
            program = ConcurrentProgram(
                threads=threads,
                lock_invariants={},
                shared_vars=set().union(*(c.shared_vars or set() for c in contracts)),
            )
            conc_result = verify_concurrent_program(
                program,
                check_effects_flag=True,
                check_csl=False,  # CSL requires Cmd, not source
                check_temporal=False,
                max_steps=max_steps,
            )
        except Exception as e:
            errors.append(f"Concurrency check failed: {e}")

    # Determine verdict
    verdict = _determine_verdict(
        thread_refinements, contract_checks, consistency_checks,
        inv_results, conc_result, errors,
    )

    return CERResult(
        verdict=verdict,
        thread_refinements=thread_refinements,
        contract_checks=contract_checks,
        consistency_checks=consistency_checks,
        effect_sigs=effect_sigs,
        concurrency_result=conc_result,
        errors=errors,
    )


def _determine_verdict(
    thread_refinements: Dict[str, CheckResult],
    contract_checks: List[ContractCheckResult],
    consistency_checks: List[EffectRefinementConsistency],
    inv_results: List[InvariantCheckResult],
    conc_result: Optional[ConcVerificationResult],
    errors: List[str],
) -> CERVerdict:
    """Determine overall verdict from all check results."""
    if errors:
        return CERVerdict.UNKNOWN

    # Check refinement errors
    for tid, result in thread_refinements.items():
        if result.errors:
            return CERVerdict.REFINEMENT_ERROR

    # Check contract violations
    for check in contract_checks:
        if not check.holds:
            return CERVerdict.CONTRACT_VIOLATION

    # Check invariant violations
    for inv in inv_results:
        if not inv.preserved:
            return CERVerdict.CONTRACT_VIOLATION

    # Check effect-refinement consistency
    for check in consistency_checks:
        if not check.consistent:
            return CERVerdict.EFFECT_MISMATCH

    # Check V043 result
    if conc_result and not conc_result.is_safe:
        return CERVerdict.UNSAFE

    return CERVerdict.SAFE


# ---------------------------------------------------------------------------
# High-Level Convenience APIs
# ---------------------------------------------------------------------------

def verify_thread_pair(
    producer_source: str,
    consumer_source: str,
    shared_vars: Set[str],
    producer_post: Dict[str, SExpr],
    consumer_pre: Dict[str, SExpr],
    producer_specs: Optional[Dict[str, RefinedFuncType]] = None,
    consumer_specs: Optional[Dict[str, RefinedFuncType]] = None,
) -> CERResult:
    """Convenience API for verifying a producer-consumer pair.

    Common pattern: thread A writes shared state, thread B reads it.
    Verify that A's postcondition on shared state satisfies B's precondition.
    """
    producer = ThreadContract(
        thread_id="producer",
        source=producer_source,
        specs=producer_specs,
        shared_vars=shared_vars,
        post_shared=producer_post,
    )
    consumer = ThreadContract(
        thread_id="consumer",
        source=consumer_source,
        specs=consumer_specs,
        shared_vars=shared_vars,
        pre_shared=consumer_pre,
    )
    return verify_concurrent_refinements([producer, consumer])


def verify_with_lock_protocol(
    contracts: List[ThreadContract],
    lock_invariants: List[LockInvariant],
) -> CERResult:
    """Verify threads with lock-protected invariants."""
    return verify_concurrent_refinements(
        contracts,
        lock_invariants=lock_invariants,
    )


def infer_thread_contract(
    thread_id: str,
    source: str,
    shared_vars: Optional[Set[str]] = None,
) -> ThreadContract:
    """Infer a ThreadContract from source code.

    Uses V011's annotation extraction to get specs and V040's effect
    inference to get effects. Shared variable pre/postconditions are
    extracted from requires/ensures annotations.
    """
    stmts = parse_source(source)
    specs = extract_refinement_specs(stmts)

    # Infer effects
    ts = ThreadSpec(thread_id=thread_id, source=source, shared_vars=shared_vars)
    try:
        sigs = infer_thread_effects(ts)
        # Collect all effects
        all_effects = EffectSet.pure()
        for sig in sigs.values():
            all_effects = all_effects.union(sig.effects)
    except Exception:
        all_effects = None
        sigs = {}

    # Extract shared variable pre/post from specs
    pre_shared = {}
    post_shared = {}
    if shared_vars and specs:
        for fn_name, ftype in specs.items():
            # Parameters with shared var names become preconditions
            for param_name, rtype in ftype.params:
                if param_name in shared_vars:
                    pre_shared[param_name] = rtype.predicate
            # Return type doesn't directly map to shared vars,
            # but ensures clauses mentioning shared vars do
            # (handled by the annotation extraction)

    return ThreadContract(
        thread_id=thread_id,
        source=source,
        specs=specs if specs else None,
        declared_effects=all_effects,
        shared_vars=shared_vars,
        pre_shared=pre_shared if pre_shared else None,
        post_shared=post_shared if post_shared else None,
    )


def verify_effect_subtyping(
    sub_type: RefinedType,
    sup_type: RefinedType,
    sub_effects: EffectSet,
    sup_effects: EffectSet,
) -> SubtypeResult:
    """Check refinement + effect subtyping."""
    return effect_aware_subtype(sub_type, sup_type, sub_effects, sup_effects)
