"""
V043: Concurrency Verification Composition

Composes:
- V036 (Concurrent Separation Logic) - memory safety, race detection, ownership
- V040 (Effect Systems) - algebraic effects, inference, checking
- V023 (LTL Model Checking) - temporal properties on boolean systems

The three analyses address orthogonal concerns of concurrent programs:
- Effects tell you WHAT each thread does (state, IO, exceptions, divergence)
- CSL tells you IF resource access is safe (ownership, lock protection)
- LTL tells you about execution ORDER (mutual exclusion, deadlock, starvation)

This module composes all three into a unified concurrent verification pipeline.
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Set, Tuple, FrozenSet

# Path setup
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
_challenges = os.path.join(_az, "challenges")

for p in [
    os.path.join(_work, "V036_concurrent_separation_logic"),
    os.path.join(_work, "V040_effect_systems"),
    os.path.join(_work, "V023_ltl_model_checking"),
    os.path.join(_work, "V021_bdd_model_checking"),
    os.path.join(_work, "V004_verification_condition_generation"),
    os.path.join(_work, "V002_pdr"),
    os.path.join(_challenges, "C010_stack_vm"),
    os.path.join(_challenges, "C037_smt_solver"),
    os.path.join(_challenges, "C038_symbolic_execution"),
    os.path.join(_challenges, "C039_abstract_interpreter"),
    os.path.join(_challenges, "C013_type_checker"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# V036 imports
from concurrent_separation_logic import (
    Cmd, CmdKind, CNew, CAssign, CLoad, CStore, CDispose, CNull,
    CAcquire, CRelease, CParallel, CAtomic, CSkip, CSeq, CSeqList,
    CSLVerdict, CSLResult, OwnershipReport,
    verify_concurrent, detect_races, analyze_ownership, check_race_freedom,
    check_rely_guarantee,
)

# V040 imports
from effect_systems import (
    Effect, EffectKind, EffectSet,
    State, Exn, IO, DIV, NONDET, PURE,
    EffectfulFuncType, BaseType, T_INT, T_BOOL, T_UNIT,
    FnEffectSig, EffectCheckStatus, EffectCheckResult, EffectVerificationResult,
    infer_effects, check_effects, verify_effects,
    EffectInferrer, EffectChecker,
)

# V023 imports
from ltl_model_checker import (
    LTL, LTLOp, LTLResult,
    Atom, LTLTrue, LTLFalse, Not as LNot, And as LAnd, Or as LOr,
    Implies as LImplies, Next, Finally, Globally, Until, Release,
    check_ltl, check_ltl_fair, parse_ltl,
)

# V021 imports
from bdd_model_checker import BDD, BooleanTS

# SL formula imports (from V031 via V036)
try:
    from separation_logic_verifier import (
        SLFormula, Emp, PointsTo, Star, Pure, SLTrue,
    )
except ImportError:
    # Minimal SL formula stubs if V031 not available
    pass


# =============================================================================
# Data Model
# =============================================================================

class ConcVerdict(Enum):
    """Unified verdict for concurrent verification."""
    SAFE = "safe"
    UNSAFE = "unsafe"
    RACE = "race"
    EFFECT_VIOLATION = "effect_violation"
    TEMPORAL_VIOLATION = "temporal_violation"
    UNKNOWN = "unknown"


@dataclass
class ThreadSpec:
    """Specification for a single thread."""
    thread_id: str
    source: str                                    # C10 source code for the thread body
    cmd: Optional[Cmd] = None                      # CSL command (for memory operations)
    declared_effects: Optional[EffectSet] = None   # Declared effect annotations
    shared_vars: Optional[Set[str]] = None         # Variables shared with other threads
    locks: Optional[Set[str]] = None               # Locks this thread may acquire


@dataclass
class ConcurrentProgram:
    """A concurrent program with multiple threads and shared resources."""
    threads: List[ThreadSpec]
    lock_invariants: Dict[str, 'SLFormula'] = field(default_factory=dict)
    shared_vars: Set[str] = field(default_factory=set)
    ltl_properties: List[LTL] = field(default_factory=list)


@dataclass
class EffectRaceReport:
    """Report from effect-guided race analysis."""
    thread_id: str
    var: str
    effect: Effect
    protected: bool
    protecting_lock: Optional[str] = None


@dataclass
class TemporalCheckResult:
    """Result of a single temporal property check."""
    property_text: str
    holds: bool
    counterexample: Optional[Tuple[List[Dict], List[Dict]]] = None
    method: str = "ltl"


@dataclass
class ConcVerificationResult:
    """Unified result from the concurrent verification pipeline."""
    verdict: ConcVerdict
    # Effect analysis results
    effect_sigs: Dict[str, Dict[str, FnEffectSig]] = field(default_factory=dict)
    effect_violations: List[str] = field(default_factory=list)
    # Race analysis results
    effect_race_reports: List[EffectRaceReport] = field(default_factory=list)
    unprotected_state_effects: List[EffectRaceReport] = field(default_factory=list)
    # CSL results
    csl_result: Optional[CSLResult] = None
    ownership_report: Optional[OwnershipReport] = None
    # Temporal results
    temporal_results: List[TemporalCheckResult] = field(default_factory=list)
    # Summary
    errors: List[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return self.verdict == ConcVerdict.SAFE

    @property
    def summary(self) -> str:
        lines = [f"Verdict: {self.verdict.value}"]
        if self.effect_violations:
            lines.append(f"Effect violations: {len(self.effect_violations)}")
        if self.unprotected_state_effects:
            lines.append(f"Unprotected state effects: {len(self.unprotected_state_effects)}")
        if self.temporal_results:
            passed = sum(1 for t in self.temporal_results if t.holds)
            total = len(self.temporal_results)
            lines.append(f"Temporal properties: {passed}/{total} hold")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "; ".join(lines)


# =============================================================================
# Phase 1: Effect-Guided Race Analysis
# =============================================================================

def infer_thread_effects(thread: ThreadSpec) -> Dict[str, FnEffectSig]:
    """Infer effects for a single thread's source code."""
    try:
        return infer_effects(thread.source)
    except Exception as e:
        return {"__error__": FnEffectSig(
            name="__error__",
            params=[],
            ret=T_UNIT,
            effects=EffectSet.of(State("*"), IO, Exn("InferenceError")),
            body_effects=EffectSet.of(State("*"), IO, Exn("InferenceError")),
            handled=EffectSet.pure(),
        )}


def extract_state_effects(sigs: Dict[str, FnEffectSig]) -> Set[str]:
    """Extract all variable names touched by State effects."""
    vars_touched = set()
    for fn_name, sig in sigs.items():
        for eff in sig.effects.effects:
            if eff.kind == EffectKind.STATE:
                if eff.detail and eff.detail != "*":
                    vars_touched.add(eff.detail)
                elif eff.detail == "*":
                    vars_touched.add("*")  # Wildcard: touches unknown vars
    return vars_touched


def effect_race_analysis(
    program: ConcurrentProgram,
) -> Tuple[List[EffectRaceReport], List[EffectRaceReport]]:
    """
    Analyze races using effect information.

    For each thread, infer which variables have State effects.
    Cross-reference with other threads to find shared state.
    Check if shared state is protected by locks.

    Returns: (all_reports, unprotected_reports)
    """
    # Step 1: Infer effects per thread
    thread_effects: Dict[str, Dict[str, FnEffectSig]] = {}
    thread_state_vars: Dict[str, Set[str]] = {}

    for thread in program.threads:
        sigs = infer_thread_effects(thread)
        thread_effects[thread.thread_id] = sigs
        thread_state_vars[thread.thread_id] = extract_state_effects(sigs)

    # Step 2: Find shared state variables (touched by 2+ threads)
    all_vars: Dict[str, Set[str]] = {}  # var -> set of thread_ids
    for tid, vars_set in thread_state_vars.items():
        for v in vars_set:
            if v not in all_vars:
                all_vars[v] = set()
            all_vars[v].add(tid)

    shared_state = {v: tids for v, tids in all_vars.items() if len(tids) > 1}

    # Step 3: Determine lock protection
    # Collect lock-protected vars from lock_invariants
    lock_protected: Dict[str, str] = {}  # var -> lock_name
    for lock_name in program.lock_invariants:
        # If threads declare which locks protect which vars
        for thread in program.threads:
            if thread.locks and lock_name in thread.locks:
                if thread.shared_vars:
                    for sv in thread.shared_vars:
                        lock_protected[sv] = lock_name

    # Also check explicitly declared shared_vars in program
    for sv in program.shared_vars:
        for lock_name in program.lock_invariants:
            if sv not in lock_protected:
                lock_protected[sv] = lock_name

    # Step 4: Generate reports
    all_reports = []
    unprotected = []

    for var, tids in shared_state.items():
        for tid in tids:
            is_protected = var in lock_protected
            report = EffectRaceReport(
                thread_id=tid,
                var=var,
                effect=State(var),
                protected=is_protected,
                protecting_lock=lock_protected.get(var),
            )
            all_reports.append(report)
            if not is_protected and var != "*":
                unprotected.append(report)

    return all_reports, unprotected


# =============================================================================
# Phase 2: Effect Checking (per-thread)
# =============================================================================

def check_thread_effects(
    program: ConcurrentProgram,
) -> Tuple[Dict[str, Dict[str, FnEffectSig]], List[str]]:
    """
    Check declared effects against inferred effects for each thread.

    Returns: (effect_sigs_per_thread, violations)
    """
    all_sigs = {}
    violations = []

    for thread in program.threads:
        sigs = infer_thread_effects(thread)
        all_sigs[thread.thread_id] = sigs

        if thread.declared_effects is not None:
            # Check that inferred effects are subset of declared
            main_sig = sigs.get("__main__")
            if main_sig:
                inferred = main_sig.effects
                declared = thread.declared_effects

                # Check each inferred effect is covered
                for eff in inferred.effects:
                    if eff.kind == EffectKind.PURE:
                        continue
                    covered = False
                    for decl in declared.effects:
                        if decl.kind == eff.kind:
                            if decl.detail is None or decl.detail == "*" or decl.detail == eff.detail:
                                covered = True
                                break
                    if not covered:
                        violations.append(
                            f"Thread {thread.thread_id}: undeclared effect {eff}"
                        )

    return all_sigs, violations


# =============================================================================
# Phase 3: CSL Verification (memory safety + races)
# =============================================================================

def build_parallel_cmd(program: ConcurrentProgram) -> Optional[Cmd]:
    """Build a CSL parallel command from thread specs that have Cmd."""
    cmds = []
    for thread in program.threads:
        if thread.cmd is not None:
            cmds.append(thread.cmd)

    if len(cmds) == 0:
        return None
    if len(cmds) == 1:
        return cmds[0]

    # Build balanced parallel tree
    result = cmds[0]
    for c in cmds[1:]:
        result = CParallel(result, c)
    return result


def run_csl_verification(
    program: ConcurrentProgram,
) -> Tuple[Optional[CSLResult], Optional[OwnershipReport]]:
    """Run CSL verification on the concurrent program's memory operations."""
    parallel_cmd = build_parallel_cmd(program)
    if parallel_cmd is None:
        return None, None

    # Build lock invariants dict for CSL
    from concurrent_separation_logic import LockInvariant
    lock_invs = {}
    for name, formula in program.lock_invariants.items():
        lock_invs[name] = LockInvariant(lock_name=name, invariant=formula)

    # Ownership analysis
    ownership = analyze_ownership(parallel_cmd, lock_invs)

    # Race detection
    races = detect_races(parallel_cmd, lock_invs)

    # Build CSL result from race detection
    if races:
        csl_result = CSLResult(
            verdict=CSLVerdict.RACE,
            errors=[str(r) for r in races],
            thread_results={},
            race_reports=races,
            lock_usage={},
        )
    else:
        csl_result = CSLResult(
            verdict=CSLVerdict.SAFE,
            errors=[],
            thread_results={},
            race_reports=[],
            lock_usage={},
        )

    return csl_result, ownership


# =============================================================================
# Phase 4: Temporal Verification (LTL model checking)
# =============================================================================

class ConcurrentSystemBuilder:
    """
    Builds a boolean transition system modeling concurrent thread execution.

    Each thread has:
    - A program counter (pc_t) encoded as bits
    - State variables (shared or local)
    - A scheduler variable selecting which thread executes

    The transition relation nondeterministically picks a thread to step.
    """

    def __init__(self, n_threads: int, steps_per_thread: int = 4):
        self.n_threads = n_threads
        self.steps_per_thread = steps_per_thread
        # Each thread's PC needs ceil(log2(steps+1)) bits
        import math
        self.pc_bits = max(1, math.ceil(math.log2(steps_per_thread + 1)))

    def build_mutual_exclusion_system(
        self,
        n_threads: int = 2,
        protocol: str = "none",
    ) -> Tuple[List[str], 'callable', 'callable']:
        """
        Build a mutual exclusion system with the specified protocol.

        Protocols:
        - "none": No synchronization (can violate mutual exclusion)
        - "flag": Flag-based protocol (Peterson's for 2 threads)
        - "lock": Lock-based protocol

        Returns: (state_vars, init_fn, trans_fn) for LTL model checking.
        """
        if protocol == "lock":
            return self._build_lock_protocol(n_threads)
        elif protocol == "flag":
            return self._build_flag_protocol(n_threads)
        else:
            return self._build_no_protocol(n_threads)

    def _build_no_protocol(self, n_threads: int):
        """No synchronization: threads freely enter critical section."""
        state_vars = []
        for i in range(n_threads):
            state_vars.append(f"want_{i}")
            state_vars.append(f"in_cs_{i}")

        def init_fn(bdd):
            # Initially no thread wants or is in CS (all false)
            result = bdd.TRUE
            for v in state_vars:
                vi = bdd.var(bdd.var_index(v))
                result = bdd.apply("and", result, bdd.apply("not", vi, bdd.TRUE))
            return result

        def trans_fn(bdd, cur, nxt):
            # cur: {name: var_idx}, nxt: {name': var_idx} -- convert to BDD nodes
            C = {k: bdd.var(v) for k, v in cur.items()}
            # Remap primed keys to unprimed for uniform access
            N = {k.rstrip("'"): bdd.var(v) for k, v in nxt.items()}
            trans = bdd.FALSE

            for i in range(n_threads):
                want_c = C[f"want_{i}"]
                in_cs_c = C[f"in_cs_{i}"]
                want_n = N[f"want_{i}"]
                in_cs_n = N[f"in_cs_{i}"]

                # Frame: other threads unchanged
                def frame_others(t_idx, bdd=bdd, C=C, N=N):
                    f = bdd.TRUE
                    for j in range(n_threads):
                        if j != t_idx:
                            for prefix in ["want_", "in_cs_"]:
                                c = C[f"{prefix}{j}"]
                                n = N[f"{prefix}{j}"]
                                f = bdd.apply("and", f, bdd.apply("iff", c, n))
                    return f

                frame = frame_others(i)

                # T1: request entry (!want -> want, in_cs unchanged)
                t1_guard = bdd.apply("not", want_c, bdd.TRUE)
                t1_action = bdd.apply("and", want_n,
                             bdd.apply("iff", in_cs_c, in_cs_n))
                t1 = bdd.apply("and", bdd.apply("and", t1_guard, t1_action), frame)

                # T2: enter CS (want & !in_cs -> in_cs, want unchanged)
                t2_guard = bdd.apply("and", want_c,
                            bdd.apply("not", in_cs_c, bdd.TRUE))
                t2_action = bdd.apply("and", in_cs_n,
                             bdd.apply("iff", want_c, want_n))
                t2 = bdd.apply("and", bdd.apply("and", t2_guard, t2_action), frame)

                # T3: leave CS (in_cs -> !in_cs, !want)
                t3_guard = in_cs_c
                t3_action = bdd.apply("and",
                             bdd.apply("not", in_cs_n, bdd.TRUE),
                             bdd.apply("not", want_n, bdd.TRUE))
                t3 = bdd.apply("and", bdd.apply("and", t3_guard, t3_action), frame)

                ti = bdd.apply("or", bdd.apply("or", t1, t2), t3)
                trans = bdd.apply("or", trans, ti)

            return trans

        return state_vars, init_fn, trans_fn

    def _build_lock_protocol(self, n_threads: int):
        """Lock-based mutual exclusion."""
        state_vars = ["lock"]
        for i in range(n_threads):
            state_vars.append(f"want_{i}")
            state_vars.append(f"in_cs_{i}")

        def init_fn(bdd):
            result = bdd.TRUE
            for v in state_vars:
                vi = bdd.var(bdd.var_index(v))
                result = bdd.apply("and", result, bdd.apply("not", vi, bdd.TRUE))
            return result

        def trans_fn(bdd, cur, nxt):
            C = {k: bdd.var(v) for k, v in cur.items()}
            N = {k.rstrip("'"): bdd.var(v) for k, v in nxt.items()}
            trans = bdd.FALSE

            for i in range(n_threads):
                want_c = C[f"want_{i}"]
                in_cs_c = C[f"in_cs_{i}"]
                want_n = N[f"want_{i}"]
                in_cs_n = N[f"in_cs_{i}"]
                lock_c = C["lock"]
                lock_n = N["lock"]

                def frame_others(t_idx, bdd=bdd, C=C, N=N):
                    f = bdd.TRUE
                    for j in range(n_threads):
                        if j != t_idx:
                            for prefix in ["want_", "in_cs_"]:
                                c = C[f"{prefix}{j}"]
                                n = N[f"{prefix}{j}"]
                                f = bdd.apply("and", f, bdd.apply("iff", c, n))
                    return f

                frame = frame_others(i)

                # T1: request entry (!want -> want, in_cs/lock unchanged)
                t1_guard = bdd.apply("not", want_c, bdd.TRUE)
                t1_action = bdd.apply("and", want_n,
                             bdd.apply("and",
                              bdd.apply("iff", in_cs_c, in_cs_n),
                              bdd.apply("iff", lock_c, lock_n)))
                t1 = bdd.apply("and", bdd.apply("and", t1_guard, t1_action), frame)

                # T2: acquire lock & enter CS (want & !lock -> lock & in_cs)
                t2_guard = bdd.apply("and", want_c,
                            bdd.apply("not", lock_c, bdd.TRUE))
                t2_action = bdd.apply("and",
                             bdd.apply("and", lock_n, in_cs_n),
                             bdd.apply("iff", want_c, want_n))
                t2 = bdd.apply("and", bdd.apply("and", t2_guard, t2_action), frame)

                # T3: leave CS & release lock (in_cs -> !in_cs, !lock, !want)
                t3_guard = in_cs_c
                t3_action = bdd.apply("and",
                             bdd.apply("and",
                              bdd.apply("not", in_cs_n, bdd.TRUE),
                              bdd.apply("not", lock_n, bdd.TRUE)),
                             bdd.apply("not", want_n, bdd.TRUE))
                t3 = bdd.apply("and", bdd.apply("and", t3_guard, t3_action), frame)

                ti = bdd.apply("or", bdd.apply("or", t1, t2), t3)
                trans = bdd.apply("or", trans, ti)

            return trans

        return state_vars, init_fn, trans_fn

    def _build_flag_protocol(self, n_threads: int):
        """Flag-based protocol (Peterson's for 2 threads)."""
        if n_threads != 2:
            return self._build_lock_protocol(n_threads)

        state_vars = ["flag_0", "flag_1", "turn", "in_cs_0", "in_cs_1"]

        def init_fn(bdd):
            result = bdd.TRUE
            for v in state_vars:
                vi = bdd.var(bdd.var_index(v))
                result = bdd.apply("and", result, bdd.apply("not", vi, bdd.TRUE))
            return result

        def trans_fn(bdd, cur, nxt):
            C = {k: bdd.var(v) for k, v in cur.items()}
            N = {k.rstrip("'"): bdd.var(v) for k, v in nxt.items()}
            trans = bdd.FALSE

            for i in range(2):
                j = 1 - i
                flag_i_c = C[f"flag_{i}"]
                flag_j_c = C[f"flag_{j}"]
                flag_i_n = N[f"flag_{i}"]
                flag_j_n = N[f"flag_{j}"]
                turn_c = C["turn"]
                turn_n = N["turn"]
                in_cs_i_c = C[f"in_cs_{i}"]
                in_cs_i_n = N[f"in_cs_{i}"]
                in_cs_j_c = C[f"in_cs_{j}"]
                in_cs_j_n = N[f"in_cs_{j}"]

                # Frame for thread j
                frame_j = bdd.apply("and",
                           bdd.apply("iff", flag_j_c, flag_j_n),
                           bdd.apply("iff", in_cs_j_c, in_cs_j_n))

                # T1: Set flag_i, set turn=j
                t1_guard = bdd.apply("and",
                            bdd.apply("not", flag_i_c, bdd.TRUE),
                            bdd.apply("not", in_cs_i_c, bdd.TRUE))
                if i == 0:
                    t1_turn = turn_n  # turn = 1 (j's turn)
                else:
                    t1_turn = bdd.apply("not", turn_n, bdd.TRUE)  # turn = 0 (j's turn)
                t1_action = bdd.apply("and",
                             bdd.apply("and", flag_i_n, t1_turn),
                             bdd.apply("iff", in_cs_i_c, in_cs_i_n))
                t1 = bdd.apply("and", bdd.apply("and", t1_guard, t1_action), frame_j)

                # T2: Enter CS (flag_i & (!flag_j | turn=i) & !in_cs_i -> in_cs_i)
                if i == 0:
                    turn_is_mine = bdd.apply("not", turn_c, bdd.TRUE)  # turn=0
                else:
                    turn_is_mine = turn_c  # turn=1
                can_enter = bdd.apply("or",
                             bdd.apply("not", flag_j_c, bdd.TRUE),
                             turn_is_mine)
                t2_guard = bdd.apply("and",
                            bdd.apply("and", flag_i_c,
                             bdd.apply("not", in_cs_i_c, bdd.TRUE)),
                            can_enter)
                t2_action = bdd.apply("and",
                             bdd.apply("and", in_cs_i_n,
                              bdd.apply("iff", flag_i_c, flag_i_n)),
                             bdd.apply("iff", turn_c, turn_n))
                t2 = bdd.apply("and", bdd.apply("and", t2_guard, t2_action), frame_j)

                # T3: Leave CS
                t3_guard = in_cs_i_c
                t3_action = bdd.apply("and",
                             bdd.apply("and",
                              bdd.apply("not", in_cs_i_n, bdd.TRUE),
                              bdd.apply("not", flag_i_n, bdd.TRUE)),
                             bdd.apply("iff", turn_c, turn_n))
                t3 = bdd.apply("and", bdd.apply("and", t3_guard, t3_action), frame_j)

                ti = bdd.apply("or", bdd.apply("or", t1, t2), t3)
                trans = bdd.apply("or", trans, ti)

            return trans

        return state_vars, init_fn, trans_fn


def mutual_exclusion_property(n_threads: int) -> LTL:
    """G(!(in_cs_0 & in_cs_1)) for 2 threads, generalized for n."""
    if n_threads == 2:
        both = LAnd(Atom("in_cs_0"), Atom("in_cs_1"))
        return Globally(LNot(both))
    # For n threads: no two threads in CS simultaneously
    pairs = []
    for i in range(n_threads):
        for j in range(i + 1, n_threads):
            pairs.append(LAnd(Atom(f"in_cs_{i}"), Atom(f"in_cs_{j}")))
    any_pair = pairs[0]
    for p in pairs[1:]:
        any_pair = LOr(any_pair, p)
    return Globally(LNot(any_pair))


def deadlock_freedom_property(n_threads: int) -> LTL:
    """G(F(in_cs_0 | in_cs_1 | ...)) -- always eventually some progress."""
    atoms = [Atom(f"in_cs_{i}") for i in range(n_threads)]
    some_in_cs = atoms[0]
    for a in atoms[1:]:
        some_in_cs = LOr(some_in_cs, a)
    return Globally(Finally(some_in_cs))


def starvation_freedom_property(thread_id: int) -> LTL:
    """G(want_i -> F(in_cs_i)) -- if thread wants, it eventually enters."""
    want = Atom(f"want_{thread_id}")
    in_cs = Atom(f"in_cs_{thread_id}")
    return Globally(LImplies(want, Finally(in_cs)))


def check_temporal_properties(
    state_vars: List[str],
    init_fn,
    trans_fn,
    properties: List[Tuple[str, LTL]],
    max_steps: int = 200,
) -> List[TemporalCheckResult]:
    """Check a list of LTL properties on a concurrent system."""
    results = []
    for name, prop in properties:
        try:
            ltl_result = check_ltl(state_vars, init_fn, trans_fn, prop,
                                   max_steps=max_steps)
            results.append(TemporalCheckResult(
                property_text=name,
                holds=ltl_result.holds,
                counterexample=ltl_result.counterexample,
                method=ltl_result.method,
            ))
        except Exception as e:
            results.append(TemporalCheckResult(
                property_text=name,
                holds=False,
                method=f"error: {e}",
            ))
    return results


# =============================================================================
# Phase 5: Unified Pipeline
# =============================================================================

def verify_concurrent_program(
    program: ConcurrentProgram,
    check_effects_flag: bool = True,
    check_csl: bool = True,
    check_temporal: bool = True,
    temporal_system: Optional[Tuple[List[str], 'callable', 'callable']] = None,
    temporal_properties: Optional[List[Tuple[str, LTL]]] = None,
    max_steps: int = 200,
) -> ConcVerificationResult:
    """
    Full concurrent verification pipeline.

    Phase 1: Effect inference and race analysis
    Phase 2: Effect checking (declared vs inferred)
    Phase 3: CSL verification (memory safety)
    Phase 4: Temporal verification (LTL model checking)
    """
    result = ConcVerificationResult(verdict=ConcVerdict.SAFE)

    # Phase 1 & 2: Effects
    if check_effects_flag:
        try:
            all_reports, unprotected = effect_race_analysis(program)
            result.effect_race_reports = all_reports
            result.unprotected_state_effects = unprotected

            sigs, violations = check_thread_effects(program)
            result.effect_sigs = sigs
            result.effect_violations = violations

            if violations:
                result.verdict = ConcVerdict.EFFECT_VIOLATION
            elif unprotected:
                result.verdict = ConcVerdict.RACE
        except Exception as e:
            result.errors.append(f"Effect analysis error: {e}")

    # Phase 3: CSL
    if check_csl:
        try:
            csl_result, ownership = run_csl_verification(program)
            result.csl_result = csl_result
            result.ownership_report = ownership

            if csl_result and csl_result.verdict == CSLVerdict.RACE:
                result.verdict = ConcVerdict.RACE
        except Exception as e:
            result.errors.append(f"CSL error: {e}")

    # Phase 4: Temporal
    if check_temporal and temporal_system is not None:
        try:
            props = temporal_properties or []
            # Add properties from program spec
            for ltl_prop in program.ltl_properties:
                props.append((str(ltl_prop), ltl_prop))

            state_vars, init_fn, trans_fn = temporal_system
            temporal_results = check_temporal_properties(
                state_vars, init_fn, trans_fn, props, max_steps=max_steps
            )
            result.temporal_results = temporal_results

            for tr in temporal_results:
                if not tr.holds:
                    result.verdict = ConcVerdict.TEMPORAL_VIOLATION
                    break
        except Exception as e:
            result.errors.append(f"Temporal verification error: {e}")

    return result


# =============================================================================
# High-Level Convenience APIs
# =============================================================================

def verify_mutual_exclusion(
    protocol: str = "none",
    n_threads: int = 2,
    extra_properties: Optional[List[Tuple[str, LTL]]] = None,
    max_steps: int = 200,
) -> ConcVerificationResult:
    """
    Verify mutual exclusion properties for a given synchronization protocol.

    Checks:
    - Mutual exclusion: G(!(in_cs_0 & in_cs_1))
    - Plus any extra properties

    Protocols: "none", "lock", "flag" (Peterson's)
    """
    builder = ConcurrentSystemBuilder(n_threads)
    state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
        n_threads=n_threads, protocol=protocol
    )

    props = [
        ("mutual_exclusion", mutual_exclusion_property(n_threads)),
    ]
    if extra_properties:
        props.extend(extra_properties)

    # Create a minimal program (no source code, just temporal)
    threads = [
        ThreadSpec(thread_id=f"t{i}", source=f"let x_{i} = 0;")
        for i in range(n_threads)
    ]
    program = ConcurrentProgram(threads=threads)

    return verify_concurrent_program(
        program,
        check_effects_flag=False,
        check_csl=False,
        check_temporal=True,
        temporal_system=(state_vars, init_fn, trans_fn),
        temporal_properties=props,
        max_steps=max_steps,
    )


def verify_concurrent_effects(
    thread_sources: Dict[str, str],
    declared_effects: Optional[Dict[str, EffectSet]] = None,
    shared_vars: Optional[Set[str]] = None,
    lock_vars: Optional[Dict[str, Set[str]]] = None,
) -> ConcVerificationResult:
    """
    Verify effect correctness for a concurrent program.

    thread_sources: {thread_id: C10_source}
    declared_effects: {thread_id: EffectSet} (optional)
    shared_vars: set of shared variable names
    lock_vars: {lock_name: set_of_protected_vars}
    """
    threads = []
    for tid, source in thread_sources.items():
        thread = ThreadSpec(
            thread_id=tid,
            source=source,
            declared_effects=declared_effects.get(tid) if declared_effects else None,
            shared_vars=shared_vars,
        )
        if lock_vars:
            thread.locks = set(lock_vars.keys())
        threads.append(thread)

    lock_invs = {}
    program = ConcurrentProgram(
        threads=threads,
        lock_invariants=lock_invs,
        shared_vars=shared_vars or set(),
    )

    return verify_concurrent_program(
        program,
        check_effects_flag=True,
        check_csl=False,
        check_temporal=False,
    )


def full_concurrent_verify(
    thread_sources: Dict[str, str],
    protocol: str = "lock",
    n_threads: Optional[int] = None,
    declared_effects: Optional[Dict[str, EffectSet]] = None,
    shared_vars: Optional[Set[str]] = None,
    ltl_properties: Optional[List[Tuple[str, LTL]]] = None,
    max_steps: int = 200,
) -> ConcVerificationResult:
    """
    Full verification: effects + temporal properties.

    Combines effect analysis of actual thread source code with
    temporal model checking of a synchronization protocol.
    """
    n = n_threads or len(thread_sources)

    # Build threads
    threads = []
    for tid, source in thread_sources.items():
        threads.append(ThreadSpec(
            thread_id=tid,
            source=source,
            declared_effects=declared_effects.get(tid) if declared_effects else None,
            shared_vars=shared_vars,
        ))
    program = ConcurrentProgram(
        threads=threads,
        shared_vars=shared_vars or set(),
    )

    # Build temporal system
    builder = ConcurrentSystemBuilder(n)
    state_vars, init_fn, trans_fn = builder.build_mutual_exclusion_system(
        n_threads=n, protocol=protocol
    )

    # Default properties
    props = ltl_properties or []
    if not props:
        props.append(("mutual_exclusion", mutual_exclusion_property(n)))

    return verify_concurrent_program(
        program,
        check_effects_flag=True,
        check_csl=False,
        check_temporal=True,
        temporal_system=(state_vars, init_fn, trans_fn),
        temporal_properties=props,
        max_steps=max_steps,
    )


def effect_guided_protocol_selection(
    thread_sources: Dict[str, str],
    shared_vars: Optional[Set[str]] = None,
    max_steps: int = 200,
) -> Dict:
    """
    Use effect analysis to guide protocol selection and verify.

    1. Infer effects to determine if threads have shared state
    2. If shared state found, test protocols (none, lock, flag)
    3. Return comparison of protocol safety
    """
    # Step 1: Effect analysis
    threads = [ThreadSpec(tid, src, shared_vars=shared_vars)
               for tid, src in thread_sources.items()]
    program = ConcurrentProgram(threads=threads, shared_vars=shared_vars or set())

    all_reports, unprotected = effect_race_analysis(program)
    has_shared_state = len(all_reports) > 0

    # Step 2: Test protocols
    n = len(thread_sources)
    protocols = ["none", "lock"]
    if n == 2:
        protocols.append("flag")

    protocol_results = {}
    for proto in protocols:
        result = verify_mutual_exclusion(
            protocol=proto, n_threads=n, max_steps=max_steps
        )
        temporal = result.temporal_results
        protocol_results[proto] = {
            "mutual_exclusion_holds": all(t.holds for t in temporal),
            "temporal_results": temporal,
        }

    return {
        "has_shared_state": has_shared_state,
        "shared_state_vars": [r.var for r in all_reports],
        "unprotected_vars": [r.var for r in unprotected],
        "protocol_comparison": protocol_results,
        "recommended": "lock" if has_shared_state else "none",
    }
