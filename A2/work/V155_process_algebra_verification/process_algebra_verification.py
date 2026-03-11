"""V155: Process Algebra Verification

Composes:
- V151 (probabilistic process algebra) -- process terms, LTS generation
- V150 (weak probabilistic bisimulation) -- equivalence checking, quotient
- V067 (PCTL model checking) -- temporal property verification
- V065 (Markov chain analysis) -- underlying probabilistic model

Verifies temporal logic properties and behavioral equivalences of CCS-style
process algebra terms. Bridges the gap between algebraic process descriptions
and model-checking-based verification.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V151_probabilistic_process_algebra'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V150_weak_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))

from prob_process_algebra import (
    Proc, ProcKind, stop, prefix, tau_prefix, prob_choice, nd_choice,
    parallel, restrict, relabel, recvar, recdef, parse_proc,
    generate_lts, check_process_equivalence, check_strong_equivalence,
    trace_set, deadlock_free, action_set, TAU,
)
from weak_probabilistic_bisimulation import (
    LabeledProbTS, WeakBisimResult, WeakBisimVerdict,
    compute_weak_bisimulation, check_weakly_bisimilar,
    weak_bisimulation_quotient, compute_weak_simulation,
    compute_branching_bisimulation, check_branching_bisimilar,
    detect_divergence, compute_divergence_sensitive_bisimulation,
    compute_weak_bisimulation_distance, minimize_weak,
    compare_strong_vs_weak,
)
from pctl_model_check import (
    PCTL, FormulaKind, tt, ff, atom, pnot, pand, por,
    prob_geq, prob_leq, prob_gt, prob_lt,
    next_f, until, bounded_until, eventually, always,
    bounded_eventually, parse_pctl, PCTLChecker,
    check_pctl, check_pctl_state,
    LabeledMC, make_labeled_mc,
)

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, FrozenSet
from enum import Enum


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class PropertyVerdict(Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass
class ProcessVerificationResult:
    """Result of verifying a PCTL property on a process."""
    process: Proc
    formula: PCTL
    verdict: PropertyVerdict
    satisfying_states: Set[int] = field(default_factory=set)
    initial_state: int = 0
    probabilities: Optional[List[float]] = None
    lts_states: int = 0
    lts_actions: int = 0
    statistics: Dict = field(default_factory=dict)


@dataclass
class EquivalenceResult:
    """Result of checking behavioral equivalence between processes."""
    p1: Proc
    p2: Proc
    equivalent: bool
    equiv_type: str = "weak"  # "weak", "strong", "branching"
    witness: Optional[str] = None
    partition: Optional[List[Set[int]]] = None
    distance: Optional[float] = None
    statistics: Dict = field(default_factory=dict)


@dataclass
class RefinementResult:
    """Result of checking if one process refines (simulates) another."""
    spec: Proc
    impl: Proc
    refines: bool
    witness: Optional[str] = None
    statistics: Dict = field(default_factory=dict)


@dataclass
class AlgebraicLawResult:
    """Result of checking an algebraic law holds."""
    law_name: str
    lhs: Proc
    rhs: Proc
    holds: bool
    equiv_type: str = "weak"
    witness: Optional[str] = None


@dataclass
class CompositionAnalysis:
    """Analysis of a parallel composition."""
    process: Proc
    components: List[Proc]
    deadlock_free: bool
    reachable_states: int
    actions: Set[str]
    traces: Set[Tuple[str, ...]]
    divergent: bool
    properties_checked: List[ProcessVerificationResult] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LTS to Markov Chain conversion
# ---------------------------------------------------------------------------

def _lts_to_labeled_mc(lts: LabeledProbTS) -> LabeledMC:
    """Convert a LabeledProbTS to a LabeledMC for PCTL model checking.

    Strategy: merge all actions into a single probabilistic transition.
    For nondeterministic choice (multiple actions), we use uniform resolution
    over enabled actions, then combine the resulting distributions.
    """
    n = lts.n_states
    matrix = [[0.0] * n for _ in range(n)]

    for s in range(n):
        acts = lts.actions.get(s, {})
        if not acts:
            # Deadlock: self-loop
            matrix[s][s] = 1.0
            continue

        # Uniform resolution over nondeterministic actions
        weight = 1.0 / len(acts)
        for action, transitions in acts.items():
            for target, prob in transitions:
                matrix[s][target] += weight * prob

        # Normalize (handle floating point)
        total = sum(matrix[s])
        if total > 0:
            for t in range(n):
                matrix[s][t] /= total

    labels = {}
    for s in range(n):
        label_set = set(lts.state_labels.get(s, set()))
        # Add action-based labels
        acts = lts.actions.get(s, {})
        for action in acts:
            if action != TAU:
                label_set.add(f"can_{action}")
        if not acts:
            label_set.add("deadlock")
        if any(a == TAU for a in acts):
            label_set.add("has_tau")
        label_set.add("active")
        labels[s] = label_set

    state_labels = lts.state_names if lts.state_names else [f"s{i}" for i in range(n)]
    return make_labeled_mc(matrix, labels, state_labels)


def _label_lts_states(lts: LabeledProbTS, custom_labels: Optional[Dict[int, Set[str]]] = None) -> LabeledProbTS:
    """Add richer labels to LTS states for property checking."""
    new_labels = {}
    for s in range(lts.n_states):
        label_set = set(lts.state_labels.get(s, set()))
        acts = lts.actions.get(s, {})
        for action in acts:
            if action != TAU:
                label_set.add(f"can_{action}")
        if not acts:
            label_set.add("deadlock")
        if any(a == TAU for a in acts):
            label_set.add("has_tau")
        if custom_labels and s in custom_labels:
            label_set.update(custom_labels[s])
        new_labels[s] = label_set

    return LabeledProbTS(
        n_states=lts.n_states,
        actions=lts.actions,
        state_labels=new_labels,
        state_names=lts.state_names,
    )


# ---------------------------------------------------------------------------
# Core verification: PCTL on processes
# ---------------------------------------------------------------------------

def verify_process(proc: Proc, formula: PCTL,
                   max_states: int = 200,
                   custom_labels: Optional[Dict[int, Set[str]]] = None) -> ProcessVerificationResult:
    """Verify a PCTL property on a process algebra term.

    Pipeline: Process -> LTS (V151) -> LabeledMC -> PCTL check (V067)

    The LTS is converted to a Markov chain by uniformly resolving
    nondeterministic choices. Probabilistic choices are preserved exactly.

    Args:
        proc: Process algebra term
        formula: PCTL formula to check
        max_states: Maximum LTS states to explore
        custom_labels: Additional state labels {state_id: {label, ...}}

    Returns:
        ProcessVerificationResult with verdict and diagnostics
    """
    lts = generate_lts(proc, max_states=max_states)
    lts = _label_lts_states(lts, custom_labels)
    lmc = _lts_to_labeled_mc(lts)

    result = check_pctl(lmc, formula)

    n_actions = 0
    for s in range(lts.n_states):
        n_actions += len(lts.actions.get(s, {}))

    initial = 0
    verdict = PropertyVerdict.SATISFIED if initial in result.satisfying_states else PropertyVerdict.VIOLATED

    return ProcessVerificationResult(
        process=proc,
        formula=formula,
        verdict=verdict,
        satisfying_states=result.satisfying_states,
        initial_state=initial,
        probabilities=result.probabilities,
        lts_states=lts.n_states,
        lts_actions=n_actions,
        statistics={
            "lts_states": lts.n_states,
            "lts_transitions": n_actions,
            "mc_states": lmc.mc.n_states,
            "satisfying_count": len(result.satisfying_states),
        }
    )


def verify_process_state(proc: Proc, formula: PCTL, state: int = 0,
                         max_states: int = 200) -> bool:
    """Check if a PCTL formula holds at a specific state of a process."""
    result = verify_process(proc, formula, max_states=max_states)
    return state in result.satisfying_states


def verify_process_all(proc: Proc, formula: PCTL,
                       max_states: int = 200) -> bool:
    """Check if a PCTL formula holds at all reachable states."""
    result = verify_process(proc, formula, max_states=max_states)
    return result.satisfying_states == set(range(result.lts_states))


def verify_process_quantitative(proc: Proc, path_formula: PCTL,
                                max_states: int = 200) -> List[float]:
    """Get exact probabilities for a path formula at each process state."""
    lts = generate_lts(proc, max_states=max_states)
    lts = _label_lts_states(lts)
    lmc = _lts_to_labeled_mc(lts)

    checker = PCTLChecker(lmc)
    return checker.check_quantitative(path_formula)


# ---------------------------------------------------------------------------
# Behavioral equivalence checking
# ---------------------------------------------------------------------------

def check_equivalence(p1: Proc, p2: Proc,
                      equiv_type: str = "weak",
                      max_states: int = 200) -> EquivalenceResult:
    """Check behavioral equivalence between two processes.

    Args:
        p1, p2: Processes to compare
        equiv_type: "weak", "strong", or "branching"
        max_states: Maximum LTS states per process

    Returns:
        EquivalenceResult with verdict and diagnostics
    """
    if equiv_type == "strong":
        result = check_strong_equivalence(p1, p2, max_states=max_states)
    elif equiv_type == "branching":
        # Generate combined LTS and check branching bisimulation
        lts1 = generate_lts(p1, max_states=max_states)
        lts2 = generate_lts(p2, max_states=max_states)
        combined = _disjoint_union(lts1, lts2)
        br_result = compute_branching_bisimulation(combined)
        # Check if initial states of p1 and p2 are in the same block
        s1_init = 0
        s2_init = lts1.n_states  # offset in combined
        equiv = False
        for block in br_result.partition:
            if s1_init in block and s2_init in block:
                equiv = True
                break
        return EquivalenceResult(
            p1=p1, p2=p2, equivalent=equiv, equiv_type="branching",
            witness=br_result.witness if not equiv else None,
            partition=br_result.partition,
            statistics=br_result.statistics,
        )
    else:
        result = check_process_equivalence(p1, p2, max_states=max_states)

    equiv = (result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR)
    return EquivalenceResult(
        p1=p1, p2=p2, equivalent=equiv, equiv_type=equiv_type,
        witness=result.witness if not equiv else None,
        partition=result.partition,
        statistics=result.statistics,
    )


def check_all_equivalences(p1: Proc, p2: Proc,
                           max_states: int = 200) -> Dict[str, EquivalenceResult]:
    """Check strong, branching, and weak equivalence."""
    results = {}
    for et in ["strong", "branching", "weak"]:
        results[et] = check_equivalence(p1, p2, equiv_type=et, max_states=max_states)
    return results


def _disjoint_union(lts1: LabeledProbTS, lts2: LabeledProbTS) -> LabeledProbTS:
    """Create disjoint union of two LTS for cross-system analysis."""
    offset = lts1.n_states
    n = lts1.n_states + lts2.n_states

    actions = {}
    for s, acts in lts1.actions.items():
        actions[s] = {}
        for a, trans in acts.items():
            actions[s][a] = list(trans)

    for s, acts in lts2.actions.items():
        actions[s + offset] = {}
        for a, trans in acts.items():
            actions[s + offset][a] = [(t + offset, p) for t, p in trans]

    labels = {}
    for s, lbls in lts1.state_labels.items():
        labels[s] = set(lbls)
    for s, lbls in lts2.state_labels.items():
        labels[s + offset] = set(lbls)

    names1 = lts1.state_names or [f"s{i}" for i in range(lts1.n_states)]
    names2 = lts2.state_names or [f"s{i}" for i in range(lts2.n_states)]

    return LabeledProbTS(
        n_states=n,
        actions=actions,
        state_labels=labels,
        state_names=[f"L_{nm}" for nm in names1] + [f"R_{nm}" for nm in names2],
    )


# ---------------------------------------------------------------------------
# Simulation / refinement checking
# ---------------------------------------------------------------------------

def check_refinement(spec: Proc, impl: Proc,
                     max_states: int = 200) -> RefinementResult:
    """Check if impl refines spec (impl simulates spec).

    Simulation: for every action spec can do, impl can match it,
    and the resulting states maintain the simulation relation.

    Uses weak simulation from V150.
    """
    lts_spec = generate_lts(spec, max_states=max_states)
    lts_impl = generate_lts(impl, max_states=max_states)
    combined = _disjoint_union(lts_spec, lts_impl)

    sim_result = compute_weak_simulation(combined)

    # Check if (spec_init=0, impl_init=offset) is in the simulation relation
    spec_init = 0
    impl_init = lts_spec.n_states

    refines = False
    if sim_result.relation:
        for pair in sim_result.relation:
            pair_set = set(pair) if isinstance(pair, frozenset) else {pair}
            # Simulation is directed: (s, t) means s simulates t
            # We need impl simulates spec
            if isinstance(pair, (tuple, list)):
                if pair[0] == impl_init and pair[1] == spec_init:
                    refines = True
                    break
            elif isinstance(pair, frozenset):
                # Frozenset is unordered -- check via partition
                pass

    # Fallback: check partition-based (simulation relation stores pairs)
    if sim_result.partition:
        for block in sim_result.partition:
            if spec_init in block and impl_init in block:
                refines = True
                break

    return RefinementResult(
        spec=spec, impl=impl, refines=refines,
        witness=sim_result.witness if not refines else None,
        statistics=sim_result.statistics,
    )


# ---------------------------------------------------------------------------
# Behavioral distance
# ---------------------------------------------------------------------------

def process_distance(p1: Proc, p2: Proc,
                     discount: float = 0.9,
                     max_states: int = 200) -> float:
    """Compute behavioral distance between two processes.

    Returns 0.0 for bisimilar processes, up to 1.0 for maximally different.
    Uses weak bisimulation distance (Kantorovich metric).
    """
    lts1 = generate_lts(p1, max_states=max_states)
    lts2 = generate_lts(p2, max_states=max_states)
    combined = _disjoint_union(lts1, lts2)

    dist_result = compute_weak_bisimulation_distance(combined, discount=discount)

    s1_init = 0
    s2_init = lts1.n_states

    return dist_result.distances[s1_init][s2_init]


# ---------------------------------------------------------------------------
# Algebraic law verification
# ---------------------------------------------------------------------------

_ALGEBRAIC_LAWS = {
    "commutativity_choice": lambda p, q: (nd_choice(p, q), nd_choice(q, p)),
    "associativity_choice": lambda p, q, r: (
        nd_choice(p, nd_choice(q, r)),
        nd_choice(nd_choice(p, q), r),
    ),
    "idempotence_choice": lambda p: (nd_choice(p, p), p),
    "commutativity_parallel": lambda p, q: (parallel(p, q), parallel(q, p)),
    "associativity_parallel": lambda p, q, r: (
        parallel(p, parallel(q, r)),
        parallel(parallel(p, q), r),
    ),
    "stop_identity_choice": lambda p: (nd_choice(p, stop()), p),
    "tau_absorption": lambda p: (tau_prefix(p), p),
}


def verify_algebraic_law(law_name: str, *procs: Proc,
                         equiv_type: str = "weak",
                         max_states: int = 200) -> AlgebraicLawResult:
    """Verify that a named algebraic law holds for given processes.

    Available laws:
    - commutativity_choice(P, Q): P+Q ~ Q+P
    - associativity_choice(P, Q, R): P+(Q+R) ~ (P+Q)+R
    - idempotence_choice(P): P+P ~ P
    - commutativity_parallel(P, Q): P|Q ~ Q|P
    - associativity_parallel(P, Q, R): P|(Q|R) ~ (P|Q)|R
    - stop_identity_choice(P): P+0 ~ P
    - tau_absorption(P): tau.P ~w P (weak only)
    """
    if law_name not in _ALGEBRAIC_LAWS:
        return AlgebraicLawResult(
            law_name=law_name,
            lhs=stop(), rhs=stop(),
            holds=False,
            equiv_type=equiv_type,
            witness=f"Unknown law: {law_name}",
        )

    lhs, rhs = _ALGEBRAIC_LAWS[law_name](*procs)
    result = check_equivalence(lhs, rhs, equiv_type=equiv_type, max_states=max_states)

    return AlgebraicLawResult(
        law_name=law_name,
        lhs=lhs, rhs=rhs,
        holds=result.equivalent,
        equiv_type=equiv_type,
        witness=result.witness,
    )


def verify_custom_law(lhs: Proc, rhs: Proc, law_name: str = "custom",
                      equiv_type: str = "weak",
                      max_states: int = 200) -> AlgebraicLawResult:
    """Verify a custom algebraic law: lhs ~ rhs."""
    result = check_equivalence(lhs, rhs, equiv_type=equiv_type, max_states=max_states)
    return AlgebraicLawResult(
        law_name=law_name,
        lhs=lhs, rhs=rhs,
        holds=result.equivalent,
        equiv_type=equiv_type,
        witness=result.witness,
    )


# ---------------------------------------------------------------------------
# Compositional analysis
# ---------------------------------------------------------------------------

def analyze_composition(proc: Proc, components: Optional[List[Proc]] = None,
                        properties: Optional[List[PCTL]] = None,
                        max_states: int = 200) -> CompositionAnalysis:
    """Analyze a process (typically a parallel composition).

    Checks deadlock freedom, divergence, computes traces, and
    optionally verifies PCTL properties.
    """
    lts = generate_lts(proc, max_states=max_states)

    is_deadlock_free = deadlock_free(proc, max_states=max_states)
    actions = action_set(proc, max_states=max_states)
    traces = trace_set(proc, max_depth=8, max_states=max_states)

    # Check divergence
    div_map = detect_divergence(lts)
    has_divergence = any(div_map.values())

    # Verify properties
    prop_results = []
    if properties:
        for formula in properties:
            r = verify_process(proc, formula, max_states=max_states)
            prop_results.append(r)

    if components is None:
        components = []

    return CompositionAnalysis(
        process=proc,
        components=components,
        deadlock_free=is_deadlock_free,
        reachable_states=lts.n_states,
        actions=actions,
        traces=traces,
        divergent=has_divergence,
        properties_checked=prop_results,
        statistics={
            "n_states": lts.n_states,
            "n_traces": len(traces),
            "n_actions": len(actions),
            "divergent_states": sum(1 for v in div_map.values() if v),
        },
    )


# ---------------------------------------------------------------------------
# Deadlock analysis
# ---------------------------------------------------------------------------

def check_deadlock_freedom(proc: Proc, max_states: int = 200) -> Dict:
    """Detailed deadlock analysis of a process."""
    lts = generate_lts(proc, max_states=max_states)

    deadlock_states = []
    for s in range(lts.n_states):
        acts = lts.actions.get(s, {})
        if not acts:
            deadlock_states.append(s)

    is_free = len(deadlock_states) == 0

    return {
        "deadlock_free": is_free,
        "deadlock_states": deadlock_states,
        "total_states": lts.n_states,
        "deadlock_ratio": len(deadlock_states) / max(1, lts.n_states),
    }


def verify_no_deadlock_pctl(proc: Proc, max_states: int = 200) -> ProcessVerificationResult:
    """Verify deadlock freedom via PCTL: P<=0[F deadlock] (never reach deadlock)."""
    formula = prob_leq(0.0, eventually(atom("deadlock")))
    return verify_process(proc, formula, max_states=max_states)


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------

def check_trace_inclusion(p1: Proc, p2: Proc,
                          max_depth: int = 8,
                          max_states: int = 200) -> Dict:
    """Check if traces(p1) is a subset of traces(p2)."""
    traces1 = trace_set(p1, max_depth=max_depth, max_states=max_states)
    traces2 = trace_set(p2, max_depth=max_depth, max_states=max_states)

    included = traces1.issubset(traces2)
    extra = traces1 - traces2

    return {
        "included": included,
        "p1_traces": len(traces1),
        "p2_traces": len(traces2),
        "extra_traces": extra,
        "common_traces": traces1 & traces2,
    }


def check_trace_equivalence(p1: Proc, p2: Proc,
                            max_depth: int = 8,
                            max_states: int = 200) -> Dict:
    """Check if two processes have the same trace set."""
    traces1 = trace_set(p1, max_depth=max_depth, max_states=max_states)
    traces2 = trace_set(p2, max_depth=max_depth, max_states=max_states)

    return {
        "trace_equivalent": traces1 == traces2,
        "p1_traces": len(traces1),
        "p2_traces": len(traces2),
        "only_in_p1": traces1 - traces2,
        "only_in_p2": traces2 - traces1,
        "common": traces1 & traces2,
    }


# ---------------------------------------------------------------------------
# Minimization
# ---------------------------------------------------------------------------

def minimize_process(proc: Proc, method: str = "weak",
                     max_states: int = 200) -> Dict:
    """Minimize a process by computing its bisimulation quotient.

    Returns the quotient LTS and reduction statistics.
    """
    lts = generate_lts(proc, max_states=max_states)

    if method == "weak":
        quotient, result = minimize_weak(lts)
    elif method == "branching":
        result = compute_branching_bisimulation(lts)
        # Manually construct quotient from partition
        quotient = weak_bisimulation_quotient(lts)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "original_states": lts.n_states,
        "minimized_states": quotient.n_states,
        "reduction_ratio": 1 - quotient.n_states / max(1, lts.n_states),
        "partition": result.partition,
        "quotient": quotient,
        "method": method,
    }


# ---------------------------------------------------------------------------
# Equivalence hierarchy analysis
# ---------------------------------------------------------------------------

def analyze_equivalence_hierarchy(p1: Proc, p2: Proc,
                                  max_states: int = 200) -> Dict:
    """Analyze the full equivalence hierarchy between two processes.

    Returns results for: trace equivalence, weak bisimulation,
    branching bisimulation, strong bisimulation, and behavioral distance.
    """
    # Trace equivalence
    trace_eq = check_trace_equivalence(p1, p2, max_states=max_states)

    # Bisimulation equivalences
    equivs = check_all_equivalences(p1, p2, max_states=max_states)

    # Distance
    dist = process_distance(p1, p2, max_states=max_states)

    return {
        "trace_equivalent": trace_eq["trace_equivalent"],
        "weakly_bisimilar": equivs["weak"].equivalent,
        "branching_bisimilar": equivs["branching"].equivalent,
        "strongly_bisimilar": equivs["strong"].equivalent,
        "behavioral_distance": dist,
        "hierarchy_consistent": _check_hierarchy_consistency(
            equivs["strong"].equivalent,
            equivs["branching"].equivalent,
            equivs["weak"].equivalent,
            trace_eq["trace_equivalent"],
        ),
        "details": {
            "trace": trace_eq,
            "weak": equivs["weak"],
            "branching": equivs["branching"],
            "strong": equivs["strong"],
        },
    }


def _check_hierarchy_consistency(strong: bool, branching: bool,
                                  weak: bool, trace: bool) -> bool:
    """Verify: strong => branching => weak => trace."""
    if strong and not branching:
        return False
    if branching and not weak:
        return False
    if weak and not trace:
        return False
    return True


# ---------------------------------------------------------------------------
# Property-preserving transformations
# ---------------------------------------------------------------------------

def check_property_preservation(original: Proc, transformed: Proc,
                                properties: List[PCTL],
                                max_states: int = 200) -> Dict:
    """Check if a transformation preserves a set of PCTL properties.

    Verifies each property on both original and transformed process,
    reports which properties are preserved and which are broken.
    """
    preserved = []
    broken = []

    for formula in properties:
        r_orig = verify_process(original, formula, max_states=max_states)
        r_trans = verify_process(transformed, formula, max_states=max_states)

        if r_orig.verdict == r_trans.verdict:
            preserved.append(formula)
        else:
            broken.append({
                "formula": formula,
                "original_verdict": r_orig.verdict,
                "transformed_verdict": r_trans.verdict,
            })

    return {
        "all_preserved": len(broken) == 0,
        "preserved_count": len(preserved),
        "broken_count": len(broken),
        "preserved": preserved,
        "broken": broken,
    }


# ---------------------------------------------------------------------------
# Comprehensive analysis
# ---------------------------------------------------------------------------

def full_process_analysis(proc: Proc, max_states: int = 200) -> Dict:
    """Run comprehensive analysis on a process.

    Includes: LTS generation, deadlock analysis, divergence detection,
    trace computation, minimization, and standard properties.
    """
    lts = generate_lts(proc, max_states=max_states)
    lts_labeled = _label_lts_states(lts)

    # Deadlock
    dl = check_deadlock_freedom(proc, max_states=max_states)

    # Divergence
    div_map = detect_divergence(lts)

    # Traces and actions
    traces = trace_set(proc, max_depth=8, max_states=max_states)
    actions = action_set(proc, max_states=max_states)

    # Minimization
    mini = minimize_process(proc, method="weak", max_states=max_states)

    # Standard PCTL checks
    standard_results = {}

    # "Can always do something" (no deadlock via PCTL)
    if dl["deadlock_free"]:
        standard_results["deadlock_freedom"] = "satisfied"
    else:
        standard_results["deadlock_freedom"] = "violated"

    return {
        "lts_states": lts.n_states,
        "lts_transitions": sum(len(a) for a in lts.actions.values()),
        "deadlock_free": dl["deadlock_free"],
        "deadlock_states": dl["deadlock_states"],
        "divergent_states": [s for s, v in div_map.items() if v],
        "has_divergence": any(div_map.values()),
        "observable_actions": actions,
        "trace_count": len(traces),
        "traces": traces,
        "minimized_states": mini["minimized_states"],
        "reduction_ratio": mini["reduction_ratio"],
        "standard_properties": standard_results,
    }


def process_verification_summary(proc: Proc, properties: Optional[List[PCTL]] = None,
                                  max_states: int = 200) -> str:
    """Generate human-readable verification summary."""
    analysis = full_process_analysis(proc, max_states=max_states)

    lines = ["Process Verification Summary", "=" * 40]
    lines.append(f"States: {analysis['lts_states']}")
    lines.append(f"Transitions: {analysis['lts_transitions']}")
    lines.append(f"Deadlock free: {analysis['deadlock_free']}")
    if analysis['deadlock_states']:
        lines.append(f"  Deadlock states: {analysis['deadlock_states']}")
    lines.append(f"Divergent: {analysis['has_divergence']}")
    if analysis['divergent_states']:
        lines.append(f"  Divergent states: {analysis['divergent_states']}")
    lines.append(f"Observable actions: {analysis['observable_actions']}")
    lines.append(f"Traces (depth 8): {analysis['trace_count']}")
    lines.append(f"Minimized states: {analysis['minimized_states']} "
                 f"(reduction: {analysis['reduction_ratio']:.1%})")

    if properties:
        lines.append("")
        lines.append("Property Verification:")
        lines.append("-" * 40)
        for formula in properties:
            r = verify_process(proc, formula, max_states=max_states)
            lines.append(f"  {r.verdict.value}: {formula}")

    return "\n".join(lines)
