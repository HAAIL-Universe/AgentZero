"""
V014: Interpolation-Based CEGAR Model Checking
================================================
Composes: V012 (Craig interpolation) + V010 (predicate abstraction + CEGAR)
        + V002 (PDR/IC3) + C037 (SMT solver) + C010 (parser)

Replaces V010's WP-based predicate refinement with interpolation-based refinement.
When a spurious counterexample is found:
  1. Build a BMC-style unrolling along the abstract trace
  2. The unrolling is UNSAT (spurious trace proves infeasibility)
  3. Compute sequence interpolants via V012
  4. Extract predicates from interpolants
  5. These predicates capture multi-step relationships (not just one-step WP)

Key advantage over V010: WP-based refinement only adds one-step lookahead
predicates (P[x := f(x)]). Interpolation can discover predicates spanning
multiple transitions, which is critical for systems where the abstraction
needs multi-step reasoning to separate reachable from unreachable states.

Also provides a direct BMC+interpolation model checker (no predicate abstraction)
that maintains reachability frames strengthened by interpolants.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Tuple
from enum import Enum

# Path setup
_base = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_base, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V010_predicate_abstraction_cegar'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V012_craig_interpolation'))

from smt_solver import (SMTSolver, SMTResult, Term, Var, IntConst, BoolConst,
                         App, Op, Sort, SortKind)
from pdr import TransitionSystem, check_ts, PDRResult
from pred_abs_cegar import (
    ConcreteTS, Predicate, CEGARVerdict, CEGARStats, CEGARResult,
    cartesian_abstraction, check_counterexample_feasibility,
    auto_predicates_from_ts, extract_loop_ts,
    _and, _or, _not, _implies, _eq, _substitute, _smt_check
)
from craig_interpolation import (
    interpolate, sequence_interpolate, interpolation_refine,
    extract_predicates_from_interpolant, InterpolantResult, collect_vars
)

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class InterpCEGARVerdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class InterpCEGARStats:
    iterations: int = 0
    predicates_initial: int = 0
    predicates_final: int = 0
    spurious_traces: int = 0
    interpolation_successes: int = 0
    interpolation_failures: int = 0
    wp_fallbacks: int = 0
    pdr_calls: int = 0
    predicates_from_interpolation: int = 0
    predicates_from_wp: int = 0


@dataclass
class InterpCEGARResult:
    verdict: InterpCEGARVerdict
    invariant: Optional[list] = None
    counterexample: Optional[list] = None
    predicates: list = field(default_factory=list)
    stats: InterpCEGARStats = field(default_factory=InterpCEGARStats)


# ---------------------------------------------------------------------------
# BMC unrolling for interpolation
# ---------------------------------------------------------------------------

def _build_bmc_formulas(concrete_ts, trace_length):
    """Build step-indexed BMC formulas for interpolation.

    Returns a list of formulas [F0, F1, ..., Fn] where:
      F0 = Init(x_0)
      Fi = Trans(x_{i-1}, x_i)  for i in 1..n-1
      Fn = NOT(Prop(x_{n-1}))

    Variables are step-indexed: x_i for variable x at step i.
    """
    all_vars = concrete_ts.all_vars()
    n = trace_length

    # Create step-indexed variable mappings
    step_vars = []
    for step in range(n):
        sv = {}
        for v in all_vars:
            sv[v] = Var(f"{v}_{step}", INT)
        step_vars.append(sv)

    formulas = []

    # F0: Init(x_0)
    init_map = {v: step_vars[0][v] for v in all_vars}
    init_f = _substitute(concrete_ts.init_formula, init_map)
    formulas.append(init_f)

    # F1..Fn-1: Trans(x_{i-1}, x_i)
    for step in range(1, n):
        trans_map = {}
        for v in all_vars:
            trans_map[v] = step_vars[step - 1][v]
            trans_map[v + "'"] = step_vars[step][v]
        trans_f = _substitute(concrete_ts.trans_formula, trans_map)
        formulas.append(trans_f)

    # Fn: NOT(Prop(x_{n-1}))
    prop_map = {v: step_vars[n - 1][v] for v in all_vars}
    prop_neg = _not(_substitute(concrete_ts.prop_formula, prop_map))
    formulas.append(prop_neg)

    return formulas, step_vars


def _build_constrained_bmc_formulas(concrete_ts, abstract_trace, predicates):
    """Build BMC formulas constrained by abstract trace predicate values.

    Like _build_bmc_formulas but adds predicate constraints at each step
    from the abstract trace. This makes the unrolling tighter (closer to
    the specific spurious path), producing more relevant interpolants.
    """
    all_vars = concrete_ts.all_vars()
    n = len(abstract_trace)

    step_vars = []
    for step in range(n):
        sv = {}
        for v in all_vars:
            sv[v] = Var(f"{v}_{step}", INT)
        step_vars.append(sv)

    formulas = []

    # F0: Init(x_0) AND pred_constraints(x_0)
    init_map = {v: step_vars[0][v] for v in all_vars}
    init_f = _substitute(concrete_ts.init_formula, init_map)
    # Add predicate constraints for step 0
    pred_constrs = _predicate_constraints(predicates, abstract_trace[0], step_vars[0], all_vars)
    formulas.append(_and(init_f, *pred_constrs) if pred_constrs else init_f)

    # F1..Fn-1: Trans(x_{i-1}, x_i) AND pred_constraints(x_i)
    for step in range(1, n):
        trans_map = {}
        for v in all_vars:
            trans_map[v] = step_vars[step - 1][v]
            trans_map[v + "'"] = step_vars[step][v]
        trans_f = _substitute(concrete_ts.trans_formula, trans_map)
        pred_constrs = _predicate_constraints(predicates, abstract_trace[step], step_vars[step], all_vars)
        formulas.append(_and(trans_f, *pred_constrs) if pred_constrs else trans_f)

    # Fn: NOT(Prop(x_{n-1}))
    prop_map = {v: step_vars[n - 1][v] for v in all_vars}
    prop_neg = _not(_substitute(concrete_ts.prop_formula, prop_map))
    formulas.append(prop_neg)

    return formulas, step_vars


def _predicate_constraints(predicates, abs_state, step_vars_at_step, all_vars):
    """Build predicate constraint list from abstract state."""
    constrs = []
    for i, pred in enumerate(predicates):
        pred_name = f"b_{i}"
        if pred_name in abs_state:
            val = abs_state[pred_name]
            is_true = val == 1 if isinstance(val, int) else bool(val)
            var_map = {v: step_vars_at_step[v] for v in all_vars if v in step_vars_at_step}
            pred_at_step = _substitute(pred.formula, var_map)
            if is_true:
                constrs.append(pred_at_step)
            else:
                constrs.append(_not(pred_at_step))
    return constrs


# ---------------------------------------------------------------------------
# Interpolation-based refinement
# ---------------------------------------------------------------------------

def interpolation_refine_trace(concrete_ts, abstract_trace, predicates):
    """Refine predicates using interpolation along a spurious trace.

    Given: abstract trace is spurious (infeasible in concrete system).
    1. Build BMC formulas constrained by abstract trace
    2. Compute sequence interpolants
    3. Map step-indexed interpolant variables back to state variables
    4. Extract predicates from interpolants

    Returns: list of new Predicate objects (may be empty if interpolation fails)
    """
    n = len(abstract_trace)
    if n < 2:
        return []

    all_vars = concrete_ts.all_vars()

    # Build constrained BMC formulas
    formulas, step_vars = _build_constrained_bmc_formulas(
        concrete_ts, abstract_trace, predicates
    )

    # Compute sequence interpolants
    seq_result = sequence_interpolate(formulas)

    if seq_result.result != InterpolantResult.SUCCESS or not seq_result.interpolants:
        # Fallback: try unconstrained BMC formulas
        formulas2, step_vars2 = _build_bmc_formulas(concrete_ts, n)
        seq_result = sequence_interpolate(formulas2)
        step_vars = step_vars2

    if seq_result.result != InterpolantResult.SUCCESS or not seq_result.interpolants:
        return []

    # Extract predicates from each interpolant
    new_preds = []
    existing_strs = {str(p.formula) for p in predicates}

    for idx, interp in enumerate(seq_result.interpolants):
        if interp is None:
            continue

        # Map step-indexed variables back to state variables
        # Interpolant at position k uses variables from steps 0..k
        # The relevant state is step k's variables
        step_idx = min(idx, n - 1)
        reverse_map = {}
        for v in all_vars:
            step_name = f"{v}_{step_idx}"
            reverse_map[step_name] = Var(v, INT)

        mapped_interp = _substitute(interp, reverse_map)

        # Also try mapping from other steps (interpolant may use vars from
        # different steps; map all of them back to unindexed state vars)
        for s in range(n):
            for v in all_vars:
                step_name = f"{v}_{s}"
                if step_name not in reverse_map:
                    reverse_map[step_name] = Var(v, INT)

        mapped_interp = _substitute(interp, reverse_map)

        # Extract atomic predicates from the mapped interpolant
        atoms = extract_predicates_from_interpolant(mapped_interp)

        for atom in atoms:
            atom_str = str(atom)
            if atom_str not in existing_strs:
                existing_strs.add(atom_str)
                new_preds.append(Predicate(
                    name=f"itp_{idx}_{len(new_preds)}",
                    formula=atom
                ))

    return new_preds


# ---------------------------------------------------------------------------
# WP-based refinement (fallback from V010)
# ---------------------------------------------------------------------------

def _wp_refine_simple(concrete_ts, predicates, abstract_trace, infeasible_step):
    """Simple WP-based refinement as fallback when interpolation fails.

    Computes weakest precondition of each predicate through the transition
    at the infeasible step.
    """
    new_preds = []
    all_vars = concrete_ts.all_vars()
    existing_strs = {str(p.formula) for p in predicates}

    # Extract functional transition map: var -> update_expr
    trans_map = _extract_transition_map(concrete_ts)

    if not trans_map:
        return new_preds

    # WP of each predicate: P[x := f(x)]
    for pred in predicates:
        wp = _substitute(pred.formula, trans_map)
        wp_str = str(wp)
        if wp_str not in existing_strs and wp_str != str(pred.formula):
            existing_strs.add(wp_str)
            new_preds.append(Predicate(
                name=f"wp_{pred.name}",
                formula=wp
            ))

    # Add boundary predicates for each variable
    for v in concrete_ts.int_vars:
        vv = Var(v, INT)
        for bound in [0, 1, -1]:
            for op_name, op in [("ge", Op.GE), ("le", Op.LE), ("eq", Op.EQ)]:
                p = App(op, [vv, IntConst(bound)], BOOL)
                ps = str(p)
                if ps not in existing_strs:
                    existing_strs.add(ps)
                    new_preds.append(Predicate(name=f"bnd_{v}_{op_name}_{bound}", formula=p))

    return new_preds


def _extract_transition_map(concrete_ts):
    """Try to extract a functional map {var_name: update_expr} from trans_formula.

    Works for formulas of the form: x' = f(x) AND y' = g(x,y) AND ...
    """
    mapping = {}
    trans = concrete_ts.trans_formula
    if trans is None:
        return mapping

    conjuncts = []
    if isinstance(trans, App) and trans.op == Op.AND:
        conjuncts = trans.args
    else:
        conjuncts = [trans]

    for conj in conjuncts:
        if isinstance(conj, App) and conj.op == Op.EQ and len(conj.args) == 2:
            lhs, rhs = conj.args
            if isinstance(lhs, Var) and lhs.name.endswith("'"):
                base_name = lhs.name[:-1]
                if base_name in concrete_ts.all_vars():
                    mapping[base_name] = rhs
    return mapping


# ---------------------------------------------------------------------------
# Interpolation-based CEGAR main loop
# ---------------------------------------------------------------------------

def interp_cegar_check(concrete_ts, initial_predicates=None,
                       max_iterations=15, max_pdr_frames=50):
    """CEGAR loop with interpolation-based refinement.

    Like V010's cegar_check but uses Craig interpolation (V012) to discover
    new predicates from spurious counterexamples. Falls back to WP-based
    refinement when interpolation fails.

    Args:
        concrete_ts: ConcreteTS with init, trans, property formulas
        initial_predicates: Starting predicates (auto-generated if None)
        max_iterations: Maximum CEGAR iterations
        max_pdr_frames: Maximum PDR frames per iteration

    Returns: InterpCEGARResult
    """
    stats = InterpCEGARStats()

    # Auto-generate initial predicates if none provided
    if initial_predicates is None:
        predicates = auto_predicates_from_ts(concrete_ts)
    else:
        predicates = list(initial_predicates)

    # Dedup predicates by formula string
    predicates = _dedup_predicates(predicates)
    stats.predicates_initial = len(predicates)

    for iteration in range(max_iterations):
        stats.iterations = iteration + 1

        # Step 1: Build abstract system via Cartesian predicate abstraction
        if not predicates:
            # No predicates -- can't abstract. Try direct BMC check.
            bmc_result = _direct_bmc_check(concrete_ts, max_depth=max_pdr_frames)
            return InterpCEGARResult(
                verdict=bmc_result,
                predicates=predicates,
                stats=stats
            )

        abstract_ts = cartesian_abstraction(concrete_ts, predicates)

        # Step 2: Model check abstract system with PDR
        stats.pdr_calls += 1
        pdr_result = check_ts(abstract_ts, max_frames=max_pdr_frames)

        if pdr_result.result == PDRResult.SAFE:
            # Abstract system is safe -> concrete system is safe
            invariant = _concretize_invariant(pdr_result.invariant, predicates)
            stats.predicates_final = len(predicates)
            return InterpCEGARResult(
                verdict=InterpCEGARVerdict.SAFE,
                invariant=invariant,
                predicates=predicates,
                stats=stats
            )

        if pdr_result.result == PDRResult.UNKNOWN:
            stats.predicates_final = len(predicates)
            return InterpCEGARResult(
                verdict=InterpCEGARVerdict.UNKNOWN,
                predicates=predicates,
                stats=stats
            )

        # Step 3: Abstract system is unsafe -> check counterexample feasibility
        abstract_trace = _extract_abstract_trace(pdr_result, predicates)
        if not abstract_trace:
            stats.predicates_final = len(predicates)
            return InterpCEGARResult(
                verdict=InterpCEGARVerdict.UNKNOWN,
                predicates=predicates,
                stats=stats
            )

        is_feasible, infeasible_step, concrete_trace = \
            check_counterexample_feasibility(abstract_trace, concrete_ts, predicates)

        if is_feasible:
            # Real counterexample
            stats.predicates_final = len(predicates)
            return InterpCEGARResult(
                verdict=InterpCEGARVerdict.UNSAFE,
                counterexample=concrete_trace,
                predicates=predicates,
                stats=stats
            )

        # Step 4: Spurious counterexample -> refine via interpolation
        stats.spurious_traces += 1

        # Try interpolation-based refinement first
        new_preds = interpolation_refine_trace(
            concrete_ts, abstract_trace, predicates
        )

        if new_preds:
            stats.interpolation_successes += 1
            stats.predicates_from_interpolation += len(new_preds)
        else:
            # Fallback to WP-based refinement
            stats.interpolation_failures += 1
            new_preds = _wp_refine_simple(
                concrete_ts, predicates, abstract_trace, infeasible_step
            )
            if new_preds:
                stats.wp_fallbacks += 1
                stats.predicates_from_wp += len(new_preds)

        if not new_preds:
            # Refinement completely failed
            stats.predicates_final = len(predicates)
            return InterpCEGARResult(
                verdict=InterpCEGARVerdict.UNKNOWN,
                predicates=predicates,
                stats=stats
            )

        # Add new predicates and continue
        predicates.extend(new_preds)
        predicates = _dedup_predicates(predicates)

    stats.predicates_final = len(predicates)
    return InterpCEGARResult(
        verdict=InterpCEGARVerdict.UNKNOWN,
        predicates=predicates,
        stats=stats
    )


def _dedup_predicates(predicates):
    """Deduplicate predicates by formula string representation."""
    seen = set()
    result = []
    for p in predicates:
        key = str(p.formula)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def _extract_abstract_trace(pdr_result, predicates):
    """Extract abstract trace from PDR counterexample."""
    if pdr_result.counterexample is None:
        return None

    trace = pdr_result.counterexample.trace
    if not trace:
        return None

    abstract_trace = []
    for state in trace:
        abs_state = {}
        for i in range(len(predicates)):
            key = f"b_{i}"
            if key in state:
                abs_state[key] = state[key]
        abstract_trace.append(abs_state)

    return abstract_trace


def _concretize_invariant(abstract_invariant, predicates):
    """Convert abstract invariant (over b_i variables) to concrete formulas."""
    if abstract_invariant is None:
        return None

    concrete_invs = []
    for clause in abstract_invariant:
        # Each clause is a formula over b_i variables
        # Substitute b_i with predicate formulas
        concrete_clause = _substitute_pred_vars(clause, predicates)
        concrete_invs.append(concrete_clause)

    return concrete_invs


def _substitute_pred_vars(formula, predicates):
    """Substitute b_i variables with corresponding predicate formulas."""
    if isinstance(formula, Var):
        name = formula.name
        if name.startswith("b_"):
            try:
                idx = int(name[2:])
                if idx < len(predicates):
                    return predicates[idx].formula
            except ValueError:
                pass
        return formula
    if isinstance(formula, (IntConst, BoolConst)):
        return formula
    if isinstance(formula, App):
        new_args = [_substitute_pred_vars(a, predicates) for a in formula.args]
        # Handle INT-encoded booleans: (b_i == 1) -> pred, (b_i == 0) -> NOT(pred)
        if formula.op == Op.EQ and len(new_args) == 2:
            if isinstance(new_args[1], IntConst):
                if not isinstance(formula.args[0], (IntConst, BoolConst)):
                    pred_f = _substitute_pred_vars(formula.args[0], predicates)
                    if pred_f is not formula.args[0]:  # Was a b_i var
                        if new_args[1].value == 1:
                            return pred_f
                        elif new_args[1].value == 0:
                            return _not(pred_f)
        if formula.op == Op.GE and len(new_args) == 2:
            # b_i >= 1 means pred is true
            if isinstance(new_args[1], IntConst) and new_args[1].value == 1:
                if not isinstance(formula.args[0], (IntConst, BoolConst)):
                    pred_f = _substitute_pred_vars(formula.args[0], predicates)
                    if pred_f is not formula.args[0]:
                        return pred_f
        if formula.op == Op.LE and len(new_args) == 2:
            # b_i <= 0 means pred is false
            if isinstance(new_args[1], IntConst) and new_args[1].value == 0:
                if not isinstance(formula.args[0], (IntConst, BoolConst)):
                    pred_f = _substitute_pred_vars(formula.args[0], predicates)
                    if pred_f is not formula.args[0]:
                        return _not(pred_f)
        return App(formula.op, new_args, formula.sort)
    return formula


def _direct_bmc_check(concrete_ts, max_depth=20):
    """Direct bounded model check without abstraction."""
    all_vars = concrete_ts.all_vars()

    for depth in range(1, max_depth + 1):
        s = SMTSolver()
        step_vars = []
        for step in range(depth):
            sv = {}
            for v in all_vars:
                sv[v] = s.Int(f"{v}_{step}")
            step_vars.append(sv)

        # Init
        init_map = {v: step_vars[0][v] for v in all_vars}
        s.add(_substitute(concrete_ts.init_formula, init_map))

        # Transitions
        for step in range(depth - 1):
            trans_map = {}
            for v in all_vars:
                trans_map[v] = step_vars[step][v]
                trans_map[v + "'"] = step_vars[step + 1][v]
            s.add(_substitute(concrete_ts.trans_formula, trans_map))

        # Property violation at last step
        prop_map = {v: step_vars[-1][v] for v in all_vars}
        s.add(_not(_substitute(concrete_ts.prop_formula, prop_map)))

        result = s.check()
        if result == SMTResult.SAT:
            return InterpCEGARVerdict.UNSAFE

    return InterpCEGARVerdict.UNKNOWN


# ---------------------------------------------------------------------------
# Direct interpolation-based model checking (McMillan's approach)
# ---------------------------------------------------------------------------

def interp_model_check(concrete_ts, max_depth=30):
    """Direct interpolation-based model checking.

    Uses BMC + interpolation without predicate abstraction:
    1. Unroll system to depth k
    2. If BMC finds a real counterexample -> UNSAFE
    3. If BMC is UNSAT, compute interpolant between Init+Trans^{k-1} and Trans+!Prop
    4. If interpolant is subsumed by existing reachability -> SAFE (fixpoint)
    5. Otherwise, add interpolant to reachability and increase depth

    This is a simplified version of McMillan's interpolation-based model checking.
    """
    all_vars = concrete_ts.all_vars()

    # Maintain over-approximate reachability images R0, R1, ...
    # R0 = Init, Rk = img(R_{k-1}) overapproximated by interpolants
    reachability = [concrete_ts.init_formula]

    for depth in range(1, max_depth + 1):
        # BMC check at current depth
        formulas, step_vars = _build_bmc_formulas(concrete_ts, depth)

        # Check if BMC unrolling is SAT (real counterexample)
        bmc_sat = _check_conjunction_sat(formulas)
        if bmc_sat:
            return InterpCEGARResult(
                verdict=InterpCEGARVerdict.UNSAFE,
                stats=InterpCEGARStats(iterations=depth)
            )

        # BMC is UNSAT -> compute interpolant
        # Split: A = Init + Trans^(depth-1), B = last Trans + !Prop
        if len(formulas) < 2:
            continue

        # For depth 1: A = Init, B = !Prop
        # For depth k: A = Init + Trans_1 + ... + Trans_{k-1}, B = !Prop
        # Actually use sequence interpolation for all formulas
        seq_result = sequence_interpolate(formulas)

        if seq_result.result == InterpolantResult.SUCCESS and seq_result.interpolants:
            # Check fixpoint: does the first interpolant imply something we already know?
            interp_0 = seq_result.interpolants[0]
            if interp_0 is not None:
                # Map step-0 variables back to state variables
                reverse_map = {f"{v}_0": Var(v, INT) for v in all_vars}
                mapped = _substitute(interp_0, reverse_map)

                # Check if mapped is subsumed by Init (fixpoint)
                subsumed = _check_implication(concrete_ts.init_formula, mapped)
                if subsumed and depth > 1:
                    # Also check if all reachability images are stable
                    is_fixpoint = _check_reachability_fixpoint(
                        concrete_ts, reachability, seq_result.interpolants,
                        step_vars, all_vars
                    )
                    if is_fixpoint:
                        return InterpCEGARResult(
                            verdict=InterpCEGARVerdict.SAFE,
                            invariant=[mapped],
                            stats=InterpCEGARStats(iterations=depth)
                        )

                # Update reachability
                if len(reachability) <= depth:
                    reachability.append(mapped)
                else:
                    # Strengthen: take conjunction
                    reachability[depth] = _and(reachability[depth], mapped)

    return InterpCEGARResult(
        verdict=InterpCEGARVerdict.UNKNOWN,
        stats=InterpCEGARStats(iterations=max_depth)
    )


def _check_conjunction_sat(formulas):
    """Check if the conjunction of all formulas is SAT."""
    s = SMTSolver()
    all_var_names = set()
    for f in formulas:
        for name in collect_vars(f):
            all_var_names.add(name)

    for name in all_var_names:
        s.Int(name)

    for f in formulas:
        s.add(f)

    return s.check() == SMTResult.SAT


def _check_implication(a, b):
    """Check if a => b (i.e., a AND NOT(b) is UNSAT)."""
    s = SMTSolver()
    for name in collect_vars(a) | collect_vars(b):
        s.Int(name)

    s.add(a)
    s.add(_not(b))
    return s.check() == SMTResult.UNSAT


def _check_reachability_fixpoint(concrete_ts, reachability, interpolants,
                                  step_vars, all_vars):
    """Check if reachability has reached a fixpoint.

    A fixpoint is reached when the over-approximate image at step k+1
    is subsumed by the union of images at steps 0..k.
    """
    if len(interpolants) < 2:
        return False

    # Simple fixpoint check: does any interpolant repeat (imply a previous one)?
    for i in range(1, len(interpolants)):
        if interpolants[i] is None:
            continue
        # Map to state variables
        reverse_i = {f"{v}_{i}": Var(v, INT) for v in all_vars}
        mapped_i = _substitute(interpolants[i], reverse_i)

        for j in range(i):
            if interpolants[j] is None:
                continue
            reverse_j = {f"{v}_{j}": Var(v, INT) for v in all_vars}
            mapped_j = _substitute(interpolants[j], reverse_j)

            # Check if mapped_i => mapped_j (later image subsumed by earlier)
            if _check_implication(mapped_i, mapped_j):
                return True

    return False


# ---------------------------------------------------------------------------
# Source-level API
# ---------------------------------------------------------------------------

def verify_loop_interp(source, property_source, predicates=None, **kwargs):
    """Verify a property about a while loop using interpolation-based CEGAR.

    Args:
        source: C10 source with let inits + while loop
        property_source: SMT property lambda over ConcreteTS, e.g.:
            lambda ts: App(Op.GE, [ts.var('x'), IntConst(0)], BOOL)
        predicates: Initial predicates (auto-generated if None)

    Returns: InterpCEGARResult
    """
    cts = extract_loop_ts(source)
    prop = property_source(cts)
    cts.prop_formula = prop

    preds = predicates if predicates else None
    return interp_cegar_check(cts, preds, **kwargs)


def verify_loop_direct(source, property_source, max_depth=30):
    """Verify a property about a while loop using direct interpolation-based MC.

    Args:
        source: C10 source with let inits + while loop
        property_source: SMT property lambda over ConcreteTS
        max_depth: Maximum BMC unrolling depth

    Returns: InterpCEGARResult
    """
    cts = extract_loop_ts(source)
    prop = property_source(cts)
    cts.prop_formula = prop
    return interp_model_check(cts, max_depth=max_depth)


# ---------------------------------------------------------------------------
# Comparison API
# ---------------------------------------------------------------------------

def compare_refinement_strategies(concrete_ts, predicates=None, max_iterations=10):
    """Compare WP-based refinement (V010) vs interpolation-based (V014).

    Returns dict with both results and comparison statistics.
    """
    from pred_abs_cegar import cegar_check as v010_check

    init_preds = predicates if predicates else auto_predicates_from_ts(concrete_ts)

    # V010: WP-based
    v010_result = v010_check(concrete_ts, list(init_preds),
                             max_iterations=max_iterations)

    # V014: Interpolation-based
    v014_result = interp_cegar_check(concrete_ts, list(init_preds),
                                     max_iterations=max_iterations)

    return {
        'wp_based': {
            'verdict': v010_result.verdict.value,
            'iterations': v010_result.stats.iterations,
            'predicates_final': v010_result.stats.predicates_final,
        },
        'interpolation_based': {
            'verdict': v014_result.verdict.value,
            'iterations': v014_result.stats.iterations,
            'predicates_final': v014_result.stats.predicates_final,
            'interp_successes': v014_result.stats.interpolation_successes,
            'interp_failures': v014_result.stats.interpolation_failures,
            'wp_fallbacks': v014_result.stats.wp_fallbacks,
        },
        'v010_result': v010_result,
        'v014_result': v014_result,
    }
