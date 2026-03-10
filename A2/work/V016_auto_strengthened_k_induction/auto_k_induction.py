"""
V016: Auto-Strengthened k-Induction
====================================
Composes:
  - V015 k-Induction (base case + inductive step)
  - V007 Invariant Inference (tiered candidate generation)
  - V002 PDR (transition systems, comparison)
  - C037 SMT solver
  - C010 Parser

When plain k-induction fails (inductive step doesn't go through), this module
automatically infers strengthening invariants via V007's tiered strategy and
retries. This closes the gap between k-induction's simplicity and the need
for auxiliary invariants in non-trivial loops.

Pipeline:
  1. Try plain k-induction for k=0..max_k
  2. If UNKNOWN: extract guarded TS, infer invariants (abstract interp,
     condition weakening, relational templates, PDR)
  3. Retry k-induction with inferred invariants as strengthening
  4. If still UNKNOWN: try incremental strengthening subsets

APIs:
  - auto_k_induction(ts, max_k) -> AutoKIndResult
  - verify_loop_auto(source, property) -> AutoKIndResult
  - verify_loop_auto_with_hints(source, property, hints) -> AutoKIndResult
  - compare_strategies(ts) -> comparison dict
"""

import sys
import os
import time
import itertools

_base = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, os.path.join(_base, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_base, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_base, 'challenges', 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V007_invariant_inference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V005_ai_strengthened_pdr'))

from smt_solver import (
    SMTSolver, SMTResult, Op, Var, IntConst, BoolConst, App, Sort, INT, BOOL,
)
from pdr import TransitionSystem, check_ts, PDRResult

from k_induction import (
    KIndResult, k_induction_check, incremental_k_induction,
    k_induction_with_strengthening, check_base_case, check_inductive_step,
    bmc_check, _negate, _extract_loop_ts, _step_vars, _substitute,
    _apply_formula_at_step, _apply_trans_at_step,
)

from invariant_inference import (
    infer_loop_invariants, _build_guarded_ts, _sexpr_to_ts_smt,
    smt_to_sexpr, InferenceResult, InferredInvariant, InferenceMethod,
    _generate_abstract_candidates, _generate_init_and_bound_candidates,
    _generate_condition_invariants, _generate_relational_invariants,
    _generate_pdr_invariants, _extract_init_values,
    _try_validate_and_add,
)

from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp,
    s_and, s_or, s_not, s_implies,
    ast_to_sexpr, parse,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class AutoKIndResult:
    """Result of auto-strengthened k-induction."""
    def __init__(self, result, k=None, counterexample=None, invariants=None,
                 strengthening=None, stats=None):
        self.result = result                  # "SAFE", "UNSAFE", "UNKNOWN"
        self.k = k                            # k value used
        self.counterexample = counterexample  # list of states if UNSAFE
        self.invariants = invariants or []    # InferredInvariant list
        self.strengthening = strengthening    # combined SMT term if used
        self.stats = stats or {}

    def __repr__(self):
        inv_count = len(self.invariants)
        return f"AutoKIndResult({self.result}, k={self.k}, invariants={inv_count})"


# ---------------------------------------------------------------------------
# Core: auto-strengthened k-induction on a TransitionSystem
# ---------------------------------------------------------------------------

def auto_k_induction(ts, max_k=20, source=None, property_sexpr=None):
    """Auto-strengthened k-induction.

    1. Try plain k-induction
    2. If UNKNOWN: infer invariants and retry with strengthening
    3. If still UNKNOWN: try subsets of invariants

    Args:
        ts: TransitionSystem with property set
        max_k: maximum induction depth
        source: optional C10 source (enables richer invariant inference)
        property_sexpr: optional property as SExpr (for V007 postcondition)

    Returns:
        AutoKIndResult
    """
    start = time.time()
    stats = {
        "phase1_plain": None,
        "phase2_strengthened": None,
        "invariants_inferred": 0,
        "invariants_used": 0,
    }

    # Phase 1: Plain k-induction
    plain_result = incremental_k_induction(ts, max_k)
    stats["phase1_plain"] = plain_result.result

    if plain_result.result in ("SAFE", "UNSAFE"):
        elapsed = time.time() - start
        stats["time"] = elapsed
        return AutoKIndResult(
            result=plain_result.result,
            k=plain_result.k,
            counterexample=plain_result.counterexample,
            stats=stats,
        )

    # Phase 2: Infer invariants and retry
    invariants = _infer_strengthening_invariants(ts, source, property_sexpr)
    stats["invariants_inferred"] = len(invariants)

    if not invariants:
        elapsed = time.time() - start
        stats["time"] = elapsed
        return AutoKIndResult(result="UNKNOWN", k=max_k, stats=stats)

    # Try all invariants together
    smt_invs = _invariants_to_smt(invariants, ts)
    if smt_invs:
        strengthened_result = k_induction_with_strengthening(ts, max_k, invariants=smt_invs)
        stats["phase2_strengthened"] = strengthened_result.result
        stats["invariants_used"] = len(smt_invs)

        if strengthened_result.result == "SAFE":
            elapsed = time.time() - start
            stats["time"] = elapsed
            return AutoKIndResult(
                result="SAFE",
                k=strengthened_result.k,
                invariants=invariants,
                strengthening=_combine_smt(smt_invs),
                stats=stats,
            )

        if strengthened_result.result == "UNSAFE":
            elapsed = time.time() - start
            stats["time"] = elapsed
            return AutoKIndResult(
                result="UNSAFE",
                k=strengthened_result.k,
                counterexample=strengthened_result.counterexample,
                invariants=invariants,
                stats=stats,
            )

    # Phase 3: Try subsets (some invariants may conflict or be too strong)
    if len(smt_invs) > 1:
        subset_result = _try_invariant_subsets(ts, max_k, invariants, smt_invs, stats)
        if subset_result is not None:
            elapsed = time.time() - start
            stats["time"] = elapsed
            return subset_result

    elapsed = time.time() - start
    stats["time"] = elapsed
    return AutoKIndResult(
        result="UNKNOWN",
        k=max_k,
        invariants=invariants,
        stats=stats,
    )


def _infer_strengthening_invariants(ts, source, property_sexpr):
    """Infer candidate strengthening invariants for k-induction."""
    invariants = []
    seen = set()

    # If we have source, use V007's full inference pipeline
    if source is not None:
        try:
            inf_result = infer_loop_invariants(source, loop_index=0, postcondition=property_sexpr)
            for inv in inf_result.invariants:
                key = str(inv.sexpr)
                if key not in seen:
                    seen.add(key)
                    invariants.append(inv)
        except Exception:
            pass

    # Also try direct TS-level inference (works without source)
    ts_invs = _infer_from_ts(ts)
    for inv in ts_invs:
        key = str(inv.sexpr)
        if key not in seen:
            seen.add(key)
            invariants.append(inv)

    return invariants


def _infer_from_ts(ts):
    """Infer invariants directly from the transition system (no source needed).

    Strategies:
    - Init value bounds: extract x=c from init, try x>=c, x<=c
    - Property weakening: if prop is x>=0, try x>=-1, etc.
    - Variable non-negativity: try x>=0 for each var
    """
    results = []
    state_var_names = [name for name, _ in ts.state_vars]

    # Extract init values
    init_vals = {}
    _extract_init_from_ts(ts.init_formula, init_vals)

    for name in state_var_names:
        v = ts.var(name)

        # Non-negativity: x >= 0
        candidate = App(Op.GE, [v, IntConst(0)], BOOL)
        if _validate_ts_invariant(ts, candidate):
            results.append(InferredInvariant(
                sexpr=SBinOp('>=', SVar(name), SInt(0)),
                method=InferenceMethod.ABSTRACT_INTERP,
                description=f"ts_nonneg({name}>=0)",
                is_inductive=True,
            ))

        if name in init_vals:
            val = init_vals[name]

            # Upper bound from init
            candidate = App(Op.LE, [v, IntConst(val)], BOOL)
            if _validate_ts_invariant(ts, candidate):
                results.append(InferredInvariant(
                    sexpr=SBinOp('<=', SVar(name), SInt(val)),
                    method=InferenceMethod.ABSTRACT_INTERP,
                    description=f"ts_upper({name}<={val})",
                    is_inductive=True,
                ))

            # Lower bound from init
            candidate = App(Op.GE, [v, IntConst(val)], BOOL)
            if _validate_ts_invariant(ts, candidate):
                results.append(InferredInvariant(
                    sexpr=SBinOp('>=', SVar(name), SInt(val)),
                    method=InferenceMethod.ABSTRACT_INTERP,
                    description=f"ts_lower({name}>={val})",
                    is_inductive=True,
                ))

    # Relational: sum/diff conservation
    if len(state_var_names) >= 2:
        for n1, n2 in itertools.combinations(state_var_names, 2):
            if n1 in init_vals and n2 in init_vals:
                v1 = ts.var(n1)
                v2 = ts.var(n2)

                # Sum conservation
                c_sum = init_vals[n1] + init_vals[n2]
                candidate = App(Op.EQ, [App(Op.ADD, [v1, v2], INT), IntConst(c_sum)], BOOL)
                if _validate_ts_invariant(ts, candidate):
                    results.append(InferredInvariant(
                        sexpr=SBinOp('==', SBinOp('+', SVar(n1), SVar(n2)), SInt(c_sum)),
                        method=InferenceMethod.RELATIONAL_TEMPLATE,
                        description=f"ts_sum({n1}+{n2}=={c_sum})",
                        is_inductive=True,
                    ))

                # Diff conservation
                c_diff = init_vals[n1] - init_vals[n2]
                candidate = App(Op.EQ, [App(Op.SUB, [v1, v2], INT), IntConst(c_diff)], BOOL)
                if _validate_ts_invariant(ts, candidate):
                    results.append(InferredInvariant(
                        sexpr=SBinOp('==', SBinOp('-', SVar(n1), SVar(n2)), SInt(c_diff)),
                        method=InferenceMethod.RELATIONAL_TEMPLATE,
                        description=f"ts_diff({n1}-{n2}=={c_diff})",
                        is_inductive=True,
                    ))

    return results


def _extract_init_from_ts(formula, vals):
    """Extract var=const from init formula."""
    if isinstance(formula, App):
        if formula.op == Op.EQ and len(formula.args) == 2:
            lhs, rhs = formula.args
            if isinstance(lhs, Var) and isinstance(rhs, IntConst):
                vals[lhs.name] = rhs.value
            elif isinstance(rhs, Var) and isinstance(lhs, IntConst):
                vals[rhs.name] = lhs.value
        elif formula.op == Op.AND:
            for arg in formula.args:
                _extract_init_from_ts(arg, vals)


def _validate_ts_invariant(ts, candidate):
    """Check if candidate is an inductive invariant of the TS.

    Two checks:
    1. Init => candidate
    2. candidate AND Trans => candidate[x/x']
    """
    state_var_names = [name for name, _ in ts.state_vars]

    # Check 1: Init => candidate
    s = SMTSolver()
    for name, sort in ts.state_vars:
        if sort == INT:
            s.Int(name)
        else:
            s.Bool(name)
    s.add(ts.init_formula)
    s.add(_negate(candidate))
    if s.check() != SMTResult.UNSAT:
        return False

    # Check 2: candidate AND Trans => candidate'
    s2 = SMTSolver()
    for name, sort in ts.state_vars:
        if sort == INT:
            s2.Int(name)
            s2.Int(name + "'")
        else:
            s2.Bool(name)
            s2.Bool(name + "'")
    s2.add(candidate)
    s2.add(ts.trans_formula)

    # Substitute primed vars in candidate
    prime_map = {}
    for name, sort in ts.state_vars:
        prime_map[name] = Var(name + "'", sort)
    candidate_prime = _substitute_vars(candidate, prime_map)
    s2.add(_negate(candidate_prime))

    return s2.check() == SMTResult.UNSAT


def _substitute_vars(formula, var_map):
    """Substitute Var nodes by name."""
    if isinstance(formula, Var):
        return var_map.get(formula.name, formula)
    elif isinstance(formula, (IntConst, BoolConst)):
        return formula
    elif isinstance(formula, App):
        new_args = [_substitute_vars(a, var_map) for a in formula.args]
        return App(formula.op, new_args, formula.sort)
    return formula


def _invariants_to_smt(invariants, ts):
    """Convert InferredInvariant list to SMT terms for the given TS."""
    state_var_names = [name for name, _ in ts.state_vars]
    smt_terms = []
    for inv in invariants:
        try:
            term = _sexpr_to_ts_smt(inv.sexpr, ts, state_var_names)
            if term is not None:
                smt_terms.append(term)
        except Exception:
            pass
    return smt_terms


def _combine_smt(terms):
    """Combine SMT terms into a conjunction."""
    if not terms:
        return BoolConst(True)
    if len(terms) == 1:
        return terms[0]
    return App(Op.AND, terms, BOOL)


def _try_invariant_subsets(ts, max_k, invariants, smt_invs, stats):
    """Try subsets of invariants if all-together doesn't work.

    Strategy: try removing one invariant at a time (leave-one-out),
    then try each invariant individually.
    """
    # Leave-one-out
    for i in range(len(smt_invs)):
        subset = smt_invs[:i] + smt_invs[i+1:]
        if not subset:
            continue
        result = k_induction_with_strengthening(ts, max_k, invariants=subset)
        if result.result == "SAFE":
            used = invariants[:i] + invariants[i+1:]
            stats["phase3_subset"] = f"leave-out-{i}"
            stats["invariants_used"] = len(subset)
            return AutoKIndResult(
                result="SAFE",
                k=result.k,
                invariants=used,
                strengthening=_combine_smt(subset),
                stats=stats,
            )

    # Individual invariants
    for i, smt_inv in enumerate(smt_invs):
        result = k_induction_with_strengthening(ts, max_k, invariants=[smt_inv])
        if result.result == "SAFE":
            stats["phase3_subset"] = f"single-{i}"
            stats["invariants_used"] = 1
            return AutoKIndResult(
                result="SAFE",
                k=result.k,
                invariants=[invariants[i]],
                strengthening=smt_inv,
                stats=stats,
            )

    return None


# ---------------------------------------------------------------------------
# Source-level API
# ---------------------------------------------------------------------------

def verify_loop_auto(source, property_source, max_k=20):
    """Verify a loop property with auto-strengthened k-induction.

    Args:
        source: C10 source with a while loop
        property_source: property expression string (e.g., "x >= 0")
        max_k: maximum induction depth

    Returns:
        AutoKIndResult
    """
    from stack_vm import lex, Parser, LetDecl, BinOp, IntLit, Var as ASTVar

    ts, ts_vars = _extract_loop_ts(source)

    # Parse property
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)

    # Convert property to SExpr for V007
    prop_sexpr = _parse_property_sexpr(property_source)

    return auto_k_induction(ts, max_k, source=source, property_sexpr=prop_sexpr)


def verify_loop_auto_with_hints(source, property_source, hint_sources, max_k=20):
    """Verify with user-provided hints combined with auto-inference.

    Hints are extra invariant candidates that are validated and added
    to the auto-inferred set.

    Args:
        source: C10 source with a while loop
        property_source: property expression string
        hint_sources: list of invariant expression strings
        max_k: maximum k

    Returns:
        AutoKIndResult
    """
    from stack_vm import lex, Parser, LetDecl, BinOp, IntLit, Var as ASTVar

    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)

    prop_sexpr = _parse_property_sexpr(property_source)

    # Parse hints to SMT
    hint_smts = []
    hint_invs = []
    for h in hint_sources:
        h_smt = _parse_property(h, ts_vars)
        if _validate_ts_invariant(ts, h_smt):
            hint_smts.append(h_smt)
            hint_invs.append(InferredInvariant(
                sexpr=_parse_property_sexpr(h),
                method=InferenceMethod.CONDITION_BASED,
                description=f"hint: {h}",
                is_inductive=True,
            ))

    # Try with just hints first
    if hint_smts:
        result = k_induction_with_strengthening(ts, max_k, invariants=hint_smts)
        if result.result == "SAFE":
            return AutoKIndResult(
                result="SAFE",
                k=result.k,
                invariants=hint_invs,
                strengthening=_combine_smt(hint_smts),
                stats={"hints_used": len(hint_smts), "auto_inferred": False},
            )

    # Fall back to full auto
    full_result = auto_k_induction(ts, max_k, source=source, property_sexpr=prop_sexpr)

    # Merge hints with auto-inferred
    if full_result.result == "UNKNOWN" and hint_smts:
        all_invs = hint_invs + (full_result.invariants or [])
        all_smts = hint_smts + _invariants_to_smt(full_result.invariants or [], ts)
        if all_smts:
            combined_result = k_induction_with_strengthening(ts, max_k, invariants=all_smts)
            if combined_result.result == "SAFE":
                return AutoKIndResult(
                    result="SAFE",
                    k=combined_result.k,
                    invariants=all_invs,
                    strengthening=_combine_smt(all_smts),
                    stats={"hints_used": len(hint_smts),
                           "auto_inferred": len(full_result.invariants or [])},
                )

    return full_result


def _parse_property(expr_str, ts_vars):
    """Parse a property string to SMT using TS variables."""
    from stack_vm import lex, Parser, LetDecl, BinOp, IntLit, Var as ASTVar

    tokens = lex(f"let __p = ({expr_str});")
    stmts = Parser(tokens).parse().stmts
    expr = stmts[0].value

    def to_smt(e):
        if isinstance(e, IntLit):
            return IntConst(e.value)
        elif isinstance(e, ASTVar):
            if e.name in ts_vars:
                return ts_vars[e.name]
            return IntConst(0)
        elif isinstance(e, BinOp):
            op_map = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
                      '<': Op.LT, '>': Op.GT, '<=': Op.LE, '>=': Op.GE,
                      '==': Op.EQ, '!=': Op.NEQ}
            l = to_smt(e.left)
            r = to_smt(e.right)
            op = op_map.get(e.op)
            if op is None:
                raise ValueError(f"Unknown op: {e.op}")
            sort = BOOL if op in (Op.LT, Op.GT, Op.LE, Op.GE, Op.EQ, Op.NEQ) else INT
            return App(op, [l, r], sort)
        return IntConst(0)

    return to_smt(expr)


def _parse_property_sexpr(expr_str):
    """Parse a property string to SExpr."""
    from stack_vm import lex, Parser
    tokens = lex(f"let __p = ({expr_str});")
    stmts = Parser(tokens).parse().stmts
    return ast_to_sexpr(stmts[0].value)


# ---------------------------------------------------------------------------
# Comparison API
# ---------------------------------------------------------------------------

def compare_strategies(ts, max_k=20, source=None):
    """Compare plain k-induction, auto-strengthened k-induction, and PDR.

    Returns dict with results and timing for each approach.
    """
    results = {}

    # Plain k-induction
    start = time.time()
    plain = incremental_k_induction(ts, max_k)
    results["plain_k_induction"] = {
        "result": plain.result,
        "k": plain.k,
        "time": time.time() - start,
    }

    # Auto-strengthened k-induction
    start = time.time()
    auto = auto_k_induction(ts, max_k, source=source)
    results["auto_k_induction"] = {
        "result": auto.result,
        "k": auto.k,
        "invariants": len(auto.invariants),
        "time": time.time() - start,
    }

    # PDR
    start = time.time()
    try:
        pdr_out = check_ts(ts, max_frames=50)
        results["pdr"] = {
            "result": pdr_out.result.value.upper(),
            "time": time.time() - start,
        }
    except Exception as e:
        results["pdr"] = {
            "result": f"ERROR: {e}",
            "time": time.time() - start,
        }

    return results


def compare_with_source(source, property_source, max_k=20):
    """Compare strategies on a source-level loop + property."""
    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)
    return compare_strategies(ts, max_k, source=source)
