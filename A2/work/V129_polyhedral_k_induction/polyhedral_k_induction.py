"""V129: Polyhedral k-Induction.

Composes V105 (polyhedral abstract interpretation) + V015 (k-induction) + V016 (auto k-induction).

The polyhedral domain provides RELATIONAL constraints (e.g., x + y <= 10) in addition
to per-variable bounds. These are richer than interval-only candidates and can strengthen
k-induction proofs that require relational invariants.

Pipeline:
  Phase 1: Plain k-induction (quick check)
  Phase 2: Polyhedral-strengthened k-induction
    - Run PolyhedralInterpreter on source -> PolyhedralDomain env
    - Extract interval bounds + relational constraints as candidates
    - Convert LinearConstraint -> SMT App formulas
    - Validate inductively, then strengthen k-induction
  Phase 3: V016 auto k-induction fallback
  Phase 4: Combined (polyhedral + auto invariants)
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V016_auto_strengthened_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))

from polyhedral_domain import PolyhedralInterpreter, PolyhedralDomain, LinearConstraint
from stack_vm import WhileStmt
from k_induction import (
    incremental_k_induction, k_induction_with_strengthening,
    KIndResult, _extract_loop_ts,
)
from auto_k_induction import (
    auto_k_induction, _parse_property,
    _validate_ts_invariant, _invariants_to_smt,
)
from pdr import TransitionSystem
from smt_solver import SMTSolver, SMTResult, Var, IntConst, BoolConst, App, Op, INT, BOOL


# ---------------------------------------------------------------------------
# Polyhedral interpreter that captures loop invariants (fixpoints)
# ---------------------------------------------------------------------------

class InvariantCapturingInterpreter(PolyhedralInterpreter):
    """Subclass that captures the widened fixpoint (loop invariant) during analysis."""

    def __init__(self, max_iterations=50):
        super().__init__(max_iterations)
        self.loop_invariants = []  # List of PolyhedralDomain (one per while loop)

    def _interpret_while(self, stmt, env, functions):
        """Override to capture the fixpoint state before exit refinement."""
        current = env.copy()

        for i in range(self.max_iterations):
            then_env, _ = self._refine_condition(stmt.cond, current)
            body_env = self._interpret_block(stmt.body, then_env, functions)

            if body_env.is_bot():
                break

            joined = current.join(body_env)
            next_env = current.widen(joined)

            if next_env.equals(current):
                break
            current = next_env

        # Capture the fixpoint (loop invariant) BEFORE exit refinement
        self.loop_invariants.append(current.copy())

        _, exit_env = self._refine_condition(stmt.cond, current)
        return exit_env


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PolyhedralCandidate:
    """An invariant candidate extracted from polyhedral analysis."""
    formula: App
    description: str
    source_kind: str   # "interval_lower", "interval_upper", "relational", "equality"
    variables: List[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class PolyhedralKIndResult:
    """Result of polyhedral k-induction verification."""
    result: str  # "SAFE", "UNSAFE", "UNKNOWN"
    k: Optional[int] = None
    counterexample: Optional[list] = None
    polyhedral_candidates: List[PolyhedralCandidate] = field(default_factory=list)
    used_candidates: List[PolyhedralCandidate] = field(default_factory=list)
    strengthening: Optional[App] = None
    stats: dict = field(default_factory=dict)
    polyhedral_env: Optional[PolyhedralDomain] = None

    def __repr__(self):
        return f"PolyhedralKIndResult(result={self.result!r}, k={self.k})"


# ---------------------------------------------------------------------------
# Candidate extraction from polyhedral domain
# ---------------------------------------------------------------------------

def _constraint_to_smt(lc: LinearConstraint, ts: TransitionSystem) -> Optional[App]:
    """Convert a LinearConstraint to an SMT formula over TS variables.

    LinearConstraint: sum(coeffs[var] * var) <= bound  (or == if is_equality)
    """
    cd = lc.coeffs_dict
    var_names = set()
    for vn in cd:
        var_names.add(vn)

    # Check all variables exist in TS
    ts_var_names = {name for name, _ in ts.state_vars}
    for vn in var_names:
        if vn not in ts_var_names:
            return None

    # Build SMT LHS: sum of coeff * var
    terms = []
    for vn, coeff in sorted(cd.items()):
        c = int(coeff)
        v = Var(vn, INT)
        if c == 1:
            terms.append(v)
        elif c == -1:
            terms.append(App(Op.SUB, [IntConst(0), v], INT))
        elif c == 0:
            continue
        else:
            terms.append(App(Op.MUL, [IntConst(c), v], INT))

    bound_val = int(lc.bound)

    if not terms:
        # Constant constraint: 0 <= bound or 0 == bound
        if lc.is_equality:
            return BoolConst(0 == bound_val)
        else:
            return BoolConst(0 <= bound_val)

    # Sum terms
    lhs = terms[0]
    for t in terms[1:]:
        lhs = App(Op.ADD, [lhs, t], INT)

    rhs = IntConst(bound_val)

    if lc.is_equality:
        return App(Op.EQ, [lhs, rhs], BOOL)
    else:
        return App(Op.LE, [lhs, rhs], BOOL)


def _extract_interval_candidates(env: PolyhedralDomain, ts: TransitionSystem) -> List[PolyhedralCandidate]:
    """Extract per-variable interval bound candidates."""
    candidates = []
    ts_var_names = {name for name, _ in ts.state_vars}

    for vname in env.var_names:
        if vname not in ts_var_names:
            continue
        lo, hi = env.get_interval(vname)
        v = Var(vname, INT)

        if lo > float('-inf'):
            lo_int = int(lo)
            formula = App(Op.GE, [v, IntConst(lo_int)], BOOL)
            candidates.append(PolyhedralCandidate(
                formula=formula,
                description=f"{vname} >= {lo_int}",
                source_kind="interval_lower",
                variables=[vname],
                priority=7,
            ))

        if hi < float('inf'):
            hi_int = int(hi)
            formula = App(Op.LE, [v, IntConst(hi_int)], BOOL)
            candidates.append(PolyhedralCandidate(
                formula=formula,
                description=f"{vname} <= {hi_int}",
                source_kind="interval_upper",
                variables=[vname],
                priority=7,
            ))

    return candidates


def _extract_relational_candidates(env: PolyhedralDomain, ts: TransitionSystem) -> List[PolyhedralCandidate]:
    """Extract relational constraints (involving 2+ variables) as candidates."""
    candidates = []
    ts_var_names = {name for name, _ in ts.state_vars}

    for lc in env.constraints:
        # Only relational constraints (2+ variables)
        involved_vars = [vn for vn in lc.coeffs_dict if lc.coeffs_dict[vn] != 0]
        if len(involved_vars) < 2:
            continue
        # All vars must be in TS
        if not all(v in ts_var_names for v in involved_vars):
            continue

        formula = _constraint_to_smt(lc, ts)
        if formula is None:
            continue

        desc = _constraint_description(lc)
        candidates.append(PolyhedralCandidate(
            formula=formula,
            description=desc,
            source_kind="equality" if lc.is_equality else "relational",
            variables=involved_vars,
            priority=9 if lc.is_equality else 8,
        ))

    return candidates


def _extract_all_constraint_candidates(env: PolyhedralDomain, ts: TransitionSystem) -> List[PolyhedralCandidate]:
    """Extract ALL constraints (including single-variable) directly from the domain."""
    candidates = []
    ts_var_names = {name for name, _ in ts.state_vars}

    for lc in env.constraints:
        involved_vars = [vn for vn in lc.coeffs_dict if lc.coeffs_dict[vn] != 0]
        if not all(v in ts_var_names for v in involved_vars):
            continue
        formula = _constraint_to_smt(lc, ts)
        if formula is None:
            continue
        desc = _constraint_description(lc)
        kind = "equality" if lc.is_equality else ("relational" if len(involved_vars) >= 2 else "interval_constraint")
        candidates.append(PolyhedralCandidate(
            formula=formula,
            description=desc,
            source_kind=kind,
            variables=involved_vars,
            priority=9 if lc.is_equality else (8 if len(involved_vars) >= 2 else 6),
        ))

    return candidates


def _constraint_description(lc: LinearConstraint) -> str:
    """Human-readable description of a LinearConstraint."""
    parts = []
    for vn, coeff in sorted(lc.coeffs_dict.items()):
        c = int(coeff)
        if c == 0:
            continue
        if c == 1:
            parts.append(vn)
        elif c == -1:
            parts.append(f"-{vn}")
        else:
            parts.append(f"{c}*{vn}")
    lhs = " + ".join(parts) if parts else "0"
    op = "==" if lc.is_equality else "<="
    return f"{lhs} {op} {int(lc.bound)}"


# ---------------------------------------------------------------------------
# Candidate validation & selection
# ---------------------------------------------------------------------------

def _validate_candidate(ts: TransitionSystem, candidate: PolyhedralCandidate) -> bool:
    """Check if candidate is an inductive invariant for ts."""
    return _validate_ts_invariant(ts, candidate.formula)


def _select_best_candidates(
    candidates: List[PolyhedralCandidate],
    ts: TransitionSystem,
    max_candidates: int = 15,
) -> List[PolyhedralCandidate]:
    """Rank, deduplicate, validate, and return top candidates."""
    # Sort by priority (descending)
    sorted_cands = sorted(candidates, key=lambda c: -c.priority)

    # Deduplicate by description
    seen = set()
    unique = []
    for c in sorted_cands:
        if c.description not in seen:
            seen.add(c.description)
            unique.append(c)

    # Validate inductively
    valid = []
    for c in unique[:max_candidates * 2]:
        if _validate_candidate(ts, c):
            valid.append(c)
            if len(valid) >= max_candidates:
                break

    return valid


def _combine_smt(terms: List[App]) -> Optional[App]:
    """Conjoin multiple SMT formulas with AND."""
    if not terms:
        return None
    result = terms[0]
    for t in terms[1:]:
        result = App(Op.AND, [result, t], BOOL)
    return result


# ---------------------------------------------------------------------------
# Core verification engine
# ---------------------------------------------------------------------------

def polyhedral_k_induction(
    ts: TransitionSystem,
    source: str,
    max_k: int = 20,
    max_iterations: int = 50,
) -> PolyhedralKIndResult:
    """Main polyhedral k-induction verifier (4-phase pipeline).

    Args:
        ts: Transition system with property set
        source: C10 source code for polyhedral analysis
        max_k: Maximum k for k-induction
        max_iterations: Max iterations for polyhedral abstract interpreter

    Returns:
        PolyhedralKIndResult
    """
    t0 = time.time()
    stats = {}
    all_candidates = []
    used = []
    poly_env = None

    # Phase 1: Plain k-induction
    r1 = incremental_k_induction(ts, max_k)
    stats["phase1_plain"] = r1.result
    if r1.result == "SAFE":
        return PolyhedralKIndResult(
            result="SAFE", k=r1.k, stats={**stats, "time": time.time() - t0},
        )
    if r1.result == "UNSAFE":
        return PolyhedralKIndResult(
            result="UNSAFE", k=r1.k, counterexample=r1.counterexample,
            stats={**stats, "time": time.time() - t0},
        )

    # Phase 2: Polyhedral-strengthened k-induction
    try:
        interp = InvariantCapturingInterpreter(max_iterations=max_iterations)
        analysis = interp.analyze(source)
        poly_env = analysis["env"]

        # Use loop invariants (fixpoints) for candidate extraction, not post-loop state
        # Loop invariants hold DURING the loop -- these are the inductive invariants
        invariant_envs = interp.loop_invariants

        # Extract candidates from each loop invariant + post-loop env
        interval_cands = []
        relational_cands = []
        all_constraint_cands = []

        for inv_env in invariant_envs:
            interval_cands.extend(_extract_interval_candidates(inv_env, ts))
            relational_cands.extend(_extract_relational_candidates(inv_env, ts))
            all_constraint_cands.extend(_extract_all_constraint_candidates(inv_env, ts))

        # Merge (deduplicate later in _select_best_candidates)
        all_candidates = interval_cands + relational_cands + all_constraint_cands

        stats["candidates_interval"] = len(interval_cands)
        stats["candidates_relational"] = len(relational_cands)
        stats["candidates_all_constraints"] = len(all_constraint_cands)
        stats["candidates_raw"] = len(all_candidates)

        valid_candidates = _select_best_candidates(all_candidates, ts)
        stats["candidates_valid"] = len(valid_candidates)

        if valid_candidates:
            inv_formulas = [c.formula for c in valid_candidates]
            r2 = k_induction_with_strengthening(ts, max_k, invariants=inv_formulas)
            stats["phase2_polyhedral"] = r2.result

            if r2.result == "SAFE":
                used = valid_candidates
                return PolyhedralKIndResult(
                    result="SAFE", k=r2.k,
                    polyhedral_candidates=all_candidates,
                    used_candidates=used,
                    strengthening=_combine_smt(inv_formulas),
                    stats={**stats, "time": time.time() - t0},
                    polyhedral_env=poly_env,
                )

            # Try subsets (leave-one-out)
            if len(valid_candidates) > 1:
                for i in range(len(valid_candidates)):
                    subset = [c for j, c in enumerate(valid_candidates) if j != i]
                    sub_formulas = [c.formula for c in subset]
                    rs = k_induction_with_strengthening(ts, max_k, invariants=sub_formulas)
                    if rs.result == "SAFE":
                        used = subset
                        stats["phase2_subset"] = f"SAFE (dropped {i})"
                        return PolyhedralKIndResult(
                            result="SAFE", k=rs.k,
                            polyhedral_candidates=all_candidates,
                            used_candidates=used,
                            strengthening=_combine_smt(sub_formulas),
                            stats={**stats, "time": time.time() - t0},
                            polyhedral_env=poly_env,
                        )
        else:
            stats["phase2_polyhedral"] = "NO_CANDIDATES"

    except Exception as e:
        stats["phase2_error"] = str(e)

    # Phase 3: V016 auto k-induction fallback
    try:
        r3 = auto_k_induction(ts, max_k, source=source)
        stats["phase3_auto"] = r3.result
        if r3.result == "SAFE":
            return PolyhedralKIndResult(
                result="SAFE", k=r3.k,
                polyhedral_candidates=all_candidates,
                stats={**stats, "time": time.time() - t0},
                polyhedral_env=poly_env,
            )
        if r3.result == "UNSAFE":
            return PolyhedralKIndResult(
                result="UNSAFE", k=r3.k, counterexample=r3.counterexample,
                polyhedral_candidates=all_candidates,
                stats={**stats, "time": time.time() - t0},
                polyhedral_env=poly_env,
            )
    except Exception as e:
        stats["phase3_error"] = str(e)

    # Phase 4: Combined (polyhedral + auto invariants)
    try:
        valid_candidates = _select_best_candidates(all_candidates, ts) if all_candidates else []
        auto_invs = []
        if r3 and hasattr(r3, 'invariants') and r3.invariants:
            auto_invs = _invariants_to_smt(r3.invariants, ts)

        if valid_candidates or auto_invs:
            combined = [c.formula for c in valid_candidates] + auto_invs
            r4 = k_induction_with_strengthening(ts, max_k, invariants=combined)
            stats["phase4_combined"] = r4.result
            if r4.result == "SAFE":
                used = valid_candidates
                return PolyhedralKIndResult(
                    result="SAFE", k=r4.k,
                    polyhedral_candidates=all_candidates,
                    used_candidates=used,
                    strengthening=_combine_smt(combined),
                    stats={**stats, "time": time.time() - t0},
                    polyhedral_env=poly_env,
                )
    except Exception:
        pass

    return PolyhedralKIndResult(
        result="UNKNOWN",
        polyhedral_candidates=all_candidates,
        stats={**stats, "time": time.time() - t0},
        polyhedral_env=poly_env,
    )


# ---------------------------------------------------------------------------
# Source-level APIs
# ---------------------------------------------------------------------------

def verify_loop_polyhedral(source: str, property_source: str, max_k: int = 20) -> PolyhedralKIndResult:
    """Verify a loop property using polyhedral k-induction.

    Args:
        source: C10 source code with a while loop
        property_source: Property string (e.g., "x >= 0")
        max_k: Maximum k for k-induction

    Returns:
        PolyhedralKIndResult
    """
    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)
    return polyhedral_k_induction(ts, source, max_k)


def verify_loop_polyhedral_with_config(
    source: str, property_source: str,
    max_k: int = 20, max_iterations: int = 50,
) -> PolyhedralKIndResult:
    """Verify with custom polyhedral analysis settings."""
    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)
    return polyhedral_k_induction(ts, source, max_k, max_iterations=max_iterations)


def get_polyhedral_candidates(source: str, property_source: str) -> List[PolyhedralCandidate]:
    """Extract and validate polyhedral candidates (inspection API)."""
    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)

    interp = InvariantCapturingInterpreter(max_iterations=50)
    interp.analyze(source)

    all_cands = []
    for inv_env in interp.loop_invariants:
        all_cands.extend(_extract_interval_candidates(inv_env, ts))
        all_cands.extend(_extract_relational_candidates(inv_env, ts))

    return _select_best_candidates(all_cands, ts)


def get_polyhedral_env(source: str) -> PolyhedralDomain:
    """Run polyhedral analysis and return the final environment."""
    interp = InvariantCapturingInterpreter(max_iterations=50)
    analysis = interp.analyze(source)
    return analysis["env"]


def get_loop_invariant_envs(source: str) -> List[PolyhedralDomain]:
    """Get the loop invariant (widened fixpoint) environments for each loop."""
    interp = InvariantCapturingInterpreter(max_iterations=50)
    interp.analyze(source)
    return interp.loop_invariants


# ---------------------------------------------------------------------------
# Comparison API
# ---------------------------------------------------------------------------

def compare_strategies(source: str, property_source: str, max_k: int = 20) -> dict:
    """Compare plain vs auto vs polyhedral k-induction."""
    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)

    results = {}

    # Plain k-induction
    t0 = time.time()
    r1 = incremental_k_induction(ts, max_k)
    results["plain_k_induction"] = {
        "result": r1.result, "k": r1.k, "time": time.time() - t0,
    }

    # Auto k-induction (V016)
    t0 = time.time()
    r2 = auto_k_induction(ts, max_k, source=source)
    results["auto_k_induction"] = {
        "result": r2.result, "k": r2.k,
        "invariants": len(r2.invariants) if hasattr(r2, 'invariants') and r2.invariants else 0,
        "time": time.time() - t0,
    }

    # Polyhedral k-induction (V129)
    t0 = time.time()
    r3 = polyhedral_k_induction(ts, source, max_k)
    results["polyhedral_k_induction"] = {
        "result": r3.result, "k": r3.k,
        "candidates_raw": r3.stats.get("candidates_raw", 0),
        "candidates_valid": r3.stats.get("candidates_valid", 0),
        "used": len(r3.used_candidates),
        "time": time.time() - t0,
    }

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def polyhedral_k_summary(source: str, property_source: str, max_k: int = 20) -> str:
    """Generate human-readable summary of polyhedral k-induction."""
    result = verify_loop_polyhedral(source, property_source, max_k)
    lines = [
        "Polyhedral k-Induction Summary",
        "=" * 40,
        f"Result: {result.result}",
    ]
    if result.k is not None:
        lines.append(f"k: {result.k}")

    raw = result.stats.get("candidates_raw", 0)
    valid = result.stats.get("candidates_valid", 0)
    lines.append(f"Candidates: {raw} raw, {valid} valid")
    lines.append(f"  interval: {result.stats.get('candidates_interval', 0)}")
    lines.append(f"  relational: {result.stats.get('candidates_relational', 0)}")

    if result.used_candidates:
        lines.append(f"Used candidates ({len(result.used_candidates)}):")
        for c in result.used_candidates:
            lines.append(f"  - {c.description} [{c.source_kind}]")

    if result.polyhedral_env and not result.polyhedral_env.is_bot():
        constraints = result.polyhedral_env.get_constraints()
        if constraints:
            lines.append(f"Polyhedral constraints ({len(constraints)}):")
            for c in constraints[:10]:
                lines.append(f"  {c}")

    for key in ("phase1_plain", "phase2_polyhedral", "phase2_subset", "phase3_auto", "phase4_combined"):
        if key in result.stats:
            lines.append(f"  {key}: {result.stats[key]}")

    t = result.stats.get("time", 0)
    lines.append(f"Time: {t:.3f}s")

    return "\n".join(lines)
