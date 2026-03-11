"""V127: Landmark-Guided k-Induction

Composes V124 (landmark widening) + V016 (auto-strengthened k-induction) to use
per-loop landmark analysis for generating better invariant candidates.

Pipeline:
  1. V124 analyzes loop structure (landmarks, recurrences, per-variable thresholds)
  2. Extract invariant candidates from loop profiles (bounds, recurrence limits, thresholds)
  3. Feed candidates as strengthening invariants to V015 k-induction
  4. Fall back to V016 auto-inference if landmark candidates insufficient
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V124_landmark_widening'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V016_auto_strengthened_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V005_ai_strengthened_pdr'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from fractions import Fraction

from landmark_widening import (
    landmark_analyze, LandmarkResult, LoopProfile, Landmark, LandmarkKind,
    LandmarkConfig, LandmarkInterpreter
)
from auto_k_induction import (
    auto_k_induction, AutoKIndResult, verify_loop_auto,
    _infer_strengthening_invariants, _invariants_to_smt, _validate_ts_invariant,
    _parse_property, _parse_property_sexpr,
    compare_strategies as v016_compare_strategies
)
from k_induction import (
    k_induction_check, incremental_k_induction,
    k_induction_with_strengthening, KIndResult,
    _extract_loop_ts
)
from pdr import TransitionSystem
from smt_solver import SMTSolver, SMTResult, Var, IntConst, BoolConst, App, Op, INT, BOOL


# ---- Data structures ----

@dataclass
class LandmarkCandidate:
    """An invariant candidate derived from landmark analysis."""
    formula: App           # SMT formula
    description: str       # Human-readable
    source_kind: str       # "init_bound", "condition_bound", "recurrence_limit", "threshold", "narrowing"
    variable: str          # Variable it constrains
    priority: int = 0      # Higher = more likely useful


@dataclass
class LandmarkKIndResult:
    """Result of landmark-guided k-induction."""
    result: str                                    # "SAFE", "UNSAFE", "UNKNOWN"
    k: Optional[int] = None                        # k value used
    counterexample: Optional[list] = None          # States if UNSAFE
    landmark_candidates: List[LandmarkCandidate] = field(default_factory=list)
    used_candidates: List[LandmarkCandidate] = field(default_factory=list)
    strengthening: Optional[App] = None            # Combined SMT if used
    stats: dict = field(default_factory=dict)
    loop_profile: Optional[LoopProfile] = None     # From V124

    def __repr__(self):
        return f"LandmarkKIndResult({self.result}, k={self.k}, candidates={len(self.landmark_candidates)})"


# ---- Landmark -> invariant candidate extraction ----

def _extract_candidates_from_profile(profile: LoopProfile, ts: TransitionSystem) -> List[LandmarkCandidate]:
    """Extract invariant candidates from a V124 LoopProfile."""
    candidates = []
    var_names = {name for name, _ in ts.state_vars}

    for landmark in profile.landmarks:
        vname = landmark.variable
        if vname not in var_names:
            continue

        var = ts.var(vname)
        val = int(landmark.value)

        if landmark.kind == LandmarkKind.INIT_VALUE:
            # Init value as lower or upper bound
            candidates.append(LandmarkCandidate(
                formula=App(Op.GE, [var, IntConst(val)], BOOL),
                description=f"{vname} >= {val} (init value)",
                source_kind="init_bound",
                variable=vname,
                priority=landmark.priority,
            ))
            candidates.append(LandmarkCandidate(
                formula=App(Op.LE, [var, IntConst(val)], BOOL),
                description=f"{vname} <= {val} (init upper)",
                source_kind="init_bound",
                variable=vname,
                priority=landmark.priority - 1,
            ))

        elif landmark.kind == LandmarkKind.CONDITION_BOUND:
            # Loop condition bound -> variable stays within
            candidates.append(LandmarkCandidate(
                formula=App(Op.LE, [var, IntConst(val)], BOOL),
                description=f"{vname} <= {val} (condition bound)",
                source_kind="condition_bound",
                variable=vname,
                priority=landmark.priority,
            ))
            candidates.append(LandmarkCandidate(
                formula=App(Op.GE, [var, IntConst(0)], BOOL),
                description=f"{vname} >= 0 (non-negativity from condition)",
                source_kind="condition_bound",
                variable=vname,
                priority=landmark.priority - 2,
            ))

        elif landmark.kind == LandmarkKind.BRANCH_THRESHOLD:
            candidates.append(LandmarkCandidate(
                formula=App(Op.LE, [var, IntConst(val)], BOOL),
                description=f"{vname} <= {val} (branch threshold)",
                source_kind="threshold",
                variable=vname,
                priority=landmark.priority,
            ))

    # Recurrence-derived bounds
    for rec in profile.recurrences:
        vname = rec.var
        if vname not in var_names:
            continue
        var = ts.var(vname)

        if rec.condition_bound is not None:
            bound_val = int(rec.condition_bound)
            candidates.append(LandmarkCandidate(
                formula=App(Op.LE, [var, IntConst(bound_val)], BOOL),
                description=f"{vname} <= {bound_val} (recurrence limit)",
                source_kind="recurrence_limit",
                variable=vname,
                priority=8,
            ))

        if rec.init_lower is not None and rec.init_lower != float('-inf'):
            init_val = int(rec.init_lower)
            candidates.append(LandmarkCandidate(
                formula=App(Op.GE, [var, IntConst(init_val)], BOOL),
                description=f"{vname} >= {init_val} (recurrence init)",
                source_kind="recurrence_limit",
                variable=vname,
                priority=7,
            ))

    # Per-variable thresholds as bounds
    for vname, thresholds in profile.per_var_thresholds.items():
        if vname not in var_names:
            continue
        var = ts.var(vname)

        if thresholds:
            max_thresh = int(max(thresholds))
            min_thresh = int(min(thresholds))
            candidates.append(LandmarkCandidate(
                formula=App(Op.LE, [var, IntConst(max_thresh)], BOOL),
                description=f"{vname} <= {max_thresh} (max threshold)",
                source_kind="threshold",
                variable=vname,
                priority=5,
            ))
            if min_thresh >= 0:
                candidates.append(LandmarkCandidate(
                    formula=App(Op.GE, [var, IntConst(min_thresh)], BOOL),
                    description=f"{vname} >= {min_thresh} (min threshold)",
                    source_kind="threshold",
                    variable=vname,
                    priority=5,
                ))

    return candidates


def _extract_candidates_from_analysis(lm_result: LandmarkResult, ts: TransitionSystem) -> List[LandmarkCandidate]:
    """Extract candidates from V124 analysis results (post-fixpoint ranges)."""
    candidates = []
    var_names = {name for name, _ in ts.state_vars}

    # Extract bounds from final polyhedral environment
    env = lm_result.env
    for vname in var_names:
        if vname not in env.var_names:
            continue
        lo, hi = env.get_interval(vname)
        var = ts.var(vname)

        if lo is not None and lo > float('-inf'):
            lo_int = int(lo)
            candidates.append(LandmarkCandidate(
                formula=App(Op.GE, [var, IntConst(lo_int)], BOOL),
                description=f"{vname} >= {lo_int} (analysis lower bound)",
                source_kind="narrowing",
                variable=vname,
                priority=6,
            ))
        if hi is not None and hi < float('inf'):
            hi_int = int(hi)
            candidates.append(LandmarkCandidate(
                formula=App(Op.LE, [var, IntConst(hi_int)], BOOL),
                description=f"{vname} <= {hi_int} (analysis upper bound)",
                source_kind="narrowing",
                variable=vname,
                priority=6,
            ))

    return candidates


# ---- Candidate validation and selection ----

def _validate_candidate(ts: TransitionSystem, candidate: LandmarkCandidate) -> bool:
    """Validate that a candidate is an inductive invariant for the TS."""
    return _validate_ts_invariant(ts, candidate.formula)


def _select_best_candidates(candidates: List[LandmarkCandidate], ts: TransitionSystem,
                            max_candidates: int = 10) -> List[LandmarkCandidate]:
    """Select and validate the best landmark candidates."""
    # Sort by priority (highest first), deduplicate by description
    seen = set()
    unique = []
    for c in sorted(candidates, key=lambda x: -x.priority):
        if c.description not in seen:
            seen.add(c.description)
            unique.append(c)

    # Validate each candidate
    valid = []
    for c in unique[:max_candidates * 2]:  # Check more than needed
        if _validate_candidate(ts, c):
            valid.append(c)
            if len(valid) >= max_candidates:
                break

    return valid


# ---- Core: landmark-guided k-induction ----

def landmark_k_induction(ts: TransitionSystem, source: str,
                         max_k: int = 20, config: Optional[LandmarkConfig] = None) -> LandmarkKIndResult:
    """Main entry: k-induction with landmark-derived strengthening.

    Pipeline:
      1. Plain k-induction (quick check)
      2. V124 landmark analysis -> candidate extraction -> validation
      3. k-induction with landmark strengthening
      4. Fallback: V016 auto-inference
      5. Fallback: combined (landmarks + auto)
    """
    t0 = time.time()
    stats = {}

    # Phase 1: Plain k-induction
    plain = incremental_k_induction(ts, max_k)
    stats["phase1_plain"] = plain.result
    if plain.result in ("SAFE", "UNSAFE"):
        return LandmarkKIndResult(
            result=plain.result,
            k=plain.k,
            counterexample=plain.counterexample,
            stats={**stats, "time": time.time() - t0},
        )

    # Phase 2: Landmark analysis
    lm_result = landmark_analyze(source, config)
    stats["landmark_loops"] = len(lm_result.loop_profiles)

    # Extract candidates from all loop profiles
    all_candidates = []
    profile = None
    for loop_id, lp in lm_result.loop_profiles.items():
        profile = lp
        all_candidates.extend(_extract_candidates_from_profile(lp, ts))

    # Also extract from analysis results
    all_candidates.extend(_extract_candidates_from_analysis(lm_result, ts))
    stats["candidates_raw"] = len(all_candidates)

    # Validate and select
    valid_candidates = _select_best_candidates(all_candidates, ts)
    stats["candidates_valid"] = len(valid_candidates)

    if valid_candidates:
        # Try strengthening with landmark candidates
        smt_invs = [c.formula for c in valid_candidates]
        result = k_induction_with_strengthening(ts, max_k, invariants=smt_invs)
        stats["phase2_landmark"] = result.result

        if result.result == "SAFE":
            return LandmarkKIndResult(
                result="SAFE",
                k=result.k,
                landmark_candidates=all_candidates,
                used_candidates=valid_candidates,
                strengthening=_combine_smt(smt_invs),
                stats={**stats, "time": time.time() - t0},
                loop_profile=profile,
            )

        # Try subsets of landmark candidates
        if len(valid_candidates) > 1:
            for i in range(len(valid_candidates)):
                subset = [c for j, c in enumerate(valid_candidates) if j != i]
                smt_sub = [c.formula for c in subset]
                sub_result = k_induction_with_strengthening(ts, max_k, invariants=smt_sub)
                if sub_result.result == "SAFE":
                    stats["phase2_subset"] = f"leave-out-{i}"
                    return LandmarkKIndResult(
                        result="SAFE",
                        k=sub_result.k,
                        landmark_candidates=all_candidates,
                        used_candidates=subset,
                        strengthening=_combine_smt(smt_sub),
                        stats={**stats, "time": time.time() - t0},
                        loop_profile=profile,
                    )

    # Phase 3: V016 auto-inference fallback
    auto = auto_k_induction(ts, max_k, source=source)
    stats["phase3_auto"] = auto.result

    if auto.result in ("SAFE", "UNSAFE"):
        return LandmarkKIndResult(
            result=auto.result,
            k=auto.k,
            counterexample=auto.counterexample,
            landmark_candidates=all_candidates,
            used_candidates=valid_candidates,
            strengthening=auto.strengthening,
            stats={**stats, "time": time.time() - t0},
            loop_profile=profile,
        )

    # Phase 4: Combined (landmarks + auto invariants)
    if valid_candidates and auto.invariants:
        combined_smt = [c.formula for c in valid_candidates]
        auto_smt = _invariants_to_smt(auto.invariants, ts)
        combined_smt.extend(auto_smt)
        combined_result = k_induction_with_strengthening(ts, max_k, invariants=combined_smt)
        stats["phase4_combined"] = combined_result.result

        if combined_result.result == "SAFE":
            return LandmarkKIndResult(
                result="SAFE",
                k=combined_result.k,
                landmark_candidates=all_candidates,
                used_candidates=valid_candidates,
                strengthening=_combine_smt(combined_smt),
                stats={**stats, "time": time.time() - t0},
                loop_profile=profile,
            )

    return LandmarkKIndResult(
        result="UNKNOWN",
        k=max_k,
        landmark_candidates=all_candidates,
        used_candidates=valid_candidates,
        stats={**stats, "time": time.time() - t0},
        loop_profile=profile,
    )


def _combine_smt(terms: List[App]) -> Optional[App]:
    """Combine SMT terms with AND."""
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    result = terms[0]
    for t in terms[1:]:
        result = App(Op.AND, [result, t], BOOL)
    return result


# ---- Source-level APIs ----

def verify_loop_landmark(source: str, property_source: str,
                         max_k: int = 20) -> LandmarkKIndResult:
    """Verify a loop property using landmark-guided k-induction.

    Args:
        source: C10 source with while loop
        property_source: Property expression (e.g., "x >= 0")
        max_k: Maximum induction depth
    """
    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)
    return landmark_k_induction(ts, source, max_k)


def verify_loop_landmark_with_config(source: str, property_source: str,
                                      config: LandmarkConfig,
                                      max_k: int = 20) -> LandmarkKIndResult:
    """Verify with custom landmark configuration."""
    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)
    return landmark_k_induction(ts, source, max_k, config)


# ---- Comparison APIs ----

def compare_strategies(source: str, property_source: str,
                       max_k: int = 20) -> dict:
    """Compare landmark-guided vs V016 auto vs plain k-induction."""
    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)

    # Plain k-induction
    t0 = time.time()
    plain = incremental_k_induction(ts, max_k)
    plain_time = time.time() - t0

    # V016 auto
    t0 = time.time()
    auto = auto_k_induction(ts, max_k, source=source)
    auto_time = time.time() - t0

    # V127 landmark-guided
    t0 = time.time()
    landmark = landmark_k_induction(ts, source, max_k)
    landmark_time = time.time() - t0

    return {
        "plain_k_induction": {
            "result": plain.result,
            "k": plain.k,
            "time": plain_time,
        },
        "auto_k_induction": {
            "result": auto.result,
            "k": auto.k,
            "invariants": len(auto.invariants) if auto.invariants else 0,
            "time": auto_time,
        },
        "landmark_k_induction": {
            "result": landmark.result,
            "k": landmark.k,
            "candidates_raw": landmark.stats.get("candidates_raw", 0),
            "candidates_valid": landmark.stats.get("candidates_valid", 0),
            "used": len(landmark.used_candidates),
            "time": landmark_time,
        },
    }


def get_landmark_candidates(source: str, property_source: str) -> List[LandmarkCandidate]:
    """Get landmark candidates without running k-induction (for inspection)."""
    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_property(property_source, ts_vars)
    ts.set_property(prop_smt)
    lm_result = landmark_analyze(source)

    all_candidates = []
    for loop_id, lp in lm_result.loop_profiles.items():
        all_candidates.extend(_extract_candidates_from_profile(lp, ts))
    all_candidates.extend(_extract_candidates_from_analysis(lm_result, ts))

    return _select_best_candidates(all_candidates, ts)


def landmark_k_summary(source: str, property_source: str, max_k: int = 20) -> str:
    """Human-readable summary of landmark-guided k-induction."""
    result = verify_loop_landmark(source, property_source, max_k)

    lines = [f"Landmark-Guided k-Induction: {result.result}"]
    if result.k is not None:
        lines.append(f"  k = {result.k}")
    lines.append(f"  Raw candidates: {result.stats.get('candidates_raw', 0)}")
    lines.append(f"  Valid candidates: {result.stats.get('candidates_valid', 0)}")
    lines.append(f"  Used candidates: {len(result.used_candidates)}")

    if result.used_candidates:
        lines.append("  Invariants used:")
        for c in result.used_candidates:
            lines.append(f"    - {c.description}")

    if result.loop_profile:
        lines.append(f"  Loop landmarks: {len(result.loop_profile.landmarks)}")
        lines.append(f"  Loop recurrences: {len(result.loop_profile.recurrences)}")

    for key in ("phase1_plain", "phase2_landmark", "phase3_auto", "phase4_combined"):
        if key in result.stats:
            lines.append(f"  {key}: {result.stats[key]}")

    lines.append(f"  Time: {result.stats.get('time', 0):.3f}s")
    return "\n".join(lines)
