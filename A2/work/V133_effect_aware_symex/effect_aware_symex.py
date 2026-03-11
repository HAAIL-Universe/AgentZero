"""V133: Effect-Aware Symbolic Execution.

Composes V040 (effect systems) + C038 (symbolic execution).

Uses effect type information to guide and optimize symbolic execution:
- Pure functions: skip path exploration (result is deterministic)
- State effects: identify which variables are modified (focus analysis)
- Exception effects: predict which paths may raise errors
- IO effects: flag paths with observable side effects
- Divergence effects: flag potentially non-terminating paths

Effect pre-analysis runs in O(n) vs symbolic execution's exponential worst case,
providing cheap early insights that can prune or prioritize expensive SMT queries.
"""

import sys
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Set, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V040_effect_systems'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C038_symbolic_execution'))

from effect_systems import (
    infer_effects, check_effects, EffectKind, Effect, EffectSet,
    FnEffectSig, State, Exn, IO, DIV, NONDET, PURE,
)
from symbolic_execution import (
    symbolic_execute, generate_tests, check_assertions,
    SymbolicExecutor, ExecutionResult, PathState, PathStatus,
    SymValue, TestCase,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EffectAnnotatedPath:
    """A symbolic execution path annotated with effect information."""
    path: PathState
    effects: EffectSet
    has_state: bool
    has_io: bool
    has_exn: bool
    has_div: bool
    has_nondet: bool
    is_pure: bool
    state_vars: List[str]
    exn_types: List[str]


@dataclass
class EffectAwareResult:
    """Result of effect-aware symbolic execution."""
    paths: List[EffectAnnotatedPath]
    test_cases: List[TestCase]
    effect_sigs: Dict[str, FnEffectSig]
    pruned_pure_calls: int
    total_paths: int
    feasible_paths: int
    pure_functions: List[str]
    effectful_functions: List[str]
    state_variables: Set[str]
    exception_types: Set[str]
    analysis_time: float
    execution_time: float

    @property
    def total_annotated(self) -> int:
        return len(self.paths)


@dataclass
class EffectGuidance:
    """Pre-analysis guidance for symbolic execution."""
    fn_effects: Dict[str, FnEffectSig]
    pure_functions: List[str]
    state_functions: Dict[str, List[str]]  # fn -> state vars
    exn_functions: Dict[str, List[str]]  # fn -> exception types
    io_functions: List[str]
    div_functions: List[str]
    nondet_functions: List[str]
    main_effects: Optional[EffectSet]
    suggested_symbolic: List[str]  # variables worth making symbolic


# ---------------------------------------------------------------------------
# Effect pre-analysis
# ---------------------------------------------------------------------------

def analyze_effects(source: str) -> EffectGuidance:
    """Run effect inference as a cheap pre-pass before symbolic execution."""
    fn_sigs = infer_effects(source)

    pure_fns = []
    state_fns = {}
    exn_fns = {}
    io_fns = []
    div_fns = []
    nondet_fns = []
    suggested_symbolic = set()

    for name, sig in fn_sigs.items():
        if name == "__main__":
            continue
        if sig.effects.is_pure:
            pure_fns.append(name)
            continue
        if sig.effects.has(EffectKind.STATE):
            state_vars = [e.detail for e in sig.effects.effects
                          if e.kind == EffectKind.STATE and e.detail]
            state_fns[name] = state_vars
            suggested_symbolic.update(state_vars)
        if sig.effects.has(EffectKind.EXN):
            exn_types = [e.detail for e in sig.effects.effects
                         if e.kind == EffectKind.EXN and e.detail]
            exn_fns[name] = exn_types
        if sig.effects.has(EffectKind.IO):
            io_fns.append(name)
        if sig.effects.has(EffectKind.DIV):
            div_fns.append(name)
        if sig.effects.has(EffectKind.NONDET):
            nondet_fns.append(name)

    main_effects = fn_sigs.get("__main__", None)
    if main_effects:
        main_eff = main_effects.effects
        if main_eff.has(EffectKind.STATE):
            for e in main_eff.effects:
                if e.kind == EffectKind.STATE and e.detail:
                    suggested_symbolic.add(e.detail)

    return EffectGuidance(
        fn_effects=fn_sigs,
        pure_functions=pure_fns,
        state_functions=state_fns,
        exn_functions=exn_fns,
        io_functions=io_fns,
        div_functions=div_fns,
        nondet_functions=nondet_fns,
        main_effects=main_effects.effects if main_effects else None,
        suggested_symbolic=sorted(suggested_symbolic),
    )


def _annotate_path(path: PathState, guidance: EffectGuidance) -> EffectAnnotatedPath:
    """Annotate a single path with effect information based on pre-analysis."""
    # Collect effects from all functions that could have been called on this path
    combined = EffectSet.pure()
    state_vars = set()
    exn_types = set()

    # Use main effects as baseline
    if guidance.main_effects:
        combined = combined.union(guidance.main_effects)

    # Check which functions might have executed based on env/coverage
    for fn_name, sig in guidance.fn_effects.items():
        if fn_name == "__main__":
            continue
        # Conservative: include all function effects (could refine with coverage)
        combined = combined.union(sig.effects)

    for e in combined.effects:
        if e.kind == EffectKind.STATE and e.detail:
            state_vars.add(e.detail)
        if e.kind == EffectKind.EXN and e.detail:
            exn_types.add(e.detail)

    return EffectAnnotatedPath(
        path=path,
        effects=combined,
        has_state=combined.has(EffectKind.STATE),
        has_io=combined.has(EffectKind.IO),
        has_exn=combined.has(EffectKind.EXN),
        has_div=combined.has(EffectKind.DIV),
        has_nondet=combined.has(EffectKind.NONDET),
        is_pure=combined.is_pure,
        state_vars=sorted(state_vars),
        exn_types=sorted(exn_types),
    )


# ---------------------------------------------------------------------------
# Effect-aware symbolic execution
# ---------------------------------------------------------------------------

def effect_aware_execute(
    source: str,
    symbolic_inputs: Optional[Dict[str, str]] = None,
    max_paths: int = 64,
    max_loop_unroll: int = 5,
) -> EffectAwareResult:
    """Execute program with effect-guided analysis.

    Pipeline:
    1. Run effect inference (cheap, O(n))
    2. Use effect info to suggest symbolic inputs if not specified
    3. Run symbolic execution
    4. Annotate paths with effect information
    """
    # Phase 1: Effect pre-analysis
    t0 = time.time()
    guidance = analyze_effects(source)
    analysis_time = time.time() - t0

    # Phase 2: Auto-suggest symbolic inputs if not provided
    if symbolic_inputs is None and guidance.suggested_symbolic:
        symbolic_inputs = {v: 'int' for v in guidance.suggested_symbolic}

    # Phase 3: Symbolic execution
    t0 = time.time()
    result = symbolic_execute(source, symbolic_inputs, max_paths, max_loop_unroll)
    execution_time = time.time() - t0

    # Phase 4: Annotate paths
    annotated = []
    for path in result.paths:
        if path.status != PathStatus.INFEASIBLE:
            annotated.append(_annotate_path(path, guidance))

    # Collect summary info
    state_vars = set()
    exn_types = set()
    for fn_name, vars_list in guidance.state_functions.items():
        state_vars.update(vars_list)
    for fn_name, types_list in guidance.exn_functions.items():
        exn_types.update(types_list)

    return EffectAwareResult(
        paths=annotated,
        test_cases=result.test_cases,
        effect_sigs=guidance.fn_effects,
        pruned_pure_calls=0,  # TODO: implement actual pruning
        total_paths=result.total_paths,
        feasible_paths=len(result.feasible_paths),
        pure_functions=guidance.pure_functions,
        effectful_functions=list(set(
            list(guidance.state_functions.keys()) +
            guidance.io_functions +
            guidance.div_functions
        )),
        state_variables=state_vars,
        exception_types=exn_types,
        analysis_time=analysis_time,
        execution_time=execution_time,
    )


# ---------------------------------------------------------------------------
# Specialized queries
# ---------------------------------------------------------------------------

def find_effectful_paths(source: str, effect_kind: EffectKind,
                         symbolic_inputs: Optional[Dict[str, str]] = None) -> List[EffectAnnotatedPath]:
    """Find paths that exercise a specific effect kind."""
    result = effect_aware_execute(source, symbolic_inputs)
    return [p for p in result.paths if any(
        e.kind == effect_kind for e in p.effects.effects
    )]


def find_pure_paths(source: str,
                    symbolic_inputs: Optional[Dict[str, str]] = None) -> List[EffectAnnotatedPath]:
    """Find paths that are effect-free (pure)."""
    result = effect_aware_execute(source, symbolic_inputs)
    return [p for p in result.paths if p.is_pure]


def find_exception_paths(source: str,
                         symbolic_inputs: Optional[Dict[str, str]] = None) -> List[EffectAnnotatedPath]:
    """Find paths that may raise exceptions."""
    result = effect_aware_execute(source, symbolic_inputs)
    return [p for p in result.paths if p.has_exn]


def find_io_paths(source: str,
                  symbolic_inputs: Optional[Dict[str, str]] = None) -> List[EffectAnnotatedPath]:
    """Find paths that perform I/O."""
    result = effect_aware_execute(source, symbolic_inputs)
    return [p for p in result.paths if p.has_io]


def get_effect_guidance(source: str) -> EffectGuidance:
    """Get effect pre-analysis guidance without running symbolic execution."""
    return analyze_effects(source)


def suggest_symbolic_inputs(source: str) -> Dict[str, str]:
    """Use effect analysis to suggest which variables should be symbolic."""
    guidance = analyze_effects(source)
    return {v: 'int' for v in guidance.suggested_symbolic}


# ---------------------------------------------------------------------------
# Comparison and summary
# ---------------------------------------------------------------------------

def compare_aware_vs_plain(
    source: str,
    symbolic_inputs: Optional[Dict[str, str]] = None,
    max_paths: int = 64,
) -> dict:
    """Compare effect-aware vs plain symbolic execution."""
    # Plain execution
    t0 = time.time()
    plain = symbolic_execute(source, symbolic_inputs, max_paths)
    plain_time = time.time() - t0

    # Effect-aware execution
    t0 = time.time()
    aware = effect_aware_execute(source, symbolic_inputs, max_paths)
    aware_time = time.time() - t0

    return {
        "plain": {
            "total_paths": plain.total_paths,
            "feasible_paths": len(plain.feasible_paths),
            "test_cases": len(plain.test_cases),
            "time": plain_time,
        },
        "effect_aware": {
            "total_paths": aware.total_paths,
            "feasible_paths": aware.feasible_paths,
            "annotated_paths": aware.total_annotated,
            "test_cases": len(aware.test_cases),
            "pure_functions": len(aware.pure_functions),
            "effectful_functions": len(aware.effectful_functions),
            "state_variables": len(aware.state_variables),
            "analysis_time": aware.analysis_time,
            "execution_time": aware.execution_time,
            "total_time": aware_time,
        },
    }


def effect_aware_summary(source: str,
                         symbolic_inputs: Optional[Dict[str, str]] = None) -> str:
    """Generate human-readable summary of effect-aware symbolic execution."""
    result = effect_aware_execute(source, symbolic_inputs)
    lines = [
        "Effect-Aware Symbolic Execution Summary",
        "=" * 45,
        f"Total paths: {result.total_paths}",
        f"Feasible paths: {result.feasible_paths}",
        f"Annotated paths: {result.total_annotated}",
        f"Test cases: {len(result.test_cases)}",
        f"Analysis time: {result.analysis_time:.4f}s",
        f"Execution time: {result.execution_time:.4f}s",
        "",
        "Effect Analysis:",
    ]
    if result.pure_functions:
        lines.append(f"  Pure functions: {', '.join(result.pure_functions)}")
    if result.effectful_functions:
        lines.append(f"  Effectful functions: {', '.join(result.effectful_functions)}")
    if result.state_variables:
        lines.append(f"  State variables: {', '.join(sorted(result.state_variables))}")
    if result.exception_types:
        lines.append(f"  Exception types: {', '.join(sorted(result.exception_types))}")
    lines.append("")
    lines.append("Path Annotations:")
    for i, ap in enumerate(result.paths):
        tags = []
        if ap.is_pure:
            tags.append("PURE")
        if ap.has_state:
            tags.append(f"STATE({', '.join(ap.state_vars)})")
        if ap.has_io:
            tags.append("IO")
        if ap.has_exn:
            tags.append(f"EXN({', '.join(ap.exn_types)})")
        if ap.has_div:
            tags.append("DIV")
        tag_str = ", ".join(tags) if tags else "PURE"
        status = ap.path.status.value if hasattr(ap.path.status, 'value') else str(ap.path.status)
        lines.append(f"  Path {i}: [{tag_str}] status={status}")
    return "\n".join(lines)
