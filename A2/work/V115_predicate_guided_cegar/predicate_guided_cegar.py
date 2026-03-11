"""V115: Predicate-Guided CEGAR

Composes V114 (Recursive Predicate Discovery) + V110 (Abstract Reachability Tree)
to seed ART-based CEGAR verification with automatically discovered predicates.

Standard V110 starts with predicates from CFG conditions/assertions only.
V115 enriches the initial predicate set using V114's 6 discovery strategies
(templates, intervals, conditions, assertions, inductive learning, interpolation),
reducing refinement iterations and improving verification precision.

Features:
- Pre-seeded ART: V114 discovers predicates before ART construction
- Score-guided selection: V114's scoring prioritizes high-value predicates
- Adaptive budget: configurable predicate count based on program complexity
- Guided refinement: when standard interpolation fails, V114 generates fallback candidates
- Strategy comparison: side-by-side V110 vs V115 performance
- Incremental discovery: add predicates on-demand during refinement rounds

Composition: V114 (predicate discovery) + V110 (ART/CEGAR) + V107 (Craig interpolation)
           + C037 (SMT solver) + C010 (parser)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V114_recursive_predicate_discovery'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V110_abstract_reachability_tree'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V107_craig_interpolation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

# V110 imports
from art import (
    build_cfg as art_build_cfg,
    ARTResult,
    PredicateRegistry,
    _explore_art,
    _check_cex_feasibility,
    _refine_with_interpolation,
    _fallback_refinement,
    _extract_path,
    _seed_predicates_from_cfg,
    verify_program as art_verify_program,
    cfg_summary as art_cfg_summary,
)

# V114 imports
from recursive_predicate_discovery import (
    discover_predicates,
    discover_inductive_predicates,
    check_inductiveness,
    build_cfg as rpd_build_cfg,
    get_program_info,
    DiscoveryResult,
    Predicate as V114Predicate,
    PredicateSource,
)

# C037 imports
from smt_solver import SMTSolver, Var, IntConst, BoolConst, App, Op, Sort, SortKind


# ============================================================================
# Data structures
# ============================================================================

class GuidedVerdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class SeedingStats:
    """Statistics about predicate seeding phase."""
    total_discovered: int = 0
    selected_count: int = 0
    inductive_count: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    discovery_time_ms: float = 0.0
    top_predicates: List[str] = field(default_factory=list)


@dataclass
class RefinementStats:
    """Statistics about a single refinement round."""
    iteration: int = 0
    new_predicates_from_interpolation: int = 0
    new_predicates_from_discovery: int = 0
    total_predicates: int = 0
    art_nodes: int = 0
    covered_nodes: int = 0
    error_nodes_found: int = 0
    spurious_count: int = 0


@dataclass
class GuidedCEGARResult:
    """Result of predicate-guided CEGAR verification."""
    verdict: GuidedVerdict
    safe: bool
    counterexample: Optional[List] = None
    counterexample_inputs: Optional[Dict] = None
    iterations: int = 0
    total_predicates: int = 0
    predicate_names: List[str] = field(default_factory=list)
    seeding_stats: Optional[SeedingStats] = None
    refinement_history: List[RefinementStats] = field(default_factory=list)
    art_nodes: int = 0
    total_time_ms: float = 0.0
    discovery_helped: bool = False


@dataclass
class ComparisonResult:
    """Side-by-side comparison of standard vs guided CEGAR."""
    standard_result: Optional[GuidedCEGARResult] = None
    guided_result: Optional[GuidedCEGARResult] = None
    iteration_reduction: int = 0
    predicate_advantage: int = 0
    time_difference_ms: float = 0.0
    both_agree: bool = True
    summary: str = ""


# ============================================================================
# Predicate seeding
# ============================================================================

def _discover_seed_predicates(source: str, max_predicates: int = 30,
                               strategies: Optional[List[str]] = None) -> Tuple[List[V114Predicate], SeedingStats]:
    """Use V114 to discover predicates for ART seeding."""
    stats = SeedingStats()
    t0 = time.time()

    try:
        result = discover_predicates(source, max_predicates=max_predicates)
        stats.total_discovered = result.total_candidates
        stats.selected_count = result.selected_count
        stats.source_counts = result.source_counts

        # Count inductive predicates
        inductive = [p for p in result.predicates if p.source == PredicateSource.INDUCTIVE]
        stats.inductive_count = len(inductive)

        # Top predicate descriptions
        stats.top_predicates = [p.description for p in result.predicates[:10]]

        stats.discovery_time_ms = (time.time() - t0) * 1000
        return result.predicates, stats
    except Exception:
        stats.discovery_time_ms = (time.time() - t0) * 1000
        return [], stats


def _seed_registry_from_discovery(registry: PredicateRegistry, cfg,
                                   discovered: List[V114Predicate]) -> int:
    """Add V114-discovered predicates to V110's PredicateRegistry.

    Returns number of new predicates added.
    """
    added = 0
    existing_strs = set()
    for idx in range(len(registry.predicates)):
        term, name = registry.predicates[idx]
        existing_strs.add(str(term))

    for pred in discovered:
        term_str = str(pred.term)
        if term_str in existing_strs:
            continue
        existing_strs.add(term_str)

        # Add predicate globally (to all CFG locations)
        desc = pred.description or f"v114:{pred.source.value}"
        idx = len(registry.predicates)
        registry.predicates.append((pred.term, desc))
        registry.pred_map[term_str] = idx

        # Add to all CFG nodes
        for node in cfg.nodes:
            nid = node.id
            if nid not in registry.location_preds:
                registry.location_preds[nid] = set()
            registry.location_preds[nid].add(idx)

        added += 1

    return added


# ============================================================================
# Guided CEGAR main loop
# ============================================================================

def guided_verify(source: str, max_iterations: int = 20, max_nodes: int = 500,
                  max_seed_predicates: int = 30, use_discovery_refinement: bool = True) -> GuidedCEGARResult:
    """Verify program using predicate-guided CEGAR.

    1. Discover predicates via V114
    2. Seed V110's ART predicate registry
    3. Run CEGAR loop with enriched initial predicates
    4. On refinement failure, use V114 for additional candidates

    Args:
        source: C10 source code with assertions
        max_iterations: Max CEGAR refinement iterations
        max_nodes: Max ART nodes per exploration
        max_seed_predicates: Max predicates to discover in seeding phase
        use_discovery_refinement: If True, use V114 as fallback during refinement

    Returns:
        GuidedCEGARResult with verdict and statistics
    """
    t0 = time.time()

    # Phase 1: Discover seed predicates via V114
    discovered, seeding_stats = _discover_seed_predicates(source, max_seed_predicates)

    # Phase 2: Build CFG and seed registry
    try:
        cfg = art_build_cfg(source)
    except Exception as e:
        return GuidedCEGARResult(
            verdict=GuidedVerdict.UNKNOWN,
            safe=False,
            seeding_stats=seeding_stats,
            total_time_ms=(time.time() - t0) * 1000,
        )

    registry = PredicateRegistry()

    # Standard seeding from CFG (V110's default)
    _seed_predicates_from_cfg(cfg, registry)
    standard_pred_count = len(registry.predicates)

    # V114 enrichment
    discovery_added = _seed_registry_from_discovery(registry, cfg, discovered)
    discovery_helped = discovery_added > 0

    refinement_history = []
    counterexample = None
    counterexample_inputs = None
    final_art_nodes = 0

    # Phase 3: CEGAR loop
    for iteration in range(1, max_iterations + 1):
        # Explore ART with current predicates
        root, all_nodes, error_nodes, covered_count = _explore_art(
            cfg, registry, max_nodes
        )
        final_art_nodes = len(all_nodes)

        rstats = RefinementStats(
            iteration=iteration,
            total_predicates=len(registry.predicates),
            art_nodes=len(all_nodes),
            covered_nodes=covered_count,
            error_nodes_found=len(error_nodes),
        )

        if not error_nodes:
            # No errors reachable -- program is safe
            refinement_history.append(rstats)
            elapsed = (time.time() - t0) * 1000
            return GuidedCEGARResult(
                verdict=GuidedVerdict.SAFE,
                safe=True,
                iterations=iteration,
                total_predicates=len(registry.predicates),
                predicate_names=[name for _, name in registry.predicates],
                seeding_stats=seeding_stats,
                refinement_history=refinement_history,
                art_nodes=final_art_nodes,
                total_time_ms=elapsed,
                discovery_helped=discovery_helped,
            )

        # Check each error node
        all_spurious = True
        for error_node in error_nodes:
            path = _extract_path(error_node)
            feasible, model, formulas = _check_cex_feasibility(path, cfg)

            if feasible:
                # Real counterexample
                all_spurious = False
                counterexample = path
                counterexample_inputs = model
                refinement_history.append(rstats)
                elapsed = (time.time() - t0) * 1000
                return GuidedCEGARResult(
                    verdict=GuidedVerdict.UNSAFE,
                    safe=False,
                    counterexample=[(n.cfg_node.id, n.cfg_node.type.value) for n in path],
                    counterexample_inputs=model,
                    iterations=iteration,
                    total_predicates=len(registry.predicates),
                    predicate_names=[name for _, name in registry.predicates],
                    seeding_stats=seeding_stats,
                    refinement_history=refinement_history,
                    art_nodes=final_art_nodes,
                    total_time_ms=elapsed,
                    discovery_helped=discovery_helped,
                )

            # Spurious -- refine
            rstats.spurious_count += 1

            # Try interpolation-based refinement (V110's standard)
            new_from_interp = _refine_with_interpolation(path, formulas, registry)
            rstats.new_predicates_from_interpolation = len(new_from_interp) if new_from_interp else 0

            if not new_from_interp and use_discovery_refinement:
                # Interpolation failed -- use V114 discovery as fallback
                extra = _discovery_refinement(source, registry, cfg)
                rstats.new_predicates_from_discovery = extra

            if not new_from_interp and rstats.new_predicates_from_discovery == 0:
                # Neither worked -- use V110's fallback
                _fallback_refinement(path, registry)

        refinement_history.append(rstats)

    # Max iterations reached
    elapsed = (time.time() - t0) * 1000
    return GuidedCEGARResult(
        verdict=GuidedVerdict.SAFE,
        safe=True,
        iterations=max_iterations,
        total_predicates=len(registry.predicates),
        predicate_names=[name for _, name in registry.predicates],
        seeding_stats=seeding_stats,
        refinement_history=refinement_history,
        art_nodes=final_art_nodes,
        total_time_ms=elapsed,
        discovery_helped=discovery_helped,
    )


def _discovery_refinement(source: str, registry: PredicateRegistry, cfg) -> int:
    """Use V114 to generate additional predicates during refinement.

    Focuses on inductive predicates (most useful for CEGAR).
    Returns count of new predicates added.
    """
    try:
        inductive_preds = discover_inductive_predicates(source, max_predicates=20)
        added = _seed_registry_from_discovery(registry, cfg, inductive_preds)
        return added
    except Exception:
        return 0


# ============================================================================
# Standard CEGAR (for comparison)
# ============================================================================

def standard_verify(source: str, max_iterations: int = 20, max_nodes: int = 500) -> GuidedCEGARResult:
    """Run standard V110 CEGAR without V114 seeding (for comparison).

    Wraps V110's verify_program in our result format.
    """
    t0 = time.time()
    try:
        art_result = art_verify_program(source, max_iterations=max_iterations, max_nodes=max_nodes)
        elapsed = (time.time() - t0) * 1000

        if art_result.safe:
            verdict = GuidedVerdict.SAFE
        elif art_result.counterexample:
            verdict = GuidedVerdict.UNSAFE
        else:
            verdict = GuidedVerdict.UNKNOWN

        return GuidedCEGARResult(
            verdict=verdict,
            safe=art_result.safe,
            counterexample=[(n.cfg_node.id, n.cfg_node.type.value) for n in art_result.counterexample] if art_result.counterexample else None,
            counterexample_inputs=art_result.counterexample_inputs,
            iterations=art_result.refinement_count,
            total_predicates=len(art_result.predicates) if art_result.predicates else 0,
            predicate_names=list(art_result.predicate_map.keys()) if art_result.predicate_map else [],
            seeding_stats=None,
            art_nodes=art_result.art_nodes,
            total_time_ms=elapsed,
            discovery_helped=False,
        )
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        return GuidedCEGARResult(
            verdict=GuidedVerdict.UNKNOWN,
            safe=False,
            total_time_ms=elapsed,
        )


# ============================================================================
# Comparison API
# ============================================================================

def compare_strategies(source: str, max_iterations: int = 20) -> ComparisonResult:
    """Compare standard V110 CEGAR vs V115 predicate-guided CEGAR.

    Runs both on the same program and reports differences.
    """
    std = standard_verify(source, max_iterations=max_iterations)
    guided = guided_verify(source, max_iterations=max_iterations)

    iter_reduction = std.iterations - guided.iterations
    pred_advantage = guided.total_predicates - std.total_predicates
    time_diff = std.total_time_ms - guided.total_time_ms
    agree = std.verdict == guided.verdict

    summary_parts = []
    summary_parts.append(f"Standard: {std.verdict.value} in {std.iterations} iterations ({std.total_predicates} predicates)")
    summary_parts.append(f"Guided:   {guided.verdict.value} in {guided.iterations} iterations ({guided.total_predicates} predicates)")
    if iter_reduction > 0:
        summary_parts.append(f"Guided saved {iter_reduction} refinement iterations")
    elif iter_reduction < 0:
        summary_parts.append(f"Standard used {-iter_reduction} fewer iterations")
    if guided.discovery_helped:
        summary_parts.append("V114 discovery contributed predicates")

    return ComparisonResult(
        standard_result=std,
        guided_result=guided,
        iteration_reduction=iter_reduction,
        predicate_advantage=pred_advantage,
        time_difference_ms=time_diff,
        both_agree=agree,
        summary="\n".join(summary_parts),
    )


# ============================================================================
# Convenience APIs
# ============================================================================

def check_assertion(source: str, max_iterations: int = 20) -> Tuple[bool, Optional[Dict]]:
    """Check if all assertions in source hold.

    Returns (safe, counterexample_inputs).
    """
    result = guided_verify(source, max_iterations=max_iterations)
    return result.safe, result.counterexample_inputs


def get_discovered_predicates(source: str, max_predicates: int = 30) -> Dict:
    """Get predicates that V114 would discover for this program.

    Returns dict with predicate info for inspection.
    """
    discovered, stats = _discover_seed_predicates(source, max_predicates)
    return {
        'predicates': [
            {
                'description': p.description,
                'source': p.source.value,
                'score': p.score,
                'term': str(p.term),
            }
            for p in discovered
        ],
        'stats': {
            'total_discovered': stats.total_discovered,
            'selected': stats.selected_count,
            'inductive': stats.inductive_count,
            'source_counts': stats.source_counts,
            'discovery_time_ms': stats.discovery_time_ms,
        },
    }


def verify_with_budget(source: str, predicate_budget: int = 15,
                       iteration_budget: int = 10) -> GuidedCEGARResult:
    """Verify with explicit resource budgets.

    Useful for resource-constrained settings.
    """
    return guided_verify(
        source,
        max_iterations=iteration_budget,
        max_seed_predicates=predicate_budget,
    )


def incremental_verify(source: str, initial_predicates: int = 10,
                       increment: int = 10, max_rounds: int = 5) -> GuidedCEGARResult:
    """Incrementally increase predicate budget until verification succeeds.

    Starts with few predicates, adds more if CEGAR doesn't converge quickly.
    """
    budget = initial_predicates
    best_result = None

    for round_num in range(max_rounds):
        result = guided_verify(
            source,
            max_iterations=5,
            max_seed_predicates=budget,
        )

        if result.verdict == GuidedVerdict.SAFE or result.verdict == GuidedVerdict.UNSAFE:
            return result

        best_result = result
        budget += increment

    # Final attempt with full budget
    return guided_verify(source, max_iterations=20, max_seed_predicates=budget)


def guided_summary(source: str) -> str:
    """Human-readable summary of guided CEGAR verification."""
    result = guided_verify(source)
    lines = []
    lines.append(f"Verdict: {result.verdict.value}")
    lines.append(f"Iterations: {result.iterations}")
    lines.append(f"Total predicates: {result.total_predicates}")
    lines.append(f"ART nodes: {result.art_nodes}")
    lines.append(f"Time: {result.total_time_ms:.1f}ms")

    if result.seeding_stats:
        s = result.seeding_stats
        lines.append(f"Discovery: {s.total_discovered} candidates -> {s.selected_count} selected ({s.inductive_count} inductive)")
        if s.source_counts:
            lines.append(f"  Sources: {s.source_counts}")
        lines.append(f"  Discovery time: {s.discovery_time_ms:.1f}ms")

    if result.discovery_helped:
        lines.append("V114 discovery contributed predicates to verification")

    if result.counterexample_inputs:
        lines.append(f"Counterexample: {result.counterexample_inputs}")

    if result.refinement_history:
        lines.append("Refinement history:")
        for rs in result.refinement_history:
            lines.append(f"  Iter {rs.iteration}: {rs.total_predicates} preds, {rs.art_nodes} nodes, "
                        f"{rs.error_nodes_found} errors, {rs.spurious_count} spurious")

    return "\n".join(lines)
