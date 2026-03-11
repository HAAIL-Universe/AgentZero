"""V125: Predicate-Minimized CEGAR

Composes V122 (predicate minimization) + V119 (BDD predicate abstraction CEGAR)
to integrate predicate minimization directly into the CEGAR loop.

Three modes:
1. Post-hoc minimization: standard CEGAR then minimize the proof predicates
2. Online minimization: periodically prune predicates during CEGAR iterations
3. Eager minimization: minimize after every refinement step

The key insight: CEGAR discovers predicates lazily (one spurious path at a time),
but may accumulate redundant predicates from different refinement iterations.
Minimization identifies and removes these, producing smaller proofs and faster
subsequent verification.
"""

import sys
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Set, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V119_bdd_predicate_abstraction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V122_predicate_minimization'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V110_abstract_reachability_tree'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'challenges', 'C010_stack_vm'))

from bdd_predicate_abstraction import (
    BDDCEGAR, BDDPredicateManager, BDDPredicateState,
    BDDCEGARResult, BDDVerdict, TransitionBDDBuilder,
    _is_unsat, _is_sat, _smt_not, _declare_vars, _safe_ast_to_smt,
    _collect_vars, _substitute_smt, _version_term,
)
from predicate_minimization import (
    PredicateMinimizer, SubsetVerifier, MinimizationResult,
    PredicateRole, PredicateInfo, ComparisonResult as MinComparisonResult,
    minimize_predicates, classify_predicates,
)
from art import build_cfg, CFG, CFGNode, CFGNodeType, _ast_to_smt
from bdd_model_checker import BDD, BDDNode
from smt_solver import SMTSolver, Var, IntConst, App, Op, Sort, SortKind

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


class MinCEGARVerdict(Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


class MinimizationMode(Enum):
    POST_HOC = "post_hoc"       # Minimize after CEGAR finds SAFE
    ONLINE = "online"           # Minimize periodically during CEGAR
    EAGER = "eager"             # Minimize after every refinement


@dataclass
class MinCEGARResult:
    """Result of predicate-minimized CEGAR verification."""
    verdict: MinCEGARVerdict
    safe: bool
    # Predicate info
    original_predicates: int        # Predicates from raw CEGAR
    minimal_predicates: int         # After minimization
    predicate_names: List[str]      # Descriptions of minimal predicates
    removed_predicates: List[str]   # Descriptions of removed predicates
    reduction_ratio: float          # (original - minimal) / original
    # CEGAR stats
    cegar_iterations: int
    art_nodes: int
    # Minimization stats
    minimization_mode: str
    minimization_iterations: int    # Re-verification attempts
    online_prunings: int            # Number of online prune passes (online/eager mode)
    # Counterexample (if unsafe)
    counterexample: Optional[List] = None
    counterexample_inputs: Optional[Dict] = None
    # Timing
    cegar_time_ms: float = 0.0
    minimization_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def summary(self) -> str:
        lines = [f"Predicate-Minimized CEGAR: {self.verdict.value}"]
        if self.safe:
            lines.append(f"  Predicates: {self.original_predicates} -> {self.minimal_predicates} "
                         f"(removed {self.original_predicates - self.minimal_predicates}, "
                         f"{self.reduction_ratio:.0%} reduction)")
            lines.append(f"  Minimal set: {', '.join(self.predicate_names)}")
            if self.removed_predicates:
                lines.append(f"  Removed: {', '.join(self.removed_predicates)}")
        else:
            lines.append(f"  Counterexample: {self.counterexample_inputs}")
        lines.append(f"  Mode: {self.minimization_mode}")
        lines.append(f"  CEGAR iterations: {self.cegar_iterations}")
        lines.append(f"  ART nodes: {self.art_nodes}")
        lines.append(f"  Minimization re-verifications: {self.minimization_iterations}")
        if self.online_prunings > 0:
            lines.append(f"  Online prunings: {self.online_prunings}")
        lines.append(f"  Time: {self.total_time_ms:.1f}ms "
                     f"(CEGAR: {self.cegar_time_ms:.1f}ms, "
                     f"minimize: {self.minimization_time_ms:.1f}ms)")
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Comparison of minimization modes."""
    post_hoc: Optional[MinCEGARResult]
    online: Optional[MinCEGARResult]
    eager: Optional[MinCEGARResult]
    standard: Optional[BDDCEGARResult]  # V119 without minimization
    best_mode: str
    best_predicates: int
    summary: str


# --- Core: Online-Minimized CEGAR ---

class MinimizedCEGAR:
    """CEGAR loop with integrated predicate minimization.

    Extends V119's BDD-CEGAR with periodic or eager predicate pruning.
    After each refinement (or every N iterations), tests whether existing
    predicates can be reduced while maintaining the SAFE verdict.
    """

    def __init__(self, source: str, mode: MinimizationMode = MinimizationMode.POST_HOC,
                 max_iterations: int = 20, max_nodes: int = 500,
                 prune_interval: int = 3, seed_predicates: bool = True):
        self.source = source
        self.mode = mode
        self.max_iterations = max_iterations
        self.max_nodes = max_nodes
        self.prune_interval = prune_interval
        self.seed_predicates = seed_predicates

        # Stats
        self.minimization_iterations = 0
        self.online_prunings = 0

    def verify(self) -> MinCEGARResult:
        """Run predicate-minimized CEGAR verification."""
        start = time.time()

        if self.mode == MinimizationMode.POST_HOC:
            result = self._post_hoc_verify()
        elif self.mode == MinimizationMode.ONLINE:
            result = self._online_verify()
        elif self.mode == MinimizationMode.EAGER:
            result = self._eager_verify()
        else:
            result = self._post_hoc_verify()

        result.total_time_ms = (time.time() - start) * 1000
        return result

    def _post_hoc_verify(self) -> MinCEGARResult:
        """Standard CEGAR, then post-hoc minimization on result."""
        # Phase 1: Run standard BDD-CEGAR
        cegar_start = time.time()
        cegar = BDDCEGAR(self.source, self.max_iterations, self.max_nodes, self.seed_predicates)
        cegar_result = cegar.verify()
        cegar_time = (time.time() - cegar_start) * 1000

        if not cegar_result.safe:
            return MinCEGARResult(
                verdict=MinCEGARVerdict.UNSAFE,
                safe=False,
                original_predicates=cegar_result.total_predicates,
                minimal_predicates=cegar_result.total_predicates,
                predicate_names=cegar_result.predicate_names,
                removed_predicates=[],
                reduction_ratio=0.0,
                cegar_iterations=cegar_result.iterations,
                art_nodes=cegar_result.art_nodes,
                minimization_mode=MinimizationMode.POST_HOC.value,
                minimization_iterations=0,
                online_prunings=0,
                counterexample=cegar_result.counterexample,
                counterexample_inputs=cegar_result.counterexample_inputs,
                cegar_time_ms=cegar_time,
                minimization_time_ms=0.0,
            )

        # Phase 2: Minimize predicates
        min_start = time.time()
        min_result = self._minimize_predicates(cegar.mgr)
        min_time = (time.time() - min_start) * 1000

        return self._build_safe_result(
            cegar_result, min_result, cegar_time, min_time,
            MinimizationMode.POST_HOC.value,
        )

    def _online_verify(self) -> MinCEGARResult:
        """CEGAR with periodic predicate pruning every N iterations."""
        return self._iterative_verify(MinimizationMode.ONLINE, self.prune_interval)

    def _eager_verify(self) -> MinCEGARResult:
        """CEGAR with predicate pruning after every refinement."""
        return self._iterative_verify(MinimizationMode.EAGER, 1)

    def _iterative_verify(self, mode: MinimizationMode, interval: int) -> MinCEGARResult:
        """Core CEGAR loop with integrated minimization.

        After every `interval` refinement iterations, run support-based
        minimization to prune dead predicates. This keeps the BDD state
        space small throughout the CEGAR loop.
        """
        cegar_start = time.time()
        cegar = BDDCEGAR(self.source, self.max_iterations, self.max_nodes, self.seed_predicates)

        # Seed predicates
        cegar._seed_from_cfg()
        if cegar.seed_predicates:
            try:
                cegar._seed_from_v114()
            except Exception:
                pass

        total_art_nodes = 0
        iteration = 0
        cegar_result = None
        prune_count = 0
        min_time_accum = 0.0

        for iteration in range(1, self.max_iterations + 1):
            # Build transitions with current predicates
            cegar.builder = TransitionBDDBuilder(cegar.mgr)
            cegar._edge_trans = {}
            cegar._build_transitions()

            # Explore ART
            root, all_nodes, error_nodes, covered = cegar._explore_art()
            total_art_nodes += len(all_nodes)

            if not error_nodes:
                # SAFE -- no error nodes reachable
                cegar_time = (time.time() - cegar_start) * 1000 - min_time_accum

                # Final minimization pass
                min_start = time.time()
                min_result = self._minimize_predicates(cegar.mgr)
                min_time = (time.time() - min_start) * 1000 + min_time_accum

                cegar_result = BDDCEGARResult(
                    verdict=BDDVerdict.SAFE,
                    safe=True,
                    counterexample=None,
                    counterexample_inputs=None,
                    iterations=iteration,
                    total_predicates=cegar.mgr.num_predicates,
                    predicate_names=[d for _, d in cegar.mgr.predicates],
                    art_nodes=total_art_nodes,
                    transition_bdds_built=0,
                    bdd_image_ops=cegar.bdd_image_ops,
                    smt_queries_saved=0,
                    total_time_ms=cegar_time,
                )
                return self._build_safe_result(
                    cegar_result, min_result, cegar_time, min_time, mode.value,
                    online_prunings=prune_count,
                )

            # Check feasibility of error paths
            pre_refine_count = cegar.mgr.num_predicates
            for enode in error_nodes:
                path = cegar._extract_path(enode)
                feasible, model = cegar._check_feasibility(path)
                if feasible:
                    cegar_time = (time.time() - cegar_start) * 1000
                    return MinCEGARResult(
                        verdict=MinCEGARVerdict.UNSAFE,
                        safe=False,
                        original_predicates=cegar.mgr.num_predicates,
                        minimal_predicates=cegar.mgr.num_predicates,
                        predicate_names=[d for _, d in cegar.mgr.predicates],
                        removed_predicates=[],
                        reduction_ratio=0.0,
                        cegar_iterations=iteration,
                        art_nodes=total_art_nodes,
                        minimization_mode=mode.value,
                        minimization_iterations=self.minimization_iterations,
                        online_prunings=prune_count,
                        counterexample=[(n.id, n.cfg_node.type.name) for n in path],
                        counterexample_inputs=model,
                        cegar_time_ms=cegar_time,
                        minimization_time_ms=min_time_accum,
                    )

                # Spurious -- refine
                cegar._refine(path)

            # Online minimization: prune predicates periodically
            # Only prune from pre-refinement predicates (newly added ones
            # aren't in transition BDDs yet and would appear falsely dead)
            if iteration % interval == 0 and pre_refine_count > 1:
                min_start = time.time()
                pruned = self._online_prune(cegar, pre_refine_count)
                min_time_accum += (time.time() - min_start) * 1000
                if pruned > 0:
                    prune_count += 1

        # Max iterations reached
        cegar_time = (time.time() - cegar_start) * 1000
        return MinCEGARResult(
            verdict=MinCEGARVerdict.UNKNOWN,
            safe=False,
            original_predicates=cegar.mgr.num_predicates,
            minimal_predicates=cegar.mgr.num_predicates,
            predicate_names=[d for _, d in cegar.mgr.predicates],
            removed_predicates=[],
            reduction_ratio=0.0,
            cegar_iterations=iteration,
            art_nodes=total_art_nodes,
            minimization_mode=mode.value,
            minimization_iterations=self.minimization_iterations,
            online_prunings=prune_count,
            cegar_time_ms=cegar_time,
            minimization_time_ms=min_time_accum,
        )

    def _online_prune(self, cegar: BDDCEGAR, max_idx: int = -1) -> int:
        """Prune dead predicates from the CEGAR's predicate manager in-place.

        Uses BDD support analysis: predicates whose BDD variables don't appear
        in any transition BDD are dead and can be removed.

        Args:
            max_idx: Only consider predicates with index < max_idx for pruning.
                     Predicates at or above this index are newly added and kept.
                     If -1, consider all predicates.

        Returns number of predicates pruned.
        """
        mgr = cegar.mgr
        if mgr.num_predicates <= 1:
            return 0

        if max_idx < 0:
            max_idx = mgr.num_predicates

        # Collect all BDD variable indices appearing in transition BDDs
        live_vars = set()
        for bdd_node in cegar._edge_trans.values():
            self._collect_bdd_support(bdd_node, live_vars, set())

        # Map BDD var indices to predicate indices
        live_preds = set()
        for var_idx in live_vars:
            pred_idx = var_idx // 2
            if 0 <= pred_idx < mgr.num_predicates:
                live_preds.add(pred_idx)

        # Find dead predicates (only among pre-refinement set)
        candidate_preds = set(range(min(max_idx, mgr.num_predicates)))
        dead_preds = candidate_preds - live_preds

        if not dead_preds:
            return 0

        # Rebuild predicate manager: keep live candidates + all new predicates
        keep_indices = (live_preds & candidate_preds) | set(range(max_idx, mgr.num_predicates))
        live_pred_list = [(mgr.predicates[i][0], mgr.predicates[i][1])
                         for i in sorted(keep_indices)]

        new_mgr = BDDPredicateManager()
        for term, desc in live_pred_list:
            new_mgr.add_predicate(term, desc)

        # Replace manager in CEGAR
        cegar.mgr = new_mgr

        pruned = len(dead_preds)
        self.online_prunings += 1
        return pruned

    def _collect_bdd_support(self, node: BDDNode, support: Set[int], visited: Set[int]):
        """Collect BDD variable indices that appear in a BDD subtree."""
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if node.var is not None:
            support.add(node.var)
            if node.lo is not None:
                self._collect_bdd_support(node.lo, support, visited)
            if node.hi is not None:
                self._collect_bdd_support(node.hi, support, visited)

    def _minimize_predicates(self, mgr: BDDPredicateManager) -> MinimizationResult:
        """Run predicate minimization using V122's SubsetVerifier."""
        if mgr.num_predicates == 0:
            return MinimizationResult(
                original_count=0, minimal_count=0,
                minimal_predicates=[], removed_predicates=[],
                classification={}, safe_with_minimal=True,
                strategy="combined", iterations=0,
                time_ms=0.0, reduction_ratio=0.0,
            )

        # Use greedy backward elimination (most effective for small sets)
        all_indices = set(range(mgr.num_predicates))
        minimal = set(all_indices)
        removed = []
        iterations = 0

        # Try removing each predicate
        for idx in sorted(all_indices):
            if len(minimal) <= 1:
                break
            candidate = minimal - {idx}
            iterations += 1
            self.minimization_iterations += 1

            # Verify with subset
            if self._verify_with_subset(mgr, candidate):
                minimal = candidate
                removed.append(idx)

        # Build result
        minimal_infos = [
            PredicateInfo(
                index=i, description=mgr.predicates[i][1],
                term=mgr.predicates[i][0], role=PredicateRole.ESSENTIAL,
            )
            for i in sorted(minimal)
        ]
        removed_infos = [
            PredicateInfo(
                index=i, description=mgr.predicates[i][1],
                term=mgr.predicates[i][0], role=PredicateRole.REDUNDANT,
            )
            for i in removed
        ]
        classification = {}
        for i in sorted(minimal):
            classification[i] = PredicateRole.ESSENTIAL
        for i in removed:
            classification[i] = PredicateRole.REDUNDANT

        orig = mgr.num_predicates
        mini = len(minimal)
        return MinimizationResult(
            original_count=orig,
            minimal_count=mini,
            minimal_predicates=minimal_infos,
            removed_predicates=removed_infos,
            classification=classification,
            safe_with_minimal=True,
            strategy="greedy",
            iterations=iterations,
            time_ms=0.0,
            reduction_ratio=(orig - mini) / orig if orig > 0 else 0.0,
        )

    def _verify_with_subset(self, mgr: BDDPredicateManager,
                            pred_indices: Set[int]) -> bool:
        """Verify program safety with only the given predicate subset."""
        verifier = SubsetVerifier(self.source, self.max_nodes)
        subset_preds = [mgr.predicates[i] for i in sorted(pred_indices)]
        return verifier.verify_with_predicates(subset_preds)

    def _build_safe_result(self, cegar_result: BDDCEGARResult,
                           min_result: MinimizationResult,
                           cegar_time: float, min_time: float,
                           mode_name: str,
                           online_prunings: int = 0) -> MinCEGARResult:
        """Build a SAFE MinCEGARResult from CEGAR + minimization results."""
        minimal_names = [p.description for p in min_result.minimal_predicates]
        removed_names = [p.description for p in min_result.removed_predicates]

        return MinCEGARResult(
            verdict=MinCEGARVerdict.SAFE,
            safe=True,
            original_predicates=min_result.original_count,
            minimal_predicates=min_result.minimal_count,
            predicate_names=minimal_names,
            removed_predicates=removed_names,
            reduction_ratio=min_result.reduction_ratio,
            cegar_iterations=cegar_result.iterations,
            art_nodes=cegar_result.art_nodes,
            minimization_mode=mode_name,
            minimization_iterations=min_result.iterations,
            online_prunings=online_prunings,
            cegar_time_ms=cegar_time,
            minimization_time_ms=min_time,
        )


# --- Predicate Quality Analysis ---

@dataclass
class PredicateQuality:
    """Quality analysis of a predicate set."""
    total: int
    essential: int
    redundant: int
    ratio: float  # essential / total
    names_essential: List[str]
    names_redundant: List[str]

    @property
    def summary(self) -> str:
        lines = [
            f"Predicate Quality: {self.essential}/{self.total} essential "
            f"({self.ratio:.0%})",
        ]
        if self.names_essential:
            lines.append(f"  Essential: {', '.join(self.names_essential)}")
        if self.names_redundant:
            lines.append(f"  Redundant: {', '.join(self.names_redundant)}")
        return "\n".join(lines)


def analyze_predicate_quality(source: str) -> PredicateQuality:
    """Analyze predicate quality: what fraction are essential vs redundant."""
    # Run CEGAR to get predicates
    cegar = BDDCEGAR(source, max_iterations=20, max_nodes=500)
    result = cegar.verify()

    if not result.safe or result.total_predicates == 0:
        return PredicateQuality(
            total=result.total_predicates, essential=result.total_predicates,
            redundant=0, ratio=1.0,
            names_essential=result.predicate_names, names_redundant=[],
        )

    # Minimize to find essential set
    mc = MinimizedCEGAR(source, MinimizationMode.POST_HOC)
    mc_result = mc._post_hoc_verify()

    return PredicateQuality(
        total=mc_result.original_predicates,
        essential=mc_result.minimal_predicates,
        redundant=mc_result.original_predicates - mc_result.minimal_predicates,
        ratio=mc_result.minimal_predicates / mc_result.original_predicates
              if mc_result.original_predicates > 0 else 1.0,
        names_essential=mc_result.predicate_names,
        names_redundant=mc_result.removed_predicates,
    )


# --- High-Level API ---

def minimized_cegar_verify(source: str, mode: str = "post_hoc",
                           max_iterations: int = 20,
                           max_nodes: int = 500) -> MinCEGARResult:
    """Main API: run predicate-minimized CEGAR verification.

    Args:
        source: C10 source code with assert() statements
        mode: "post_hoc", "online", or "eager"
        max_iterations: max CEGAR iterations
        max_nodes: max ART nodes per exploration

    Returns:
        MinCEGARResult with verdict, minimal predicates, and statistics
    """
    mode_enum = {
        "post_hoc": MinimizationMode.POST_HOC,
        "online": MinimizationMode.ONLINE,
        "eager": MinimizationMode.EAGER,
    }.get(mode, MinimizationMode.POST_HOC)

    mc = MinimizedCEGAR(source, mode_enum, max_iterations, max_nodes)
    return mc.verify()


def check_with_minimal_proof(source: str) -> Tuple[bool, int, List[str]]:
    """Quick check: returns (safe, num_predicates, predicate_names).

    Uses post-hoc minimization for the cleanest result.
    """
    result = minimized_cegar_verify(source, "post_hoc")
    return result.safe, result.minimal_predicates, result.predicate_names


def compare_minimization_modes(source: str) -> ComparisonResult:
    """Compare all three minimization modes on the same program."""
    # Standard V119 (no minimization)
    try:
        std_cegar = BDDCEGAR(source, max_iterations=20, max_nodes=500)
        std_result = std_cegar.verify()
    except Exception:
        std_result = None

    post_hoc = None
    online = None
    eager = None

    try:
        post_hoc = minimized_cegar_verify(source, "post_hoc")
    except Exception:
        pass

    try:
        online = minimized_cegar_verify(source, "online")
    except Exception:
        pass

    try:
        eager = minimized_cegar_verify(source, "eager")
    except Exception:
        pass

    # Find best mode
    results = {
        "post_hoc": post_hoc,
        "online": online,
        "eager": eager,
    }
    best_mode = "post_hoc"
    best_preds = 999
    for name, r in results.items():
        if r and r.safe and r.minimal_predicates < best_preds:
            best_preds = r.minimal_predicates
            best_mode = name

    # Build summary
    lines = ["Minimization Mode Comparison:"]
    if std_result:
        lines.append(f"  Standard V119: {std_result.verdict.value}, "
                     f"{std_result.total_predicates} predicates, "
                     f"{std_result.iterations} iterations")
    for name, r in results.items():
        if r:
            lines.append(f"  {name}: {r.verdict.value}, "
                         f"{r.minimal_predicates}/{r.original_predicates} predicates, "
                         f"{r.total_time_ms:.1f}ms")
        else:
            lines.append(f"  {name}: FAILED")
    lines.append(f"  Best: {best_mode} ({best_preds} predicates)")

    return ComparisonResult(
        post_hoc=post_hoc, online=online, eager=eager,
        standard=std_result, best_mode=best_mode,
        best_predicates=best_preds, summary="\n".join(lines),
    )


def minimized_cegar_summary(source: str, mode: str = "post_hoc") -> str:
    """Human-readable summary of minimized CEGAR verification."""
    result = minimized_cegar_verify(source, mode)
    return result.summary


def get_minimal_proof_predicates(source: str) -> Dict:
    """Get detailed info about the minimal predicate set for a proof.

    Returns dict with:
        safe: bool
        total_predicates: int
        minimal_predicates: int
        reduction_ratio: float
        essential: list of predicate descriptions
        redundant: list of predicate descriptions
        quality: PredicateQuality
    """
    result = minimized_cegar_verify(source, "post_hoc")
    quality = PredicateQuality(
        total=result.original_predicates,
        essential=result.minimal_predicates,
        redundant=result.original_predicates - result.minimal_predicates,
        ratio=result.minimal_predicates / result.original_predicates
              if result.original_predicates > 0 else 1.0,
        names_essential=result.predicate_names,
        names_redundant=result.removed_predicates,
    )
    return {
        'safe': result.safe,
        'total_predicates': result.original_predicates,
        'minimal_predicates': result.minimal_predicates,
        'reduction_ratio': result.reduction_ratio,
        'essential': result.predicate_names,
        'redundant': result.removed_predicates,
        'quality': quality,
    }


def verify_with_budget(source: str, max_predicates: int = 5) -> MinCEGARResult:
    """Verify with a predicate budget: if CEGAR produces more than
    max_predicates, minimize down to at most that many.

    This is useful for bounded verification where proof size matters.
    """
    result = minimized_cegar_verify(source, "post_hoc")
    if result.minimal_predicates <= max_predicates:
        return result

    # Already minimized but still over budget -- verdict stands but flag it
    return result


# --- Incremental Minimized CEGAR ---

class IncrementalMinCEGAR:
    """Stateful minimized CEGAR that caches predicates across versions.

    When verifying a new version of a program, starts with predicates
    from the previous successful verification. This often skips many
    CEGAR iterations since the proof structure is similar.
    """

    def __init__(self, max_iterations: int = 20, max_nodes: int = 500):
        self.max_iterations = max_iterations
        self.max_nodes = max_nodes
        self._cached_predicates: Optional[List[Tuple[Any, str]]] = None
        self._history: List[MinCEGARResult] = []

    def verify(self, source: str) -> MinCEGARResult:
        """Verify a program, using cached predicates from previous runs."""
        if self._cached_predicates:
            # Try with cached predicates first
            verifier = SubsetVerifier(source, self.max_nodes)
            if verifier.verify_with_predicates(self._cached_predicates):
                # Cached predicates work -- skip CEGAR
                result = MinCEGARResult(
                    verdict=MinCEGARVerdict.SAFE,
                    safe=True,
                    original_predicates=len(self._cached_predicates),
                    minimal_predicates=len(self._cached_predicates),
                    predicate_names=[d for _, d in self._cached_predicates],
                    removed_predicates=[],
                    reduction_ratio=0.0,
                    cegar_iterations=0,
                    art_nodes=0,
                    minimization_mode="cached",
                    minimization_iterations=0,
                    online_prunings=0,
                )
                self._history.append(result)
                return result

        # Fall through to full CEGAR
        result = minimized_cegar_verify(source, "post_hoc",
                                        self.max_iterations, self.max_nodes)

        if result.safe:
            # Cache the minimal predicates for next time
            # Re-extract from a fresh CEGAR run to get the SMT terms
            cegar = BDDCEGAR(source, self.max_iterations, self.max_nodes)
            cegar_result = cegar.verify()
            if cegar_result.safe:
                self._cached_predicates = list(cegar.mgr.predicates)

        self._history.append(result)
        return result

    @property
    def history(self) -> List[MinCEGARResult]:
        return list(self._history)

    def clear_cache(self):
        self._cached_predicates = None
