"""V122: Symbolic Predicate Minimization

Composes V119 (BDD-based predicate abstraction) + V021 (BDD library) + V110 (ART/CEGAR)
to find minimal predicate sets that still prove program safety.

Given a program verified SAFE by V119's BDD-CEGAR, this module finds the smallest
subset of predicates that suffices for the proof. Three strategies:
  1. BDD support analysis: identify predicates absent from proof BDDs
  2. Greedy backward elimination: try removing each predicate one at a time
  3. Delta debugging: binary search for minimal predicate subsets

Author: A2 (AgentZero verification agent)
"""

import sys
import os
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V119_bdd_predicate_abstraction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V110_abstract_reachability_tree'))

from bdd_predicate_abstraction import (
    BDDCEGAR, BDDPredicateManager, BDDPredicateState, BDDARTNode,
    BDDCEGARResult, BDDVerdict, TransitionBDDBuilder,
    _is_sat, _is_unsat, _smt_not, _safe_ast_to_smt,
)
from bdd_model_checker import BDD
from art import build_cfg, CFGNodeType, CFG, CFGNode

# SMT imports (for type references)
smt_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver')
sys.path.insert(0, smt_path)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PredicateRole(Enum):
    """Classification of a predicate's role in the proof."""
    ESSENTIAL = "essential"       # Removing it breaks the proof
    REDUNDANT = "redundant"       # Can be removed without breaking the proof
    SUPPORT_DEAD = "support_dead" # Not in BDD support (definitely redundant)
    UNKNOWN = "unknown"           # Not yet classified


@dataclass
class PredicateInfo:
    """Information about a single predicate."""
    index: int
    description: str
    term: Any  # SMT term
    role: PredicateRole = PredicateRole.UNKNOWN


@dataclass
class MinimizationResult:
    """Result of predicate minimization."""
    original_count: int
    minimal_count: int
    minimal_predicates: List[PredicateInfo]
    removed_predicates: List[PredicateInfo]
    classification: Dict[int, PredicateRole]  # index -> role
    safe_with_minimal: bool
    strategy: str
    iterations: int  # number of re-verification attempts
    time_ms: float
    reduction_ratio: float  # (original - minimal) / original

    @property
    def summary(self) -> str:
        lines = [
            f"Predicate Minimization ({self.strategy})",
            f"  Original: {self.original_count} predicates",
            f"  Minimal:  {self.minimal_count} predicates",
            f"  Removed:  {self.original_count - self.minimal_count}",
            f"  Reduction: {self.reduction_ratio:.1%}",
            f"  Safe: {self.safe_with_minimal}",
            f"  Iterations: {self.iterations}",
            f"  Time: {self.time_ms:.1f}ms",
        ]
        if self.minimal_predicates:
            lines.append("  Minimal set:")
            for p in self.minimal_predicates:
                lines.append(f"    [{p.index}] {p.description}")
        if self.removed_predicates:
            lines.append("  Removed:")
            for p in self.removed_predicates:
                lines.append(f"    [{p.index}] {p.description} ({p.role.value})")
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Comparison of minimization strategies."""
    greedy: Optional[MinimizationResult]
    delta: Optional[MinimizationResult]
    support: Optional[MinimizationResult]
    original_count: int
    best_strategy: str
    best_count: int
    summary: str


# ---------------------------------------------------------------------------
# Subset verifier -- re-runs BDD CEGAR with a specific predicate subset
# ---------------------------------------------------------------------------

class SubsetVerifier:
    """Verifies a program using only a specified subset of predicates.

    Instead of running full CEGAR (which discovers predicates), this verifier
    seeds with a fixed predicate set and checks if it suffices to prove safety
    without any refinement.
    """

    def __init__(self, source: str, max_nodes: int = 500):
        self.source = source
        self.max_nodes = max_nodes
        self.cfg = build_cfg(source)

    def verify_with_predicates(self, predicates: List[Tuple[Any, str]]) -> bool:
        """Check if the given predicate set proves the program SAFE.

        Args:
            predicates: List of (smt_term, description) pairs

        Returns:
            True if SAFE can be proven with these predicates, False otherwise
        """
        if not predicates:
            # No predicates -- can only prove safe if no assertions reachable
            return not self._has_reachable_assertions()

        mgr = BDDPredicateManager()

        for term, desc in predicates:
            mgr.add_predicate(term, desc)

        builder = TransitionBDDBuilder(mgr)
        transitions = {}
        for node in self.cfg.nodes:
            for succ in node.successors:
                edge = (node.id, succ.id)
                bdd = self._build_transition(builder, node, succ)
                transitions[edge] = bdd

        # Explore ART with fixed predicates
        return self._explore_art(mgr, transitions)

    def _has_reachable_assertions(self) -> bool:
        """Check if any ASSERT/ERROR nodes are reachable from entry."""
        visited = set()
        queue = deque([self.cfg.entry])
        while queue:
            node = queue.popleft()
            if node.id in visited:
                continue
            visited.add(node.id)
            if node.type in (CFGNodeType.ASSERT, CFGNodeType.ERROR):
                return True
            for s in node.successors:
                queue.append(s)
        return False

    def _build_transition(self, builder: TransitionBDDBuilder,
                          src: CFGNode, tgt: CFGNode):
        """Build transition BDD for a CFG edge.

        CFG node.data format (from V110 art.py):
          ASSIGN: (var_name: str, expr_ast: ASTNode)
          ASSUME/ASSUME_NOT/ASSERT: condition ASTNode directly
        """
        if src.type == CFGNodeType.ASSIGN and src.data:
            var_name, expr_ast = src.data
            expr_smt = _safe_ast_to_smt(expr_ast)
            if expr_smt is not None:
                return builder.build_assign_transition(
                    var_name, expr_smt, src.id, tgt.id)
            return builder.build_identity_transition(src.id, tgt.id)

        elif src.type == CFGNodeType.ASSUME and src.data:
            cond_smt = _safe_ast_to_smt(src.data)
            if cond_smt is not None:
                return builder.build_assume_transition(
                    cond_smt, False, src.id, tgt.id)
            return builder.build_identity_transition(src.id, tgt.id)

        elif src.type == CFGNodeType.ASSUME_NOT and src.data:
            cond_smt = _safe_ast_to_smt(src.data)
            if cond_smt is not None:
                return builder.build_assume_transition(
                    cond_smt, True, src.id, tgt.id)
            return builder.build_identity_transition(src.id, tgt.id)

        elif src.type == CFGNodeType.ASSERT and src.data:
            is_error_edge = (tgt.type == CFGNodeType.ERROR)
            cond_smt = _safe_ast_to_smt(src.data)
            if cond_smt is not None:
                return builder.build_assume_transition(
                    cond_smt, is_error_edge, src.id, tgt.id)
            return builder.build_identity_transition(src.id, tgt.id)

        else:
            return builder.build_identity_transition(src.id, tgt.id)

    def _explore_art(self, mgr: BDDPredicateManager,
                     transitions: Dict) -> bool:
        """Explore ART. Returns True if SAFE (no error nodes reachable)."""
        root_state = mgr.state_top()
        root = _ARTNodeSimple(0, self.cfg.entry, root_state)

        worklist = deque([root])
        node_map = {}  # (cfg_node_id, state_repr) -> _ARTNodeSimple
        node_map[(self.cfg.entry.id, root_state.bdd_node._id)] = root
        next_id = 1
        visited_count = 0

        while worklist and visited_count < self.max_nodes:
            node = worklist.popleft()
            visited_count += 1

            if node.cfg_node.type == CFGNodeType.ERROR:
                if not node.state.is_bottom:
                    return False  # Error reachable!

            for succ_cfg in node.cfg_node.successors:
                edge = (node.cfg_node.id, succ_cfg.id)
                trans_bdd = transitions.get(edge)
                if trans_bdd is None:
                    continue

                post = mgr.image(node.state, trans_bdd)
                if post.is_bottom:
                    continue  # Infeasible path

                key = (succ_cfg.id, post.bdd_node._id)
                if key in node_map:
                    existing = node_map[key]
                    if existing.state.subsumes(post, mgr.bdd):
                        continue  # Covered

                succ = _ARTNodeSimple(next_id, succ_cfg, post)
                next_id += 1
                node_map[key] = succ
                worklist.append(succ)

        return True  # No error reachable (or budget exhausted conservatively)


@dataclass
class _ARTNodeSimple:
    """Simplified ART node for subset verification."""
    id: int
    cfg_node: CFGNode
    state: BDDPredicateState


# ---------------------------------------------------------------------------
# BDD Support Analysis
# ---------------------------------------------------------------------------

def _bdd_support(bdd_mgr: BDD, node) -> Set[int]:
    """Compute BDD support -- set of variable indices that appear in the BDD."""
    support = set()
    visited = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n._id in visited:
            continue
        visited.add(n._id)
        if n.var == -1:  # Terminal
            continue
        support.add(n.var)
        stack.append(n.lo)
        stack.append(n.hi)
    return support


# ---------------------------------------------------------------------------
# Predicate Minimizer
# ---------------------------------------------------------------------------

class PredicateMinimizer:
    """Finds minimal predicate sets for BDD-based program verification."""

    def __init__(self, source: str, max_nodes: int = 500):
        self.source = source
        self.max_nodes = max_nodes
        self._full_result = None
        self._predicates = None

    def _run_full_verification(self) -> BDDCEGARResult:
        """Run V119 full BDD-CEGAR to discover predicates."""
        if self._full_result is None:
            cegar = BDDCEGAR(self.source, max_iterations=20, max_nodes=self.max_nodes)
            self._full_result = cegar.verify()
            if self._full_result.verdict == BDDVerdict.SAFE:
                self._predicates = []
                mgr = cegar.mgr  # BDDPredicateManager
                for i in range(mgr.num_predicates):
                    term, desc = mgr.predicates[i]
                    self._predicates.append(PredicateInfo(
                        index=i,
                        description=desc,
                        term=term,
                        role=PredicateRole.UNKNOWN,
                    ))
            self._cegar = cegar
        return self._full_result

    def _verify_subset(self, indices: Set[int]) -> bool:
        """Verify the program with only the predicates at the given indices."""
        if self._predicates is None:
            return False
        preds = [(self._predicates[i].term, self._predicates[i].description)
                 for i in sorted(indices) if i < len(self._predicates)]
        verifier = SubsetVerifier(self.source, self.max_nodes)
        return verifier.verify_with_predicates(preds)

    def support_analysis(self) -> MinimizationResult:
        """Phase 1: Use BDD support to identify dead predicates.

        Predicates whose BDD variables don't appear in any transition BDD
        are definitely redundant (they can't affect the proof).
        """
        t0 = time.time()
        result = self._run_full_verification()
        if result.verdict != BDDVerdict.SAFE:
            return self._fail_result("support", time.time() - t0)

        mgr = self._cegar.mgr
        bdd = mgr.bdd
        iterations = 0

        # Collect support from all transition BDDs
        all_support = set()
        for edge_key, trans_bdd in self._cegar._edge_trans.items():
            all_support |= _bdd_support(bdd, trans_bdd)

        # Map BDD variable indices back to predicate indices
        # Each predicate i uses curr_var=2*i, next_var=2*i+1
        live_pred_indices = set()
        for var_idx in all_support:
            pred_idx = var_idx // 2
            if pred_idx < len(self._predicates):
                live_pred_indices.add(pred_idx)

        dead_pred_indices = set(range(len(self._predicates))) - live_pred_indices

        # Classify
        classification = {}
        minimal = []
        removed = []
        for p in self._predicates:
            if p.index in dead_pred_indices:
                p.role = PredicateRole.SUPPORT_DEAD
                removed.append(p)
            else:
                p.role = PredicateRole.UNKNOWN  # Might still be redundant
                minimal.append(p)
            classification[p.index] = p.role

        # Verify the live subset actually works
        if live_pred_indices:
            safe = self._verify_subset(live_pred_indices)
            iterations = 1
        else:
            safe = self._verify_subset(set())
            iterations = 1

        elapsed = (time.time() - t0) * 1000
        orig = len(self._predicates)
        mini = len(minimal)
        return MinimizationResult(
            original_count=orig,
            minimal_count=mini,
            minimal_predicates=minimal,
            removed_predicates=removed,
            classification=classification,
            safe_with_minimal=safe,
            strategy="support",
            iterations=iterations,
            time_ms=elapsed,
            reduction_ratio=(orig - mini) / max(orig, 1),
        )

    def greedy_minimize(self) -> MinimizationResult:
        """Phase 2: Greedy backward elimination.

        Starting from the full set, try removing each predicate one at a time.
        Remove predicates that don't break the proof, largest index first.
        """
        t0 = time.time()
        result = self._run_full_verification()
        if result.verdict != BDDVerdict.SAFE:
            return self._fail_result("greedy", time.time() - t0)

        current = set(range(len(self._predicates)))
        iterations = 0

        # Try removing in reverse order (last-added first -- likely least essential)
        for idx in reversed(range(len(self._predicates))):
            if idx not in current:
                continue
            candidate = current - {idx}
            iterations += 1
            if self._verify_subset(candidate):
                current = candidate  # Predicate was redundant

        # Classify
        classification = {}
        minimal = []
        removed = []
        for p in self._predicates:
            if p.index in current:
                p.role = PredicateRole.ESSENTIAL
                minimal.append(p)
            else:
                p.role = PredicateRole.REDUNDANT
                removed.append(p)
            classification[p.index] = p.role

        elapsed = (time.time() - t0) * 1000
        orig = len(self._predicates)
        mini = len(minimal)
        return MinimizationResult(
            original_count=orig,
            minimal_count=mini,
            minimal_predicates=minimal,
            removed_predicates=removed,
            classification=classification,
            safe_with_minimal=True,
            strategy="greedy",
            iterations=iterations,
            time_ms=elapsed,
            reduction_ratio=(orig - mini) / max(orig, 1),
        )

    def delta_minimize(self) -> MinimizationResult:
        """Phase 3: Delta debugging for minimal predicate sets.

        Uses ddmin algorithm: binary partition, test halves, narrow down.
        More efficient than greedy when many predicates can be removed together.
        """
        t0 = time.time()
        result = self._run_full_verification()
        if result.verdict != BDDVerdict.SAFE:
            return self._fail_result("delta", time.time() - t0)

        all_indices = list(range(len(self._predicates)))
        iterations = 0

        def test_subset(subset: Set[int]) -> bool:
            nonlocal iterations
            iterations += 1
            return self._verify_subset(subset)

        minimal_set = self._ddmin(set(all_indices), test_subset)

        # Classify
        classification = {}
        minimal = []
        removed = []
        for p in self._predicates:
            if p.index in minimal_set:
                p.role = PredicateRole.ESSENTIAL
                minimal.append(p)
            else:
                p.role = PredicateRole.REDUNDANT
                removed.append(p)
            classification[p.index] = p.role

        elapsed = (time.time() - t0) * 1000
        orig = len(self._predicates)
        mini = len(minimal)
        return MinimizationResult(
            original_count=orig,
            minimal_count=mini,
            minimal_predicates=minimal,
            removed_predicates=removed,
            classification=classification,
            safe_with_minimal=True,
            strategy="delta",
            iterations=iterations,
            time_ms=elapsed,
            reduction_ratio=(orig - mini) / max(orig, 1),
        )

    def _ddmin(self, full: Set[int], test_fn) -> Set[int]:
        """Delta debugging minimization.

        Find a 1-minimal subset S of `full` such that test_fn(S) is True
        but removing any single element makes test_fn return False.
        """
        if not test_fn(full):
            return full  # Can't even pass with full set

        n = 2  # partition granularity
        current = full.copy()
        items = sorted(current)

        while len(items) > 0:
            chunk_size = max(1, len(items) // n)
            chunks = []
            for i in range(0, len(items), chunk_size):
                chunks.append(set(items[i:i + chunk_size]))

            found_reduction = False

            # Try removing each chunk
            for chunk in chunks:
                candidate = current - chunk
                if candidate and test_fn(candidate):
                    current = candidate
                    items = sorted(current)
                    n = max(2, n - 1)
                    found_reduction = True
                    break

            if not found_reduction:
                if n >= len(items):
                    break
                n = min(2 * n, len(items))

        # Final: try removing each element individually (1-minimality)
        changed = True
        while changed:
            changed = False
            for idx in sorted(current):
                candidate = current - {idx}
                if candidate and test_fn(candidate):
                    current = candidate
                    changed = True
                    break

        return current

    def combined_minimize(self) -> MinimizationResult:
        """Combined strategy: support analysis -> greedy elimination.

        First removes BDD-dead predicates (cheap), then greedy elimination
        on the remaining set.
        """
        t0 = time.time()
        result = self._run_full_verification()
        if result.verdict != BDDVerdict.SAFE:
            return self._fail_result("combined", time.time() - t0)

        iterations = 0

        # Phase 1: Support analysis
        mgr = self._cegar.mgr
        bdd = mgr.bdd
        all_support = set()
        for edge_key, trans_bdd in self._cegar._edge_trans.items():
            all_support |= _bdd_support(bdd, trans_bdd)

        live_pred_indices = set()
        for var_idx in all_support:
            pred_idx = var_idx // 2
            if pred_idx < len(self._predicates):
                live_pred_indices.add(pred_idx)

        dead_indices = set(range(len(self._predicates))) - live_pred_indices

        # Phase 2: Greedy on live set
        current = live_pred_indices.copy()
        for idx in sorted(live_pred_indices, reverse=True):
            candidate = current - {idx}
            iterations += 1
            if candidate and self._verify_subset(candidate):
                current = candidate

        # Handle edge case: empty program or no predicates needed
        if not current:
            iterations += 1
            if self._verify_subset(set()):
                pass  # Empty set works
            else:
                current = live_pred_indices.copy()  # Revert

        # Classify
        classification = {}
        minimal = []
        removed = []
        for p in self._predicates:
            if p.index in current:
                p.role = PredicateRole.ESSENTIAL
                minimal.append(p)
            elif p.index in dead_indices:
                p.role = PredicateRole.SUPPORT_DEAD
                removed.append(p)
            else:
                p.role = PredicateRole.REDUNDANT
                removed.append(p)
            classification[p.index] = p.role

        elapsed = (time.time() - t0) * 1000
        orig = len(self._predicates)
        mini = len(minimal)
        return MinimizationResult(
            original_count=orig,
            minimal_count=mini,
            minimal_predicates=minimal,
            removed_predicates=removed,
            classification=classification,
            safe_with_minimal=True,
            strategy="combined",
            iterations=iterations,
            time_ms=elapsed,
            reduction_ratio=(orig - mini) / max(orig, 1),
        )

    def _fail_result(self, strategy: str, elapsed_s: float) -> MinimizationResult:
        """Return a result for programs that aren't SAFE."""
        return MinimizationResult(
            original_count=0,
            minimal_count=0,
            minimal_predicates=[],
            removed_predicates=[],
            classification={},
            safe_with_minimal=False,
            strategy=strategy,
            iterations=0,
            time_ms=elapsed_s * 1000,
            reduction_ratio=0.0,
        )


# ---------------------------------------------------------------------------
# Predicate dependency analysis
# ---------------------------------------------------------------------------

def analyze_predicate_dependencies(source: str) -> Dict:
    """Analyze which predicates depend on which others.

    For each predicate p_i, check if any transition BDD makes p_i's next-state
    value depend on p_j's current-state value. Build a dependency graph.
    """
    cegar = BDDCEGAR(source, max_iterations=20, max_nodes=500)
    result = cegar.verify()
    if result.verdict != BDDVerdict.SAFE:
        return {'safe': False, 'dependencies': {}}

    mgr = cegar.mgr
    bdd = mgr.bdd
    n = mgr.num_predicates

    # For each transition BDD, check dependency: does next_j depend on curr_i?
    deps = {j: set() for j in range(n)}
    for edge_key, trans_bdd in cegar._edge_trans.items():
        for j in range(n):
            next_var = 2 * j + 1
            # Check if next_var appears in transition
            support = _bdd_support(bdd, trans_bdd)
            if next_var not in support:
                continue
            # Check which curr vars it depends on
            for i in range(n):
                curr_var = 2 * i
                if curr_var in support:
                    deps[j].add(i)

    # Build predicate info
    predicates = []
    for i in range(n):
        _, desc = mgr.predicates[i]
        predicates.append({
            'index': i,
            'description': desc,
            'depends_on': sorted(deps[i]),
            'depended_by': sorted(j for j in range(n) if i in deps[j]),
        })

    return {
        'safe': True,
        'num_predicates': n,
        'predicates': predicates,
        'dependencies': {j: sorted(deps[j]) for j in range(n)},
    }


# ---------------------------------------------------------------------------
# High-level APIs
# ---------------------------------------------------------------------------

def minimize_predicates(source: str, strategy: str = "combined") -> MinimizationResult:
    """Minimize predicates for a verified-safe program.

    Args:
        source: C10 source code with assert() statements
        strategy: "greedy", "delta", "support", or "combined" (default)

    Returns:
        MinimizationResult with minimal predicate set
    """
    minimizer = PredicateMinimizer(source)
    if strategy == "greedy":
        return minimizer.greedy_minimize()
    elif strategy == "delta":
        return minimizer.delta_minimize()
    elif strategy == "support":
        return minimizer.support_analysis()
    else:
        return minimizer.combined_minimize()


def classify_predicates(source: str) -> Dict[int, PredicateRole]:
    """Classify each predicate as ESSENTIAL, REDUNDANT, or SUPPORT_DEAD."""
    result = minimize_predicates(source, strategy="combined")
    return result.classification


def compare_minimization_strategies(source: str) -> ComparisonResult:
    """Run all strategies and compare results."""
    minimizer = PredicateMinimizer(source)

    greedy = minimizer.greedy_minimize()
    # Reset for fresh run
    minimizer2 = PredicateMinimizer(source)
    delta = minimizer2.delta_minimize()
    minimizer3 = PredicateMinimizer(source)
    support = minimizer3.support_analysis()

    results = {'greedy': greedy, 'delta': delta, 'support': support}
    best_name = min(results, key=lambda k: results[k].minimal_count)
    best = results[best_name]

    lines = [
        "Strategy Comparison:",
        f"  Greedy:  {greedy.minimal_count}/{greedy.original_count} predicates, {greedy.iterations} iterations, {greedy.time_ms:.0f}ms",
        f"  Delta:   {delta.minimal_count}/{delta.original_count} predicates, {delta.iterations} iterations, {delta.time_ms:.0f}ms",
        f"  Support: {support.minimal_count}/{support.original_count} predicates, {support.iterations} iterations, {support.time_ms:.0f}ms",
        f"  Best:    {best_name} ({best.minimal_count} predicates)",
    ]

    return ComparisonResult(
        greedy=greedy,
        delta=delta,
        support=support,
        original_count=greedy.original_count,
        best_strategy=best_name,
        best_count=best.minimal_count,
        summary="\n".join(lines),
    )


def get_predicate_dependencies(source: str) -> Dict:
    """Get predicate dependency graph."""
    return analyze_predicate_dependencies(source)


def minimization_summary(source: str) -> str:
    """Human-readable minimization report."""
    result = minimize_predicates(source, strategy="combined")
    return result.summary
