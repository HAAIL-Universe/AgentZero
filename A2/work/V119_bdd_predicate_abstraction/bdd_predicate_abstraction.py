"""V119: BDD-Based Predicate Abstraction

Composes V021 (BDD) + V110 (ART/CEGAR) + V115 (Predicate-Guided CEGAR)

Instead of per-predicate SMT queries for abstract post computation,
encodes the abstract transition relation as a BDD. Once built, abstract
post becomes a single BDD image operation -- amortizing the SMT cost
across all future abstract post computations at that CFG edge.

Key idea (Cartesian abstraction):
  For each CFG edge (assignment/assume) and each predicate p_i:
    - Determine which input predicates p_j imply p_i holds after the edge
    - Encode as BDD clause: (b_j => b_i') for each such implication
  The full transition BDD is the conjunction of all such clauses.

Abstract post: image(current_state_bdd, edge_trans_bdd)
  = exists curr_vars. (current_state_bdd AND edge_trans_bdd)[next->curr]
"""

import sys
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Set, FrozenSet

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V110_abstract_reachability_tree'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V115_predicate_guided_cegar'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V114_recursive_predicate_discovery'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from bdd import BDD, BDDNode
from abstract_reachability_tree import (
    build_cfg, CFGNode, CFGNodeType,
    _ast_to_smt, _substitute_smt
)
from smt_solver import SMTSolver, Var, IntConst, BoolConst, App, Op, Sort, SortKind

# Try importing V115 for predicate discovery
try:
    from predicate_guided_cegar import (
        guided_verify as v115_guided_verify,
        get_discovered_predicates as v115_get_predicates,
    )
    HAS_V115 = True
except ImportError:
    HAS_V115 = False

# Try importing V114 for predicate discovery
try:
    from recursive_predicate_discovery import (
        discover_predicates as v114_discover,
        discover_inductive_predicates as v114_discover_inductive,
    )
    HAS_V114 = True
except ImportError:
    HAS_V114 = False


INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ---------- Data structures ----------

class BDDVerdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class BDDPredicateState:
    """Abstract state as a BDD over predicate variables."""
    bdd_node: BDDNode
    is_bottom: bool = False

    def subsumes(self, other: 'BDDPredicateState', mgr: BDD) -> bool:
        """self >= other iff other => self (other's states are subset of self's)."""
        if self.is_bottom:
            return other.is_bottom
        if other.is_bottom:
            return True
        # other => self iff NOT(other) OR self is tautology
        # Equivalently: other AND NOT(self) is FALSE
        neg_self = mgr.NOT(self.bdd_node)
        check = mgr.AND(other.bdd_node, neg_self)
        return check == mgr.FALSE

    def join(self, other: 'BDDPredicateState', mgr: BDD) -> 'BDDPredicateState':
        """Over-approximation: OR of both states."""
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        return BDDPredicateState(mgr.OR(self.bdd_node, other.bdd_node))


@dataclass
class BDDARTNode:
    """Node in the BDD-based Abstract Reachability Tree."""
    id: int
    cfg_node_id: int
    cfg_node_type: CFGNodeType
    state: BDDPredicateState
    parent: Optional['BDDARTNode'] = None
    children: List['BDDARTNode'] = field(default_factory=list)
    covered_by: Optional['BDDARTNode'] = None
    depth: int = 0

    @property
    def is_covered(self) -> bool:
        return self.covered_by is not None

    @property
    def is_error(self) -> bool:
        return self.cfg_node_type == CFGNodeType.ERROR


@dataclass
class TransitionBDD:
    """Abstract transition BDD for one CFG edge."""
    source_id: int
    target_id: int
    bdd_node: BDDNode  # over curr_vars and next_vars


@dataclass
class BDDCEGARResult:
    """Result of BDD-based predicate abstraction CEGAR."""
    verdict: BDDVerdict
    safe: bool
    counterexample: Optional[List] = None
    counterexample_inputs: Optional[Dict] = None
    iterations: int = 0
    total_predicates: int = 0
    predicate_names: List[str] = field(default_factory=list)
    art_nodes: int = 0
    transition_bdds_built: int = 0
    bdd_image_ops: int = 0
    smt_queries_saved: int = 0
    total_time_ms: float = 0.0


@dataclass
class ComparisonResult:
    """BDD vs SMT-based predicate abstraction comparison."""
    bdd_result: Optional[BDDCEGARResult] = None
    smt_result: Optional[Dict] = None
    bdd_time_ms: float = 0.0
    smt_time_ms: float = 0.0
    both_agree: bool = True
    bdd_image_ops: int = 0
    smt_queries: int = 0
    summary: str = ""


# ---------- Predicate Manager ----------

class BDDPredicateManager:
    """Maps predicates (SMT terms) to BDD variables.

    Each predicate p_i gets two BDD variables:
      - curr_var[i]: represents p_i in current state
      - next_var[i]: represents p_i in next state (primed)
    """

    def __init__(self):
        self.bdd = BDD()
        self.predicates: List[Tuple[object, str]] = []  # (smt_term, description)
        self.pred_map: Dict[str, int] = {}  # str(term) -> index
        self.curr_vars: List[int] = []  # BDD variable indices for current state
        self.next_vars: List[int] = []  # BDD variable indices for next state
        self._curr_bdds: List[BDDNode] = []
        self._next_bdds: List[BDDNode] = []

    @property
    def num_predicates(self) -> int:
        return len(self.predicates)

    def add_predicate(self, term, description: str = "") -> int:
        """Add a predicate. Returns its index."""
        key = str(term)
        if key in self.pred_map:
            return self.pred_map[key]

        idx = len(self.predicates)
        self.predicates.append((term, description or key))
        self.pred_map[key] = idx

        # Allocate two BDD variables: curr (2*idx) and next (2*idx+1)
        curr_bdd_idx = 2 * idx
        next_bdd_idx = 2 * idx + 1

        # Ensure BDD has enough variables
        while self.bdd.num_vars <= next_bdd_idx:
            self.bdd.num_vars += 1

        self.curr_vars.append(curr_bdd_idx)
        self.next_vars.append(next_bdd_idx)
        self._curr_bdds.append(self.bdd.var(curr_bdd_idx))
        self._next_bdds.append(self.bdd.var(next_bdd_idx))

        return idx

    def curr_bdd(self, pred_idx: int) -> BDDNode:
        """BDD variable for predicate in current state."""
        return self._curr_bdds[pred_idx]

    def next_bdd(self, pred_idx: int) -> BDDNode:
        """BDD variable for predicate in next state."""
        return self._next_bdds[pred_idx]

    def state_top(self) -> BDDPredicateState:
        """Top state: any predicate valuation is possible."""
        return BDDPredicateState(self.bdd.TRUE)

    def state_bottom(self) -> BDDPredicateState:
        """Bottom state: no concrete states."""
        return BDDPredicateState(self.bdd.FALSE, is_bottom=True)

    def state_from_predicates(self, pred_indices: Set[int]) -> BDDPredicateState:
        """State where exactly the given predicates are known true."""
        if not pred_indices:
            return self.state_top()
        bdd_node = self.bdd.TRUE
        for idx in pred_indices:
            bdd_node = self.bdd.AND(bdd_node, self.curr_bdd(idx))
        return BDDPredicateState(bdd_node)

    def image(self, state: BDDPredicateState, trans: BDDNode) -> BDDPredicateState:
        """Compute abstract post via BDD image.

        image(S, T) = exists curr. (S(curr) AND T(curr, next))[next -> curr]
        """
        if state.is_bottom:
            return self.state_bottom()

        # Conjoin state with transition
        combined = self.bdd.AND(state.bdd_node, trans)
        if combined == self.bdd.FALSE:
            return self.state_bottom()

        # Existentially quantify current-state variables
        result = self.bdd.exists_multi(self.curr_vars, combined)
        if result == self.bdd.FALSE:
            return self.state_bottom()

        # Rename next vars to current vars
        rename_map = {}
        for i in range(self.num_predicates):
            rename_map[self.next_vars[i]] = self.curr_vars[i]
        result = self.bdd.rename(result, rename_map)

        return BDDPredicateState(result)


# ---------- Transition BDD Builder ----------

class TransitionBDDBuilder:
    """Builds abstract transition BDDs for CFG edges using SMT queries.

    For each edge and predicate pair (p_i, p_j'), determines whether
    knowing p_i in the current state implies p_j in the next state.
    """

    def __init__(self, mgr: BDDPredicateManager):
        self.mgr = mgr
        self.smt_queries = 0
        self._cache: Dict[Tuple[int, int], TransitionBDD] = {}

    def build_assign_transition(self, var_name: str, expr_ast,
                                source_id: int, target_id: int) -> TransitionBDD:
        """Build transition BDD for an assignment: var := expr."""
        cache_key = (source_id, target_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bdd = self.mgr.bdd
        n = self.mgr.num_predicates
        trans = bdd.TRUE

        for j in range(n):
            # For predicate p_j in next state:
            # Check which current predicates imply p_j after assignment
            pred_term, _ = self.mgr.predicates[j]

            # Compute p_j[var := expr] (substituted predicate)
            subst_term = _substitute_smt(pred_term, var_name, expr_ast)

            # Check: does each current predicate p_i imply subst_term?
            # Also check: does subst_term hold unconditionally?
            # And: does subst_term hold given no predicates?

            # Strategy: Cartesian abstraction
            # For each p_j': build clause relating curr vars to p_j'
            # p_j' is implied if: conjunction of assumed curr preds => subst_term
            # We approximate with per-predicate implications

            # Check if subst_term is always true (tautology)
            solver = SMTSolver()
            _declare_vars(solver, pred_term)
            _declare_vars(solver, subst_term)
            solver.add(_smt_not(subst_term))
            self.smt_queries += 1
            if solver.check() == "unsat":
                # p_j' always holds after this assignment
                trans = bdd.AND(trans, self.mgr.next_bdd(j))
                continue

            # Check if subst_term is always false
            solver2 = SMTSolver()
            _declare_vars(solver2, pred_term)
            _declare_vars(solver2, subst_term)
            solver2.add(subst_term)
            self.smt_queries += 1
            if solver2.check() == "unsat":
                # p_j' never holds
                trans = bdd.AND(trans, bdd.NOT(self.mgr.next_bdd(j)))
                continue

            # Per-predicate implications
            implies_clauses = []
            for i in range(n):
                src_term, _ = self.mgr.predicates[i]
                # Check: p_i => subst_term?
                s = SMTSolver()
                _declare_vars(s, src_term)
                _declare_vars(s, subst_term)
                s.add(src_term)
                s.add(_smt_not(subst_term))
                self.smt_queries += 1
                if s.check() == "unsat":
                    # p_i implies p_j after assignment
                    # curr_i => next_j: NOT(curr_i) OR next_j
                    clause = bdd.OR(
                        bdd.NOT(self.mgr.curr_bdd(i)),
                        self.mgr.next_bdd(j)
                    )
                    implies_clauses.append(clause)

            # Also check: NOT(p_i) => subst_term? (negation implies next)
            for i in range(n):
                src_term, _ = self.mgr.predicates[i]
                s = SMTSolver()
                _declare_vars(s, src_term)
                _declare_vars(s, subst_term)
                s.add(_smt_not(src_term))
                s.add(_smt_not(subst_term))
                self.smt_queries += 1
                if s.check() == "unsat":
                    # NOT(p_i) implies p_j after assignment
                    clause = bdd.OR(
                        self.mgr.curr_bdd(i),
                        self.mgr.next_bdd(j)
                    )
                    implies_clauses.append(clause)

            if implies_clauses:
                for clause in implies_clauses:
                    trans = bdd.AND(trans, clause)
            # If no implications found, p_j' is unconstrained (TRUE for that bit)

        result = TransitionBDD(source_id, target_id, trans)
        self._cache[cache_key] = result
        return result

    def build_assume_transition(self, condition_ast, is_negated: bool,
                                source_id: int, target_id: int) -> TransitionBDD:
        """Build transition BDD for assume(cond) or assume(!cond).

        Predicates that are implied by the condition become true in next state.
        Predicates contradicted by the condition make the transition infeasible.
        Other predicates are preserved (frame condition).
        """
        cache_key = (source_id, target_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bdd = self.mgr.bdd
        n = self.mgr.num_predicates
        cond_term = _safe_ast_to_smt(condition_ast)

        if cond_term is None:
            # Can't convert condition: identity transition (preserve all)
            result = self._build_identity_transition(source_id, target_id)
            self._cache[cache_key] = result
            return result

        if is_negated:
            cond_term = _smt_not(cond_term)

        trans = bdd.TRUE

        for j in range(n):
            pred_term, _ = self.mgr.predicates[j]

            # Frame: predicates are preserved through assumes
            # curr_j <=> next_j (unless condition implies/contradicts)
            frame = bdd.IFF(self.mgr.curr_bdd(j), self.mgr.next_bdd(j))

            # Check if condition implies predicate
            s = SMTSolver()
            _declare_vars(s, cond_term)
            _declare_vars(s, pred_term)
            s.add(cond_term)
            s.add(_smt_not(pred_term))
            self.smt_queries += 1
            if s.check() == "unsat":
                # Condition implies p_j: force next_j = true
                trans = bdd.AND(trans, self.mgr.next_bdd(j))
                continue

            # Check if condition contradicts predicate
            s2 = SMTSolver()
            _declare_vars(s2, cond_term)
            _declare_vars(s2, pred_term)
            s2.add(cond_term)
            s2.add(pred_term)
            self.smt_queries += 1
            if s2.check() == "unsat":
                # Condition contradicts p_j: force next_j = false
                trans = bdd.AND(trans, bdd.NOT(self.mgr.next_bdd(j)))
                continue

            # Otherwise: frame (preserve current value)
            trans = bdd.AND(trans, frame)

        result = TransitionBDD(source_id, target_id, trans)
        self._cache[cache_key] = result
        return result

    def build_skip_transition(self, source_id: int, target_id: int) -> TransitionBDD:
        """Identity transition: all predicates preserved."""
        return self._build_identity_transition(source_id, target_id)

    def _build_identity_transition(self, source_id: int, target_id: int) -> TransitionBDD:
        """All predicates carry over unchanged."""
        cache_key = (source_id, target_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bdd = self.mgr.bdd
        trans = bdd.TRUE
        for j in range(self.mgr.num_predicates):
            frame = bdd.IFF(self.mgr.curr_bdd(j), self.mgr.next_bdd(j))
            trans = bdd.AND(trans, frame)

        result = TransitionBDD(source_id, target_id, trans)
        self._cache[cache_key] = result
        return result


# ---------- BDD-based CEGAR ----------

class BDDCEGAR:
    """CEGAR loop using BDD-based predicate abstraction."""

    def __init__(self, source: str, max_iterations: int = 20,
                 max_nodes: int = 500, seed_predicates: bool = True):
        self.source = source
        self.max_iterations = max_iterations
        self.max_nodes = max_nodes
        self.seed_predicates = seed_predicates

        self.cfg = build_cfg(source)
        self.mgr = BDDPredicateManager()
        self.builder: Optional[TransitionBDDBuilder] = None
        self._edge_trans: Dict[Tuple[int, int], TransitionBDD] = {}
        self._art_node_counter = 0
        self.bdd_image_ops = 0
        self.smt_queries_for_cex = 0

    def verify(self) -> BDDCEGARResult:
        """Run BDD-based CEGAR verification."""
        start = time.time()

        # Phase 1: Seed predicates from CFG assertions and conditions
        self._seed_predicates_from_cfg()

        # Phase 2: Optionally discover predicates via V114
        if self.seed_predicates and HAS_V114:
            self._seed_from_v114()

        if self.mgr.num_predicates == 0:
            # No predicates: trivially safe (no assertions to check)
            elapsed = (time.time() - start) * 1000
            return BDDCEGARResult(
                verdict=BDDVerdict.SAFE, safe=True,
                total_predicates=0, total_time_ms=elapsed
            )

        # Phase 3: CEGAR loop
        total_art_nodes = 0
        trans_built = 0

        for iteration in range(self.max_iterations):
            # Rebuild transition BDDs with current predicates
            self.builder = TransitionBDDBuilder(self.mgr)
            self._edge_trans.clear()
            self._build_all_transitions()
            trans_built += len(self._edge_trans)

            # Explore ART
            self._art_node_counter = 0
            root, all_nodes, error_nodes, covered = self._explore_art()
            total_art_nodes += len(all_nodes)

            if not error_nodes:
                # No error reachable: SAFE
                elapsed = (time.time() - start) * 1000
                return BDDCEGARResult(
                    verdict=BDDVerdict.SAFE, safe=True,
                    iterations=iteration + 1,
                    total_predicates=self.mgr.num_predicates,
                    predicate_names=[d for _, d in self.mgr.predicates],
                    art_nodes=total_art_nodes,
                    transition_bdds_built=trans_built,
                    bdd_image_ops=self.bdd_image_ops,
                    smt_queries_saved=self._estimate_smt_saved(total_art_nodes),
                    total_time_ms=(time.time() - start) * 1000
                )

            # Check counterexample feasibility
            for err_node in error_nodes:
                path = self._extract_path(err_node)
                feasible, model = self._check_feasibility(path)

                if feasible:
                    elapsed = (time.time() - start) * 1000
                    return BDDCEGARResult(
                        verdict=BDDVerdict.UNSAFE, safe=False,
                        counterexample=[(n.cfg_node_id, n.cfg_node_type.name) for n in path],
                        counterexample_inputs=model,
                        iterations=iteration + 1,
                        total_predicates=self.mgr.num_predicates,
                        predicate_names=[d for _, d in self.mgr.predicates],
                        art_nodes=total_art_nodes,
                        transition_bdds_built=trans_built,
                        bdd_image_ops=self.bdd_image_ops,
                        total_time_ms=elapsed
                    )

                # Spurious: refine with new predicates
                new_preds = self._refine(path)
                if new_preds == 0:
                    # Try V114 fallback
                    if HAS_V114:
                        new_preds = self._seed_from_v114()

                if new_preds > 0:
                    break  # Restart with new predicates

        elapsed = (time.time() - start) * 1000
        return BDDCEGARResult(
            verdict=BDDVerdict.UNKNOWN, safe=False,
            iterations=self.max_iterations,
            total_predicates=self.mgr.num_predicates,
            predicate_names=[d for _, d in self.mgr.predicates],
            art_nodes=total_art_nodes,
            transition_bdds_built=trans_built,
            bdd_image_ops=self.bdd_image_ops,
            total_time_ms=elapsed
        )

    def _seed_predicates_from_cfg(self):
        """Extract predicates from CFG assertions and conditions."""
        for node in self.cfg.nodes:
            if node.type in (CFGNodeType.ASSERT, CFGNodeType.ASSUME, CFGNodeType.ASSUME_NOT):
                cond_ast = node.data
                term = _safe_ast_to_smt(cond_ast)
                if term is not None:
                    desc = str(term)
                    self.mgr.add_predicate(term, desc)

    def _seed_from_v114(self) -> int:
        """Discover predicates via V114 and add them."""
        if not HAS_V114:
            return 0
        try:
            result = v114_discover(self.source, max_predicates=20)
            added = 0
            for pred in result.predicates:
                if hasattr(pred, 'term') and pred.term is not None:
                    before = self.mgr.num_predicates
                    self.mgr.add_predicate(pred.term, str(pred.term))
                    if self.mgr.num_predicates > before:
                        added += 1
            return added
        except Exception:
            return 0

    def _build_all_transitions(self):
        """Build transition BDDs for all CFG edges."""
        for node in self.cfg.nodes:
            for succ_id in node.successors:
                succ = self.cfg.nodes[succ_id]
                key = (node.id, succ_id)
                if key in self._edge_trans:
                    continue

                if node.type == CFGNodeType.ASSIGN:
                    var_name, expr_ast = node.data
                    expr_smt = _safe_ast_to_smt(expr_ast)
                    if expr_smt is not None:
                        t = self.builder.build_assign_transition(
                            var_name, expr_smt, node.id, succ_id
                        )
                    else:
                        t = self.builder.build_skip_transition(node.id, succ_id)
                elif node.type == CFGNodeType.ASSUME:
                    t = self.builder.build_assume_transition(
                        node.data, False, node.id, succ_id
                    )
                elif node.type == CFGNodeType.ASSUME_NOT:
                    t = self.builder.build_assume_transition(
                        node.data, True, node.id, succ_id
                    )
                else:
                    t = self.builder.build_skip_transition(node.id, succ_id)

                self._edge_trans[key] = t

    def _explore_art(self) -> Tuple[BDDARTNode, List[BDDARTNode],
                                     List[BDDARTNode], int]:
        """Build ART using BDD-based abstract post."""
        root = BDDARTNode(
            id=self._next_id(),
            cfg_node_id=self.cfg.entry.id,
            cfg_node_type=self.cfg.entry.type,
            state=self.mgr.state_top(),
            depth=0
        )

        all_nodes = [root]
        error_nodes = []
        covered_count = 0
        expanded: Dict[int, List[BDDARTNode]] = {}  # cfg_node_id -> nodes
        worklist = [root]

        while worklist and len(all_nodes) < self.max_nodes:
            current = worklist.pop()

            if current.is_covered:
                continue

            cfg_node = self.cfg.nodes[current.cfg_node_id]

            if cfg_node.type == CFGNodeType.ERROR:
                error_nodes.append(current)
                continue

            # Check coverage
            loc_nodes = expanded.get(current.cfg_node_id, [])
            is_covered = False
            for existing in loc_nodes:
                if existing.id != current.id and not existing.is_covered:
                    if existing.state.subsumes(current.state, self.mgr.bdd):
                        current.covered_by = existing
                        covered_count += 1
                        is_covered = True
                        break
            if is_covered:
                continue

            expanded.setdefault(current.cfg_node_id, []).append(current)

            # Expand successors
            for succ_id in cfg_node.successors:
                succ_cfg = self.cfg.nodes[succ_id]
                edge_key = (current.cfg_node_id, succ_id)
                trans_bdd = self._edge_trans.get(edge_key)

                if trans_bdd is None:
                    # No transition BDD: use identity (preserve state)
                    new_state = current.state
                else:
                    new_state = self.mgr.image(current.state, trans_bdd.bdd_node)
                    self.bdd_image_ops += 1

                if new_state.is_bottom:
                    continue

                child = BDDARTNode(
                    id=self._next_id(),
                    cfg_node_id=succ_id,
                    cfg_node_type=succ_cfg.type,
                    state=new_state,
                    parent=current,
                    depth=current.depth + 1
                )
                current.children.append(child)
                all_nodes.append(child)
                worklist.append(child)

        return root, all_nodes, error_nodes, covered_count

    def _extract_path(self, node: BDDARTNode) -> List[BDDARTNode]:
        """Extract path from root to node."""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def _check_feasibility(self, path: List[BDDARTNode]) -> Tuple[bool, Optional[Dict]]:
        """Check if a counterexample path is feasible using SMT."""
        solver = SMTSolver()
        var_versions: Dict[str, int] = {}
        step_formulas = []

        for i, node in enumerate(path):
            cfg_node = self.cfg.nodes[node.cfg_node_id]

            if cfg_node.type == CFGNodeType.ASSIGN:
                var_name, expr_ast = cfg_node.data
                # Get current version of expr vars
                expr_smt = _safe_ast_to_smt(expr_ast)
                if expr_smt is not None:
                    versioned_expr = _version_term(expr_smt, var_versions, solver)
                    # Increment version for assigned var
                    ver = var_versions.get(var_name, 0) + 1
                    var_versions[var_name] = ver
                    v = solver.Int(f"{var_name}_{ver}")
                    eq = App(Op.EQ, [v, versioned_expr], BOOL)
                    solver.add(eq)
                    step_formulas.append(eq)

            elif cfg_node.type in (CFGNodeType.ASSUME, CFGNodeType.ASSUME_NOT):
                cond = _safe_ast_to_smt(cfg_node.data)
                if cond is not None:
                    versioned_cond = _version_term(cond, var_versions, solver)
                    if cfg_node.type == CFGNodeType.ASSUME_NOT:
                        versioned_cond = _smt_not(versioned_cond)
                    solver.add(versioned_cond)
                    step_formulas.append(versioned_cond)

            elif cfg_node.type == CFGNodeType.ASSERT:
                cond = _safe_ast_to_smt(cfg_node.data)
                if cond is not None:
                    versioned_cond = _version_term(cond, var_versions, solver)
                    neg_cond = _smt_not(versioned_cond)
                    solver.add(neg_cond)
                    step_formulas.append(neg_cond)

        self.smt_queries_for_cex += 1
        result = solver.check()
        if result == "sat":
            model = solver.model()
            inputs = {}
            for name, val in model.items():
                if '_' in name:
                    base = name.rsplit('_', 1)[0]
                    ver_str = name.rsplit('_', 1)[1]
                    if ver_str == '0':
                        inputs[base] = val
                else:
                    inputs[name] = val
            return True, inputs
        return False, None

    def _refine(self, path: List[BDDARTNode]) -> int:
        """Refine predicates from spurious counterexample.

        Uses binary interpolation: split path at each point, extract
        predicates from the unsatisfiable conjunction.
        """
        added = 0

        # Extract conditions along the path for predicate candidates
        for node in path:
            cfg_node = self.cfg.nodes[node.cfg_node_id]

            if cfg_node.type in (CFGNodeType.ASSUME, CFGNodeType.ASSUME_NOT, CFGNodeType.ASSERT):
                cond = _safe_ast_to_smt(cfg_node.data)
                if cond is not None:
                    before = self.mgr.num_predicates
                    self.mgr.add_predicate(cond, str(cond))
                    if self.mgr.num_predicates > before:
                        added += 1

                    # Also try negation
                    neg = _smt_not(cond)
                    before = self.mgr.num_predicates
                    self.mgr.add_predicate(neg, str(neg))
                    if self.mgr.num_predicates > before:
                        added += 1

            elif cfg_node.type == CFGNodeType.ASSIGN:
                var_name, expr_ast = cfg_node.data
                expr_smt = _safe_ast_to_smt(expr_ast)
                if expr_smt is not None:
                    # Generate predicates: var >= 0, var >= expr, etc.
                    v = Var(var_name, INT)
                    zero = IntConst(0)
                    geq_zero = App(Op.GE, [v, zero], BOOL)
                    before = self.mgr.num_predicates
                    self.mgr.add_predicate(geq_zero, f"{var_name} >= 0")
                    if self.mgr.num_predicates > before:
                        added += 1

        return added

    def _estimate_smt_saved(self, total_art_nodes: int) -> int:
        """Estimate SMT queries saved by BDD-based approach.

        Without BDDs: each ART node expansion requires n SMT queries per predicate
        per successor. With BDDs: transition BDDs are built once, then image ops
        are pure BDD operations.
        """
        n = self.mgr.num_predicates
        # SMT-based would need ~n queries per expansion
        smt_would_need = total_art_nodes * n
        # BDD approach used smt_queries for building transitions + cex checks
        smt_used = (self.builder.smt_queries if self.builder else 0) + self.smt_queries_for_cex
        return max(0, smt_would_need - smt_used)

    def _next_id(self) -> int:
        self._art_node_counter += 1
        return self._art_node_counter


# ---------- SMT Helpers ----------

def _safe_ast_to_smt(ast_node) -> Optional:
    """Convert AST node to SMT term, returning None on failure."""
    try:
        return _ast_to_smt(ast_node)
    except Exception:
        return None


def _smt_not(term) -> object:
    """Negate an SMT term using complement operators."""
    if isinstance(term, App):
        complements = {
            Op.EQ: Op.NEQ, Op.NEQ: Op.EQ,
            Op.LT: Op.GE, Op.GE: Op.LT,
            Op.LE: Op.GT, Op.GT: Op.LE,
        }
        if term.op in complements:
            return App(complements[term.op], term.args, BOOL)
        if term.op == Op.AND:
            return App(Op.OR, [_smt_not(a) for a in term.args], BOOL)
        if term.op == Op.OR:
            return App(Op.AND, [_smt_not(a) for a in term.args], BOOL)
        if term.op == Op.NOT:
            return term.args[0]
    return App(Op.NOT, [term], BOOL)


def _declare_vars(solver: SMTSolver, term) -> None:
    """Declare all integer variables in an SMT term."""
    if isinstance(term, Var):
        if term.sort == INT or (hasattr(term, 'sort') and term.sort and term.sort.kind == SortKind.INT):
            solver.Int(term.name)
        else:
            solver.Bool(term.name)
    elif isinstance(term, App):
        for arg in term.args:
            _declare_vars(solver, arg)


def _substitute_smt(term, var_name: str, replacement) -> object:
    """Substitute var_name with replacement in SMT term."""
    if isinstance(term, Var):
        if term.name == var_name:
            return replacement
        return term
    if isinstance(term, (IntConst, BoolConst)):
        return term
    if isinstance(term, App):
        new_args = [_substitute_smt(a, var_name, replacement) for a in term.args]
        return App(term.op, new_args, term.sort)
    return term


def _version_term(term, var_versions: Dict[str, int], solver: SMTSolver):
    """Replace variables with versioned names for path encoding."""
    if isinstance(term, Var):
        ver = var_versions.get(term.name, 0)
        vname = f"{term.name}_{ver}"
        if term.sort == INT or (hasattr(term, 'sort') and term.sort and term.sort.kind == SortKind.INT):
            return solver.Int(vname)
        else:
            return solver.Bool(vname)
    if isinstance(term, (IntConst, BoolConst)):
        return term
    if isinstance(term, App):
        new_args = [_version_term(a, var_versions, solver) for a in term.args]
        return App(term.op, new_args, term.sort)
    return term


def _collect_vars(term) -> Set[str]:
    """Collect variable names from SMT term."""
    if isinstance(term, Var):
        return {term.name}
    if isinstance(term, App):
        result = set()
        for a in term.args:
            result |= _collect_vars(a)
        return result
    return set()


# ---------- High-level API ----------

def bdd_verify(source: str, max_iterations: int = 20,
               max_nodes: int = 500, seed_predicates: bool = True) -> BDDCEGARResult:
    """Verify a program using BDD-based predicate abstraction CEGAR.

    Args:
        source: C10 source code with assert() statements
        max_iterations: CEGAR iteration bound
        max_nodes: ART node exploration limit
        seed_predicates: Whether to use V114 predicate discovery
    """
    cegar = BDDCEGAR(source, max_iterations, max_nodes, seed_predicates)
    return cegar.verify()


def check_assertion(source: str) -> Tuple[bool, Optional[Dict]]:
    """Quick check: are all assertions safe?

    Returns (safe, counterexample_inputs).
    """
    result = bdd_verify(source)
    inputs = result.counterexample_inputs if not result.safe else None
    return result.safe, inputs


def bdd_vs_smt_comparison(source: str, max_iterations: int = 20) -> ComparisonResult:
    """Compare BDD-based vs SMT-based predicate abstraction.

    Runs both approaches on the same program and reports metrics.
    """
    # BDD-based
    t0 = time.time()
    bdd_result = bdd_verify(source, max_iterations)
    bdd_time = (time.time() - t0) * 1000

    # SMT-based (via V115 if available, else via V110)
    smt_time = 0.0
    smt_result_dict = None
    smt_queries = 0
    if HAS_V115:
        try:
            t0 = time.time()
            smt_res = v115_guided_verify(source, max_iterations=max_iterations)
            smt_time = (time.time() - t0) * 1000
            smt_result_dict = {
                'verdict': smt_res.verdict.value if hasattr(smt_res.verdict, 'value') else str(smt_res.verdict),
                'safe': smt_res.safe,
                'iterations': smt_res.iterations,
                'predicates': smt_res.total_predicates,
                'art_nodes': smt_res.art_nodes,
            }
            smt_queries = smt_res.art_nodes * smt_res.total_predicates
        except Exception:
            pass

    agree = True
    if smt_result_dict:
        agree = bdd_result.safe == smt_result_dict['safe']

    lines = []
    lines.append(f"BDD-CEGAR: {bdd_result.verdict.value} ({bdd_result.total_time_ms:.1f}ms)")
    lines.append(f"  Predicates: {bdd_result.total_predicates}, Iterations: {bdd_result.iterations}")
    lines.append(f"  ART nodes: {bdd_result.art_nodes}, Image ops: {bdd_result.bdd_image_ops}")
    lines.append(f"  SMT queries saved: ~{bdd_result.smt_queries_saved}")
    if smt_result_dict:
        lines.append(f"SMT-CEGAR: {smt_result_dict['verdict']} ({smt_time:.1f}ms)")
        lines.append(f"  Predicates: {smt_result_dict['predicates']}, Iterations: {smt_result_dict['iterations']}")
        lines.append(f"  ART nodes: {smt_result_dict['art_nodes']}")
    lines.append(f"Agreement: {agree}")

    return ComparisonResult(
        bdd_result=bdd_result,
        smt_result=smt_result_dict,
        bdd_time_ms=bdd_time,
        smt_time_ms=smt_time,
        both_agree=agree,
        bdd_image_ops=bdd_result.bdd_image_ops,
        smt_queries=smt_queries,
        summary="\n".join(lines)
    )


def get_transition_bdds(source: str) -> Dict:
    """Inspect the transition BDDs built for a program.

    Returns predicate info and per-edge BDD statistics.
    """
    cegar = BDDCEGAR(source)
    cegar._seed_predicates_from_cfg()

    if cegar.mgr.num_predicates == 0:
        return {'predicates': [], 'edges': [], 'total_bdd_nodes': 0}

    cegar.builder = TransitionBDDBuilder(cegar.mgr)
    cegar._build_all_transitions()

    preds = [{'index': i, 'description': d, 'term': str(t)}
             for i, (t, d) in enumerate(cegar.mgr.predicates)]

    edges = []
    total_nodes = 0
    for (src, tgt), trans in cegar._edge_trans.items():
        nc = cegar.mgr.bdd.node_count(trans.bdd_node)
        total_nodes += nc
        src_node = cegar.cfg.nodes[src]
        edges.append({
            'source': src, 'target': tgt,
            'source_type': src_node.type.name,
            'bdd_nodes': nc
        })

    return {
        'predicates': preds,
        'edges': edges,
        'total_bdd_nodes': total_nodes,
        'smt_queries_for_construction': cegar.builder.smt_queries
    }


def bdd_summary(source: str) -> str:
    """Human-readable verification summary."""
    result = bdd_verify(source)

    lines = [
        f"BDD Predicate Abstraction CEGAR",
        f"================================",
        f"Verdict: {result.verdict.value}",
        f"Safe: {result.safe}",
        f"Iterations: {result.iterations}",
        f"Predicates: {result.total_predicates}",
        f"ART nodes explored: {result.art_nodes}",
        f"Transition BDDs built: {result.transition_bdds_built}",
        f"BDD image operations: {result.bdd_image_ops}",
        f"SMT queries saved: ~{result.smt_queries_saved}",
        f"Time: {result.total_time_ms:.1f}ms",
    ]

    if result.predicate_names:
        lines.append(f"\nPredicates:")
        for i, name in enumerate(result.predicate_names):
            lines.append(f"  [{i}] {name}")

    if result.counterexample:
        lines.append(f"\nCounterexample path:")
        for node_id, node_type in result.counterexample:
            lines.append(f"  -> {node_type} (node {node_id})")

    if result.counterexample_inputs:
        lines.append(f"\nCounterexample inputs:")
        for k, v in result.counterexample_inputs.items():
            lines.append(f"  {k} = {v}")

    return "\n".join(lines)
