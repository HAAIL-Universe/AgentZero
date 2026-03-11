"""V119: BDD-Based Predicate Abstraction

Composes V021 (BDD Model Checking) + V110 (ART/CEGAR) + C037 (SMT) + C010 (parser)

Instead of per-predicate SMT queries for abstract post computation,
encodes the abstract transition relation as a BDD. Once built, abstract
post becomes a single BDD image operation -- amortizing the SMT cost
across all future abstract post computations at that CFG edge.

Key idea (Cartesian abstraction):
  For each CFG edge (assignment/assume) and each predicate p_j':
    - Check which current predicates p_i imply p_j holds after the edge
    - Encode as BDD clause: (curr_i => next_j) for each such implication
  The full transition BDD is the conjunction of all such clauses.

Abstract post: image(current_state_bdd, edge_trans_bdd)
  = exists curr. (current_state_bdd AND edge_trans_bdd)[next->curr]
"""

import sys
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Set

# Add paths for imports
_here = os.path.dirname(__file__)
_work = os.path.join(_here, '..')
_challenges = os.path.join(_here, '..', '..', '..', 'challenges')
for p in [
    os.path.join(_work, 'V021_bdd_model_checking'),
    os.path.join(_work, 'V110_abstract_reachability_tree'),
    os.path.join(_work, 'V115_predicate_guided_cegar'),
    os.path.join(_work, 'V114_recursive_predicate_discovery'),
    os.path.join(_challenges, 'C037_smt_solver'),
    os.path.join(_challenges, 'C010_stack_vm'),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bdd_model_checker import BDD, BDDNode
from art import (
    build_cfg, CFG, CFGNode, CFGNodeType,
    _ast_to_smt,
)
from smt_solver import SMTSolver, Var, IntConst, BoolConst, App, Op, Sort, SortKind

# Try V114 for predicate discovery
try:
    from recursive_predicate_discovery import (
        discover_predicates as v114_discover,
    )
    HAS_V114 = True
except ImportError:
    HAS_V114 = False

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


def _is_unsat(result) -> bool:
    """Check if SMT result is UNSAT (handles both enum and string)."""
    if hasattr(result, 'value'):
        return result.value == 'unsat'
    return str(result) == 'unsat'


def _is_sat(result) -> bool:
    """Check if SMT result is SAT (handles both enum and string)."""
    if hasattr(result, 'value'):
        return result.value == 'sat'
    return str(result) == 'sat'


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
        """self >= other iff other's concrete states are subset of self's.
        Equivalently: other AND NOT(self) is FALSE."""
        if self.is_bottom:
            return other.is_bottom
        if other.is_bottom:
            return True
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
    cfg_node: CFGNode
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
        return self.cfg_node.type == CFGNodeType.ERROR


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
      - curr_var[i] at BDD index 2*i: represents p_i in current state
      - next_var[i] at BDD index 2*i+1: represents p_i in next state (primed)
    """

    def __init__(self):
        self.bdd = BDD(num_vars=0)
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
        return self._curr_bdds[pred_idx]

    def next_bdd(self, pred_idx: int) -> BDDNode:
        return self._next_bdds[pred_idx]

    def state_top(self) -> BDDPredicateState:
        """Top: any predicate valuation possible."""
        return BDDPredicateState(self.bdd.TRUE)

    def state_bottom(self) -> BDDPredicateState:
        return BDDPredicateState(self.bdd.FALSE, is_bottom=True)

    def state_from_predicates(self, pred_indices: Set[int]) -> BDDPredicateState:
        """State where exactly the given predicates are known true."""
        if not pred_indices:
            return self.state_top()
        node = self.bdd.TRUE
        for idx in pred_indices:
            node = self.bdd.AND(node, self.curr_bdd(idx))
        return BDDPredicateState(node)

    def image(self, state: BDDPredicateState, trans: BDDNode) -> BDDPredicateState:
        """Compute abstract post via BDD image.

        image(S, T) = exists curr_vars. (S(curr) AND T(curr, next))[next -> curr]
        """
        if state.is_bottom:
            return self.state_bottom()

        combined = self.bdd.AND(state.bdd_node, trans)
        if combined == self.bdd.FALSE:
            return self.state_bottom()

        # Existentially quantify current-state variables
        result = self.bdd.exists_multi(self.curr_vars, combined)
        if result == self.bdd.FALSE:
            return self.state_bottom()

        # Rename next vars -> current vars
        rename_map = {}
        for i in range(self.num_predicates):
            rename_map[self.next_vars[i]] = self.curr_vars[i]
        result = self.bdd.rename(result, rename_map)

        return BDDPredicateState(result)


# ---------- SMT Helpers ----------

def _safe_ast_to_smt(ast_node, env_vars=None):
    """Convert AST node to SMT term, returning None on failure."""
    if env_vars is None:
        env_vars = {}
    try:
        return _ast_to_smt(ast_node, env_vars)
    except Exception:
        return None


def _smt_not(term):
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


def _declare_vars(solver: SMTSolver, term):
    """Declare all integer variables found in an SMT term."""
    if isinstance(term, Var):
        if term.sort and term.sort.kind == SortKind.INT:
            solver.Int(term.name)
        else:
            solver.Bool(term.name)
    elif isinstance(term, App):
        for arg in term.args:
            _declare_vars(solver, arg)


def _substitute_smt(term, var_name: str, replacement):
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
        if term.sort and term.sort.kind == SortKind.INT:
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


# ---------- Transition BDD Builder ----------

class TransitionBDDBuilder:
    """Builds abstract transition BDDs for CFG edges using SMT queries.

    Uses Cartesian abstraction: for each predicate p_j in next state,
    check independently which current predicates imply/contradict p_j
    after the transition.
    """

    def __init__(self, mgr: BDDPredicateManager):
        self.mgr = mgr
        self.smt_queries = 0
        self._cache: Dict[Tuple[int, int], BDDNode] = {}

    def build_assign_transition(self, var_name: str, expr_smt,
                                src_id: int, tgt_id: int) -> BDDNode:
        """Build transition BDD for assignment: var := expr."""
        cache_key = (src_id, tgt_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bdd = self.mgr.bdd
        n = self.mgr.num_predicates
        trans = bdd.TRUE

        for j in range(n):
            pred_term, _ = self.mgr.predicates[j]
            # Compute p_j[var := expr] (the predicate after substitution)
            subst_term = _substitute_smt(pred_term, var_name, expr_smt)

            # Check if subst_term is a tautology (always true after assignment)
            solver = SMTSolver()
            _declare_vars(solver, pred_term)
            _declare_vars(solver, subst_term)
            solver.add(_smt_not(subst_term))
            self.smt_queries += 1
            if _is_unsat(solver.check()):
                # p_j' always holds after this assignment
                trans = bdd.AND(trans, self.mgr.next_bdd(j))
                continue

            # Check if subst_term is unsatisfiable (always false)
            solver2 = SMTSolver()
            _declare_vars(solver2, pred_term)
            _declare_vars(solver2, subst_term)
            solver2.add(subst_term)
            self.smt_queries += 1
            if _is_unsat(solver2.check()):
                trans = bdd.AND(trans, bdd.NOT(self.mgr.next_bdd(j)))
                continue

            # Per-predicate implications: curr_i => next_j
            has_implication = False
            for i in range(n):
                src_term, _ = self.mgr.predicates[i]
                # Check: p_i => subst_term?
                s = SMTSolver()
                _declare_vars(s, src_term)
                _declare_vars(s, subst_term)
                s.add(src_term)
                s.add(_smt_not(subst_term))
                self.smt_queries += 1
                if _is_unsat(s.check()):
                    # curr_i => next_j
                    clause = bdd.OR(bdd.NOT(self.mgr.curr_bdd(i)), self.mgr.next_bdd(j))
                    trans = bdd.AND(trans, clause)
                    has_implication = True

            # Check negation implications: NOT(p_i) => subst_term?
            for i in range(n):
                src_term, _ = self.mgr.predicates[i]
                s = SMTSolver()
                _declare_vars(s, src_term)
                _declare_vars(s, subst_term)
                s.add(_smt_not(src_term))
                s.add(_smt_not(subst_term))
                self.smt_queries += 1
                if _is_unsat(s.check()):
                    # NOT(curr_i) => next_j
                    clause = bdd.OR(self.mgr.curr_bdd(i), self.mgr.next_bdd(j))
                    trans = bdd.AND(trans, clause)
                    has_implication = True

        self._cache[cache_key] = trans
        return trans

    def build_assume_transition(self, condition_smt, is_negated: bool,
                                src_id: int, tgt_id: int) -> BDDNode:
        """Build transition BDD for assume(cond) or assume(!cond).

        The condition acts as a guard: the transition is only feasible
        from states compatible with the condition. Predicates implied by
        the condition become true; those contradicted become false (and
        must be false in current state for feasibility).
        """
        cache_key = (src_id, tgt_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bdd = self.mgr.bdd

        cond = condition_smt
        if cond is None:
            t = self.build_identity_transition(src_id, tgt_id)
            self._cache[cache_key] = t
            return t

        if is_negated:
            cond = _smt_not(cond)

        trans = bdd.TRUE
        for j in range(self.mgr.num_predicates):
            pred_term, _ = self.mgr.predicates[j]

            # Check if condition implies predicate
            s = SMTSolver()
            _declare_vars(s, cond)
            _declare_vars(s, pred_term)
            s.add(cond)
            s.add(_smt_not(pred_term))
            self.smt_queries += 1
            if _is_unsat(s.check()):
                # cond => p_j: force next_j = true
                trans = bdd.AND(trans, self.mgr.next_bdd(j))
                continue

            # Check if condition contradicts predicate
            s2 = SMTSolver()
            _declare_vars(s2, cond)
            _declare_vars(s2, pred_term)
            s2.add(cond)
            s2.add(pred_term)
            self.smt_queries += 1
            if _is_unsat(s2.check()):
                # cond AND p_j is UNSAT: p_j must be false for cond to hold
                # Guard: curr_j must be false; next_j is false
                trans = bdd.AND(trans, bdd.NOT(self.mgr.curr_bdd(j)))
                trans = bdd.AND(trans, bdd.NOT(self.mgr.next_bdd(j)))
                continue

            # Frame: preserve current value
            trans = bdd.AND(trans, bdd.IFF(self.mgr.curr_bdd(j), self.mgr.next_bdd(j)))

        self._cache[cache_key] = trans
        return trans

    def build_identity_transition(self, src_id: int, tgt_id: int) -> BDDNode:
        """All predicates carry over unchanged."""
        cache_key = (src_id, tgt_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        bdd = self.mgr.bdd
        trans = bdd.TRUE
        for j in range(self.mgr.num_predicates):
            trans = bdd.AND(trans, bdd.IFF(self.mgr.curr_bdd(j), self.mgr.next_bdd(j)))

        self._cache[cache_key] = trans
        return trans


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
        self._edge_trans: Dict[Tuple[int, int], BDDNode] = {}
        self._art_counter = 0
        self.bdd_image_ops = 0
        self.smt_cex_queries = 0

    def verify(self) -> BDDCEGARResult:
        """Run BDD-based CEGAR verification."""
        start = time.time()

        # Phase 1: Seed predicates from CFG
        self._seed_from_cfg()

        # Phase 2: Optional V114 discovery
        if self.seed_predicates and HAS_V114:
            self._seed_from_v114()

        if self.mgr.num_predicates == 0:
            elapsed = (time.time() - start) * 1000
            return BDDCEGARResult(
                verdict=BDDVerdict.SAFE, safe=True,
                total_predicates=0, total_time_ms=elapsed
            )

        total_art_nodes = 0
        trans_built = 0

        for iteration in range(self.max_iterations):
            # Build transition BDDs
            self.builder = TransitionBDDBuilder(self.mgr)
            self._edge_trans.clear()
            self._build_transitions()
            trans_built += len(self._edge_trans)

            # Explore ART
            self._art_counter = 0
            root, all_nodes, error_nodes, covered = self._explore_art()
            total_art_nodes += len(all_nodes)

            if not error_nodes:
                elapsed = (time.time() - start) * 1000
                return BDDCEGARResult(
                    verdict=BDDVerdict.SAFE, safe=True,
                    iterations=iteration + 1,
                    total_predicates=self.mgr.num_predicates,
                    predicate_names=[d for _, d in self.mgr.predicates],
                    art_nodes=total_art_nodes,
                    transition_bdds_built=trans_built,
                    bdd_image_ops=self.bdd_image_ops,
                    smt_queries_saved=self._estimate_saved(total_art_nodes),
                    total_time_ms=(time.time() - start) * 1000
                )

            # Check counterexample feasibility
            refined = False
            for err_node in error_nodes:
                path = self._extract_path(err_node)
                feasible, model = self._check_feasibility(path)

                if feasible:
                    elapsed = (time.time() - start) * 1000
                    return BDDCEGARResult(
                        verdict=BDDVerdict.UNSAFE, safe=False,
                        counterexample=[(n.cfg_node.id, n.cfg_node.type.name) for n in path],
                        counterexample_inputs=model,
                        iterations=iteration + 1,
                        total_predicates=self.mgr.num_predicates,
                        predicate_names=[d for _, d in self.mgr.predicates],
                        art_nodes=total_art_nodes,
                        transition_bdds_built=trans_built,
                        bdd_image_ops=self.bdd_image_ops,
                        total_time_ms=elapsed
                    )

                # Spurious: refine
                new_preds = self._refine(path)
                if new_preds == 0 and HAS_V114:
                    new_preds = self._seed_from_v114()

                if new_preds > 0:
                    refined = True
                    break

            if not refined:
                # Could not refine further
                break

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

    def _seed_from_cfg(self):
        """Extract predicates from CFG assert/assume nodes."""
        env = {}
        for node in self.cfg.nodes:
            if node.type in (CFGNodeType.ASSERT, CFGNodeType.ASSUME, CFGNodeType.ASSUME_NOT):
                if node.data is not None:
                    term = _safe_ast_to_smt(node.data, env)
                    if term is not None:
                        self.mgr.add_predicate(term, str(term))

    def _seed_from_v114(self) -> int:
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

    def _build_transitions(self):
        """Build transition BDDs for all CFG edges."""
        env = {}
        for node in self.cfg.nodes:
            for succ in node.successors:
                key = (node.id, succ.id)
                if key in self._edge_trans:
                    continue

                if node.type == CFGNodeType.ASSIGN and node.data:
                    var_name, expr_ast = node.data
                    expr_smt = _safe_ast_to_smt(expr_ast, dict(env))
                    if expr_smt is not None:
                        t = self.builder.build_assign_transition(
                            var_name, expr_smt, node.id, succ.id)
                    else:
                        t = self.builder.build_identity_transition(node.id, succ.id)
                elif node.type == CFGNodeType.ASSUME and node.data:
                    cond_smt = _safe_ast_to_smt(node.data, dict(env))
                    t = self.builder.build_assume_transition(
                        cond_smt, False, node.id, succ.id)
                elif node.type == CFGNodeType.ASSUME_NOT and node.data:
                    cond_smt = _safe_ast_to_smt(node.data, dict(env))
                    t = self.builder.build_assume_transition(
                        cond_smt, True, node.id, succ.id)
                elif node.type == CFGNodeType.ASSERT and node.data:
                    # ASSERT -> ERROR: assertion fails (assume NOT cond)
                    # ASSERT -> other: assertion holds (assume cond)
                    cond_smt = _safe_ast_to_smt(node.data, dict(env))
                    is_error = succ.type == CFGNodeType.ERROR
                    t = self.builder.build_assume_transition(
                        cond_smt, is_error, node.id, succ.id)
                else:
                    t = self.builder.build_identity_transition(node.id, succ.id)

                self._edge_trans[key] = t

    def _explore_art(self):
        """Build ART using BDD-based abstract post."""
        root = BDDARTNode(
            id=self._next_id(),
            cfg_node=self.cfg.entry,
            state=self.mgr.state_top(),
            depth=0
        )

        all_nodes = [root]
        error_nodes = []
        covered_count = 0
        expanded: Dict[int, List[BDDARTNode]] = {}
        worklist = [root]

        while worklist and len(all_nodes) < self.max_nodes:
            current = worklist.pop()
            if current.is_covered:
                continue

            if current.cfg_node.type == CFGNodeType.ERROR:
                error_nodes.append(current)
                continue

            # Check coverage
            loc_id = current.cfg_node.id
            loc_nodes = expanded.get(loc_id, [])
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

            expanded.setdefault(loc_id, []).append(current)

            # Expand successors
            for succ_cfg in current.cfg_node.successors:
                edge_key = (current.cfg_node.id, succ_cfg.id)
                trans_bdd = self._edge_trans.get(edge_key)

                if trans_bdd is None:
                    new_state = current.state
                else:
                    new_state = self.mgr.image(current.state, trans_bdd)
                    self.bdd_image_ops += 1

                if new_state.is_bottom:
                    continue

                child = BDDARTNode(
                    id=self._next_id(),
                    cfg_node=succ_cfg,
                    state=new_state,
                    parent=current,
                    depth=current.depth + 1
                )
                current.children.append(child)
                all_nodes.append(child)
                worklist.append(child)

        return root, all_nodes, error_nodes, covered_count

    def _extract_path(self, node: BDDARTNode) -> List[BDDARTNode]:
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def _check_feasibility(self, path: List[BDDARTNode]) -> Tuple[bool, Optional[Dict]]:
        """Check if counterexample path is feasible using SMT."""
        solver = SMTSolver()
        var_versions: Dict[str, int] = {}
        env = {}

        for node in path:
            cfg_node = node.cfg_node

            if cfg_node.type == CFGNodeType.ASSIGN and cfg_node.data:
                var_name, expr_ast = cfg_node.data
                expr_smt = _safe_ast_to_smt(expr_ast, dict(env))
                if expr_smt is not None:
                    versioned_expr = _version_term(expr_smt, var_versions, solver)
                    ver = var_versions.get(var_name, 0) + 1
                    var_versions[var_name] = ver
                    v = solver.Int(f"{var_name}_{ver}")
                    solver.add(App(Op.EQ, [v, versioned_expr], BOOL))

            elif cfg_node.type in (CFGNodeType.ASSUME, CFGNodeType.ASSUME_NOT) and cfg_node.data:
                cond = _safe_ast_to_smt(cfg_node.data, dict(env))
                if cond is not None:
                    versioned_cond = _version_term(cond, var_versions, solver)
                    if cfg_node.type == CFGNodeType.ASSUME_NOT:
                        versioned_cond = _smt_not(versioned_cond)
                    solver.add(versioned_cond)

            elif cfg_node.type == CFGNodeType.ASSERT and cfg_node.data:
                cond = _safe_ast_to_smt(cfg_node.data, dict(env))
                if cond is not None:
                    versioned_cond = _version_term(cond, var_versions, solver)
                    solver.add(_smt_not(versioned_cond))

        self.smt_cex_queries += 1
        result = solver.check()
        if _is_sat(result):
            model = solver.model()
            inputs = {}
            for name, val in model.items():
                if '_' in name:
                    base, ver_str = name.rsplit('_', 1)
                    if ver_str == '0':
                        inputs[base] = val
                else:
                    inputs[name] = val
            return True, inputs
        return False, None

    def _refine(self, path: List[BDDARTNode]) -> int:
        """Refine: extract new predicates from spurious counterexample.

        Key refinement strategy: for each assignment var := expr on the path,
        compute backward substitutions of existing predicates. This produces
        the predicates needed to track values across assignments.
        """
        added = 0
        env = {}

        for node in path:
            cfg_node = node.cfg_node

            if cfg_node.type in (CFGNodeType.ASSUME, CFGNodeType.ASSUME_NOT, CFGNodeType.ASSERT):
                if cfg_node.data is not None:
                    cond = _safe_ast_to_smt(cfg_node.data, dict(env))
                    if cond is not None:
                        before = self.mgr.num_predicates
                        self.mgr.add_predicate(cond, str(cond))
                        if self.mgr.num_predicates > before:
                            added += 1

                        neg = _smt_not(cond)
                        before = self.mgr.num_predicates
                        self.mgr.add_predicate(neg, str(neg))
                        if self.mgr.num_predicates > before:
                            added += 1

            elif cfg_node.type == CFGNodeType.ASSIGN and cfg_node.data:
                var_name, expr_ast = cfg_node.data
                expr_smt = _safe_ast_to_smt(expr_ast, dict(env))

                # Standard: var >= 0
                v = Var(var_name, INT)
                zero = IntConst(0)
                geq_zero = App(Op.GE, [v, zero], BOOL)
                before = self.mgr.num_predicates
                self.mgr.add_predicate(geq_zero, f"{var_name} >= 0")
                if self.mgr.num_predicates > before:
                    added += 1

                # Backward substitution: for each existing predicate mentioning
                # var_name, compute p[var := expr] and add as new predicate
                if expr_smt is not None:
                    current_preds = list(self.mgr.predicates)
                    for pred_term, pred_desc in current_preds:
                        if var_name in _collect_vars(pred_term):
                            subst = _substitute_smt(pred_term, var_name, expr_smt)
                            before = self.mgr.num_predicates
                            self.mgr.add_predicate(subst, str(subst))
                            if self.mgr.num_predicates > before:
                                added += 1

        return added

    def _estimate_saved(self, total_art_nodes: int) -> int:
        n = self.mgr.num_predicates
        smt_would = total_art_nodes * n
        smt_used = (self.builder.smt_queries if self.builder else 0) + self.smt_cex_queries
        return max(0, smt_would - smt_used)

    def _next_id(self) -> int:
        self._art_counter += 1
        return self._art_counter


# ---------- High-level API ----------

def bdd_verify(source: str, max_iterations: int = 20,
               max_nodes: int = 500, seed_predicates: bool = True) -> BDDCEGARResult:
    """Verify a program using BDD-based predicate abstraction CEGAR."""
    cegar = BDDCEGAR(source, max_iterations, max_nodes, seed_predicates)
    return cegar.verify()


def check_assertion(source: str) -> Tuple[bool, Optional[Dict]]:
    """Quick check: are all assertions safe?"""
    result = bdd_verify(source)
    inputs = result.counterexample_inputs if not result.safe else None
    return result.safe, inputs


def bdd_vs_smt_comparison(source: str, max_iterations: int = 20) -> ComparisonResult:
    """Compare BDD-based vs SMT-based predicate abstraction."""
    t0 = time.time()
    bdd_result = bdd_verify(source, max_iterations)
    bdd_time = (time.time() - t0) * 1000

    smt_result_dict = None
    smt_time = 0.0
    smt_queries = 0

    # Try V110 direct for comparison
    try:
        from art import verify_program
        t0 = time.time()
        smt_res = verify_program(source, max_iterations=max_iterations)
        smt_time = (time.time() - t0) * 1000
        smt_result_dict = {
            'verdict': 'safe' if smt_res.safe else ('unsafe' if smt_res.counterexample else 'unknown'),
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

    lines = [f"BDD-CEGAR: {bdd_result.verdict.value} ({bdd_result.total_time_ms:.1f}ms)"]
    lines.append(f"  Predicates: {bdd_result.total_predicates}, Iterations: {bdd_result.iterations}")
    lines.append(f"  ART nodes: {bdd_result.art_nodes}, Image ops: {bdd_result.bdd_image_ops}")
    lines.append(f"  SMT queries saved: ~{bdd_result.smt_queries_saved}")
    if smt_result_dict:
        lines.append(f"SMT-CEGAR: {smt_result_dict['verdict']} ({smt_time:.1f}ms)")
        lines.append(f"  Predicates: {smt_result_dict['predicates']}, Iterations: {smt_result_dict['iterations']}")
    lines.append(f"Agreement: {agree}")

    return ComparisonResult(
        bdd_result=bdd_result, smt_result=smt_result_dict,
        bdd_time_ms=bdd_time, smt_time_ms=smt_time,
        both_agree=agree, bdd_image_ops=bdd_result.bdd_image_ops,
        smt_queries=smt_queries, summary="\n".join(lines)
    )


def get_transition_bdds(source: str) -> Dict:
    """Inspect transition BDDs built for a program."""
    cegar = BDDCEGAR(source)
    cegar._seed_from_cfg()

    if cegar.mgr.num_predicates == 0:
        return {'predicates': [], 'edges': [], 'total_bdd_nodes': 0}

    cegar.builder = TransitionBDDBuilder(cegar.mgr)
    cegar._build_transitions()

    preds = [{'index': i, 'description': d, 'term': str(t)}
             for i, (t, d) in enumerate(cegar.mgr.predicates)]

    edges = []
    total_nodes = 0
    for (src, tgt), trans_bdd in cegar._edge_trans.items():
        nc = cegar.mgr.bdd.node_count(trans_bdd)
        total_nodes += nc
        src_node = cegar.cfg.nodes[src]
        edges.append({
            'source': src, 'target': tgt,
            'source_type': src_node.type.name,
            'bdd_nodes': nc
        })

    return {
        'predicates': preds, 'edges': edges,
        'total_bdd_nodes': total_nodes,
        'smt_queries_for_construction': cegar.builder.smt_queries
    }


def bdd_summary(source: str) -> str:
    """Human-readable verification summary."""
    result = bdd_verify(source)
    lines = [
        "BDD Predicate Abstraction CEGAR",
        "================================",
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
        lines.append("\nPredicates:")
        for i, name in enumerate(result.predicate_names):
            lines.append(f"  [{i}] {name}")
    if result.counterexample:
        lines.append("\nCounterexample path:")
        for node_id, node_type in result.counterexample:
            lines.append(f"  -> {node_type} (node {node_id})")
    if result.counterexample_inputs:
        lines.append("\nCounterexample inputs:")
        for k, v in result.counterexample_inputs.items():
            lines.append(f"  {k} = {v}")
    return "\n".join(lines)
