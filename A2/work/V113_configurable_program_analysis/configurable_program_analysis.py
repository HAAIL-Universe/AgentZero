"""V113: Configurable Program Analysis (CPA)

CPAchecker-style framework: pluggable abstract domains into ART exploration.

Composes:
- V110 (Abstract Reachability Tree) -- CFG construction, CEGAR loop
- V020 (Abstract Domain Functor) -- pluggable numeric domains
- V104 (Relational Abstract Domains) -- zone/octagon domains
- V107 (Craig Interpolation) -- refinement
- C037 (SMT solver) -- feasibility checking
- C010 (parser) -- C10 source parsing

Key idea: Separate the CEGAR algorithm from the abstract domain.
V110 hardcodes predicate abstraction; CPA makes the domain configurable.
"""

import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Callable, Set

# -- imports from existing challenges --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V110_abstract_reachability_tree'))
from art import (
    CFG, CFGNode, CFGNodeType, build_cfg_from_source,
    _ast_to_smt, ARTResult
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V020_abstract_domain_functor'))
from domain_functor import (
    AbstractDomain as V020AbstractDomain, IntervalDomain, SignDomain, ConstDomain, ParityDomain,
    FunctorInterpreter, DomainEnv, make_sign_interval, make_full_product,
    ReducedProductDomain, ProductDomain, NEG_INF, INF
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V104_relational_abstract_domains'))
from relational_domains import ZoneDomain, OctagonDomain

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V107_craig_interpolation'))

# SMT solver
smt_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver')
sys.path.insert(0, smt_path)
from smt_solver import (
    SMTSolver, Var, IntConst, BoolConst, App, Op, Sort, SortKind, SMTResult
)

# C10 parser
c10_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm')
sys.path.insert(0, c10_path)
from stack_vm import lex, Parser


# ============================================================
# CPA Interface
# ============================================================

class AbstractState(ABC):
    """Abstract state in a CPA."""

    @abstractmethod
    def is_bottom(self) -> bool:
        """Return True if this state is infeasible (bottom)."""

    @abstractmethod
    def subsumes(self, other: 'AbstractState') -> bool:
        """Return True if self is at least as general as other (self >= other)."""

    @abstractmethod
    def join(self, other: 'AbstractState') -> 'AbstractState':
        """Least upper bound."""

    @abstractmethod
    def copy(self) -> 'AbstractState':
        """Deep copy."""

    @abstractmethod
    def __repr__(self) -> str:
        pass


class TransferRelation(ABC):
    """Computes successor abstract states."""

    @abstractmethod
    def get_abstract_successors(
        self, state: AbstractState, edge: 'CPAEdge', precision: Any
    ) -> List[AbstractState]:
        """Compute successor states for a CFG edge."""


class MergeOperator(ABC):
    """Controls when states at the same location are merged."""

    @abstractmethod
    def merge(self, state1: AbstractState, state2: AbstractState,
              precision: Any) -> AbstractState:
        """Merge state1 into state2. Return state2 if no merge, or merged state."""


class StopOperator(ABC):
    """Controls when exploration stops at a location."""

    @abstractmethod
    def stop(self, state: AbstractState, reached: List[AbstractState],
             precision: Any) -> bool:
        """Return True if state is covered by the reached set."""


class PrecisionAdjustment(ABC):
    """Adjusts precision during exploration."""

    @abstractmethod
    def adjust(self, state: AbstractState, precision: Any,
               reached: List[AbstractState]) -> Tuple[AbstractState, Any]:
        """Adjust state and precision. Return (adjusted_state, adjusted_precision)."""


class CPA(ABC):
    """Configurable Program Analysis -- bundles all components."""

    @abstractmethod
    def initial_state(self) -> AbstractState:
        """Return the initial abstract state."""

    @abstractmethod
    def initial_precision(self) -> Any:
        """Return the initial precision."""

    @abstractmethod
    def transfer(self) -> TransferRelation:
        """Return the transfer relation."""

    @abstractmethod
    def merge_op(self) -> MergeOperator:
        """Return the merge operator."""

    @abstractmethod
    def stop_op(self) -> StopOperator:
        """Return the stop operator."""

    @abstractmethod
    def precision_adjustment(self) -> PrecisionAdjustment:
        """Return the precision adjustment."""


# ============================================================
# CPA Edge (CFG transition)
# ============================================================

class EdgeType(Enum):
    ASSIGN = "assign"
    ASSUME = "assume"
    ASSUME_NOT = "assume_not"
    SKIP = "skip"
    ERROR = "error"
    ASSERT = "assert"


@dataclass
class CPAEdge:
    """A CFG edge between two locations."""
    source: CFGNode
    target: CFGNode
    edge_type: EdgeType
    data: Any = None  # (var_name, expr) for ASSIGN; expr for ASSUME/ASSERT


# ============================================================
# Merge and Stop Strategies
# ============================================================

class MergeSep(MergeOperator):
    """Never merge -- keep states separate (most precise)."""
    def merge(self, state1, state2, precision):
        return state2  # no merge


class MergeJoin(MergeOperator):
    """Always merge via join (faster, less precise)."""
    def merge(self, state1, state2, precision):
        return state1.join(state2)


class StopSep(StopOperator):
    """Stop if state is subsumed by ANY element in reached."""
    def stop(self, state, reached, precision):
        for r in reached:
            if r.subsumes(state):
                return True
        return False


class StopJoin(StopOperator):
    """Stop if state is subsumed by the join of all reached."""
    def stop(self, state, reached, precision):
        if not reached:
            return False
        joined = reached[0]
        for r in reached[1:]:
            joined = joined.join(r)
        return joined.subsumes(state)


class NoPrecisionAdjustment(PrecisionAdjustment):
    """No adjustment -- pass through."""
    def adjust(self, state, precision, reached):
        return state, precision


# ============================================================
# Interval CPA
# ============================================================

class IntervalState(AbstractState):
    """Abstract state using interval domain per variable."""

    def __init__(self, env: Optional[Dict[str, Tuple[float, float]]] = None,
                 bottom: bool = False):
        self.env = env or {}  # var_name -> (lo, hi)
        self._bottom = bottom

    def is_bottom(self) -> bool:
        return self._bottom

    def subsumes(self, other: 'IntervalState') -> bool:
        if self._bottom:
            return other._bottom
        if other._bottom:
            return True
        for var, (lo2, hi2) in other.env.items():
            lo1, hi1 = self.env.get(var, (NEG_INF, INF))
            if lo1 > lo2 or hi1 < hi2:
                return False
        return True

    def join(self, other: 'IntervalState') -> 'IntervalState':
        if self._bottom:
            return other.copy()
        if other._bottom:
            return self.copy()
        result = {}
        all_vars = set(self.env) | set(other.env)
        for v in all_vars:
            lo1, hi1 = self.env.get(v, (NEG_INF, INF))
            lo2, hi2 = other.env.get(v, (NEG_INF, INF))
            result[v] = (min(lo1, lo2), max(hi1, hi2))
        return IntervalState(result)

    def widen(self, other: 'IntervalState') -> 'IntervalState':
        if self._bottom:
            return other.copy()
        if other._bottom:
            return self.copy()
        result = {}
        all_vars = set(self.env) | set(other.env)
        for v in all_vars:
            lo1, hi1 = self.env.get(v, (NEG_INF, INF))
            lo2, hi2 = other.env.get(v, (NEG_INF, INF))
            new_lo = NEG_INF if lo2 < lo1 else lo1
            new_hi = INF if hi2 > hi1 else hi1
            result[v] = (new_lo, new_hi)
        return IntervalState(result)

    def copy(self) -> 'IntervalState':
        return IntervalState(dict(self.env), self._bottom)

    def get_interval(self, var: str) -> Tuple[float, float]:
        return self.env.get(var, (NEG_INF, INF))

    def __repr__(self) -> str:
        if self._bottom:
            return "IntervalState(BOT)"
        parts = [f"{v}:[{lo},{hi}]" for v, (lo, hi) in sorted(self.env.items())]
        return f"IntervalState({', '.join(parts)})"


class IntervalTransfer(TransferRelation):
    """Transfer relation for interval domain."""

    def get_abstract_successors(self, state, edge, precision):
        if state.is_bottom():
            return []

        if edge.edge_type == EdgeType.SKIP:
            return [state.copy()]

        if edge.edge_type == EdgeType.ERROR:
            return [state.copy()]

        if edge.edge_type == EdgeType.ASSIGN:
            var_name, expr = edge.data
            new_state = state.copy()
            val = self._eval_expr(expr, state)
            new_state.env[var_name] = val
            return [new_state]

        if edge.edge_type == EdgeType.ASSUME:
            refined = self._refine_assume(state, edge.data, positive=True)
            if refined.is_bottom():
                return []
            return [refined]

        if edge.edge_type == EdgeType.ASSUME_NOT:
            refined = self._refine_assume(state, edge.data, positive=False)
            if refined.is_bottom():
                return []
            return [refined]

        if edge.edge_type == EdgeType.ASSERT:
            return [state.copy()]

        return [state.copy()]

    def _eval_expr(self, expr, state: IntervalState) -> Tuple[float, float]:
        cls = expr.__class__.__name__
        if cls == 'IntLit':
            v = expr.value
            return (v, v)
        if cls == 'Var':
            return state.get_interval(expr.name)
        if cls == 'BoolLit':
            v = 1 if expr.value else 0
            return (v, v)
        if cls == 'BinOp':
            l = self._eval_expr(expr.left, state)
            r = self._eval_expr(expr.right, state)
            return self._bin_op(expr.op, l, r)
        if cls == 'UnaryOp':
            operand = self._eval_expr(expr.operand, state)
            if expr.op == '-':
                return (-operand[1], -operand[0])
            return (NEG_INF, INF)
        return (NEG_INF, INF)

    def _bin_op(self, op, l, r):
        lo1, hi1 = l
        lo2, hi2 = r
        if op == '+':
            return (lo1 + lo2, hi1 + hi2)
        if op == '-':
            return (lo1 - hi2, hi1 - lo2)
        if op == '*':
            products = [lo1*lo2, lo1*hi2, hi1*lo2, hi1*hi2]
            finite = [p for p in products if p != float('inf') and p != float('-inf')]
            if not finite:
                return (NEG_INF, INF)
            return (min(products), max(products))
        if op in ('<', '<=', '>', '>=', '==', '!='):
            return (0, 1)
        return (NEG_INF, INF)

    def _refine_assume(self, state: IntervalState, cond, positive: bool) -> IntervalState:
        new_state = state.copy()
        cls = cond.__class__.__name__
        if cls != 'BinOp':
            return new_state

        op = cond.op
        if not positive:
            op = {'<': '>=', '<=': '>', '>': '<=', '>=': '<',
                  '==': '!=', '!=': '=='}.get(op, op)

        left_cls = cond.left.__class__.__name__
        right_cls = cond.right.__class__.__name__

        # Refine when comparing var to literal/interval
        if left_cls == 'Var':
            var = cond.left.name
            rlo, rhi = self._eval_expr(cond.right, state)
            lo, hi = state.get_interval(var)
            if op == '<':
                hi = min(hi, rhi - 1)
            elif op == '<=':
                hi = min(hi, rhi)
            elif op == '>':
                lo = max(lo, rlo + 1)
            elif op == '>=':
                lo = max(lo, rlo)
            elif op == '==':
                lo = max(lo, rlo)
                hi = min(hi, rhi)
            if lo > hi:
                return IntervalState(bottom=True)
            new_state.env[var] = (lo, hi)

        if right_cls == 'Var':
            var = cond.right.name
            llo, lhi = self._eval_expr(cond.left, state)
            lo, hi = state.get_interval(var)
            if op == '<':
                lo = max(lo, llo + 1)
            elif op == '<=':
                lo = max(lo, llo)
            elif op == '>':
                hi = min(hi, lhi - 1)
            elif op == '>=':
                hi = min(hi, lhi)
            elif op == '==':
                lo = max(lo, llo)
                hi = min(hi, lhi)
            if lo > hi:
                return IntervalState(bottom=True)
            new_state.env[var] = (lo, hi)

        return new_state


class IntervalCPA(CPA):
    """CPA using interval abstract domain."""

    def __init__(self):
        self._transfer = IntervalTransfer()
        self._merge = MergeSep()
        self._stop = StopSep()
        self._prec_adj = NoPrecisionAdjustment()

    def initial_state(self):
        return IntervalState()

    def initial_precision(self):
        return None

    def transfer(self):
        return self._transfer

    def merge_op(self):
        return self._merge

    def stop_op(self):
        return self._stop

    def precision_adjustment(self):
        return self._prec_adj


# ============================================================
# Predicate CPA (wraps V110 predicate abstraction)
# ============================================================

class PredicateState(AbstractState):
    """Abstract state as a set of predicate indices."""

    def __init__(self, predicates: frozenset = frozenset(), bottom: bool = False):
        self.predicates = predicates
        self._bottom = bottom

    def is_bottom(self) -> bool:
        return self._bottom

    def subsumes(self, other: 'PredicateState') -> bool:
        if self._bottom:
            return other._bottom
        if other._bottom:
            return True
        # Fewer predicates = more general = subsumes
        return self.predicates.issubset(other.predicates)

    def join(self, other: 'PredicateState') -> 'PredicateState':
        if self._bottom:
            return other.copy()
        if other._bottom:
            return self.copy()
        # Intersection of predicates (shared constraints)
        return PredicateState(self.predicates & other.predicates)

    def copy(self) -> 'PredicateState':
        return PredicateState(frozenset(self.predicates), self._bottom)

    def __repr__(self) -> str:
        if self._bottom:
            return "PredicateState(BOT)"
        return f"PredicateState({set(self.predicates)})"


class PredicateRegistry:
    """Tracks predicates and their locations."""

    def __init__(self):
        self.predicates = []  # list of (smt_term, name_str)
        self.pred_map = {}    # name_str -> index
        self.location_preds = {}  # cfg_node_id -> set of pred indices

    def add_predicate(self, term, name=None, location_id=None) -> int:
        name_str = name or str(term)
        if name_str in self.pred_map:
            idx = self.pred_map[name_str]
        else:
            idx = len(self.predicates)
            self.predicates.append((term, name_str))
            self.pred_map[name_str] = idx
        if location_id is not None:
            if location_id not in self.location_preds:
                self.location_preds[location_id] = set()
            self.location_preds[location_id].add(idx)
        return idx

    def get_predicates_at(self, location_id) -> set:
        return self.location_preds.get(location_id, set()) | self.get_all_predicate_indices()

    def get_all_predicate_indices(self) -> set:
        return set(range(len(self.predicates)))

    def get_predicate_term(self, idx):
        return self.predicates[idx][0]

    def get_predicate_name(self, idx) -> str:
        return self.predicates[idx][1]


class PredicateTransfer(TransferRelation):
    """Transfer relation using predicate abstraction with SMT."""

    def __init__(self, registry: PredicateRegistry):
        self.registry = registry

    def get_abstract_successors(self, state, edge, precision):
        if state.is_bottom():
            return []

        if edge.edge_type == EdgeType.SKIP:
            return [state.copy()]

        if edge.edge_type == EdgeType.ERROR:
            return [state.copy()]

        if edge.edge_type == EdgeType.ASSIGN:
            return [self._post_assign(state, edge.data[0], edge.data[1])]

        if edge.edge_type in (EdgeType.ASSUME, EdgeType.ASSERT):
            return self._post_assume(state, edge.data, positive=True)

        if edge.edge_type == EdgeType.ASSUME_NOT:
            return self._post_assume(state, edge.data, positive=False)

        return [state.copy()]

    def _post_assign(self, state: PredicateState, var_name, expr) -> PredicateState:
        """After assignment, check which predicates hold (preserved + newly true)."""
        new_preds = set()
        env_vars = {}
        # Check ALL registered predicates (not just current ones)
        for idx in self.registry.get_all_predicate_indices():
            term = self.registry.get_predicate_term(idx)
            if idx in state.predicates:
                # Check if preserved
                if self._predicate_preserved_after_assign(state, var_name, expr, term, env_vars):
                    new_preds.add(idx)
            else:
                # Check if newly implied after assignment
                if self._predicate_implied_after_assign(state, var_name, expr, idx):
                    new_preds.add(idx)
        return PredicateState(frozenset(new_preds))

    def _predicate_preserved_after_assign(self, state, var_name, expr, pred_term, env_vars):
        """Check if pred holds after var_name := expr, given current predicates hold."""
        pred_name = str(pred_term)
        if var_name not in pred_name:
            return True
        try:
            solver = SMTSolver()
            for idx in state.predicates:
                t = self.registry.get_predicate_term(idx)
                solver.add(t)
            var_smt = Var(var_name, Sort(SortKind.INT))
            expr_smt = _ast_to_smt(expr, env_vars)
            var_new = Var(var_name + "_new", Sort(SortKind.INT))
            eq = App(Op.EQ, [var_new, expr_smt], Sort(SortKind.BOOL))
            solver.add(eq)
            # Check if pred[var -> var_new] holds
            pred_renamed = self._rename_var_in_term(pred_term, var_name, var_name + "_new")
            neg_pred = App(Op.NOT, [pred_renamed], Sort(SortKind.BOOL))
            solver.add(neg_pred)
            result = solver.check()
            return result == SMTResult.UNSAT
        except Exception:
            return False

    def _predicate_implied_after_assign(self, state, var_name, expr, pred_idx):
        """Check if predicate becomes true after var_name := expr."""
        try:
            solver = SMTSolver()
            env_vars = {}
            # Add current predicates as assumptions
            for idx in state.predicates:
                solver.add(self.registry.get_predicate_term(idx))
            # Add assignment constraint
            var_smt = Var(var_name + "_new", Sort(SortKind.INT))
            expr_smt = _ast_to_smt(expr, env_vars)
            eq = App(Op.EQ, [var_smt, expr_smt], Sort(SortKind.BOOL))
            solver.add(eq)
            # Check if NOT(pred[var -> var_new]) is UNSAT
            pred_term = self.registry.get_predicate_term(pred_idx)
            pred_renamed = self._rename_var_in_term(pred_term, var_name, var_name + "_new")
            neg = App(Op.NOT, [pred_renamed], Sort(SortKind.BOOL))
            solver.add(neg)
            result = solver.check()
            return result == SMTResult.UNSAT
        except Exception:
            return False

    def _rename_var_in_term(self, term, old_name, new_name):
        """Rename a variable in an SMT term."""
        if isinstance(term, Var):
            if term.name == old_name:
                return Var(new_name, term.sort)
            return term
        if isinstance(term, (IntConst, BoolConst)):
            return term
        if isinstance(term, App):
            new_args = [self._rename_var_in_term(a, old_name, new_name) for a in term.args]
            return App(term.op, new_args, term.sort)
        return term

    def _post_assume(self, state: PredicateState, cond, positive: bool) -> List[PredicateState]:
        """Refine state with assume condition."""
        # Check feasibility
        env_vars = {}
        try:
            solver = SMTSolver()
            for idx in state.predicates:
                solver.add(self.registry.get_predicate_term(idx))
            cond_smt = _ast_to_smt(cond, env_vars)
            if not positive:
                cond_smt = App(Op.NOT, [cond_smt], Sort(SortKind.BOOL))
            solver.add(cond_smt)
            result = solver.check()
            if result == SMTResult.UNSAT:
                return []  # infeasible
        except Exception:
            pass

        # Check which new predicates are implied
        new_preds = set(state.predicates)
        for idx in self.registry.get_all_predicate_indices():
            if idx not in new_preds:
                if self._predicate_implied_by_assume(state, cond, positive, idx):
                    new_preds.add(idx)
        return [PredicateState(frozenset(new_preds))]

    def _predicate_implied_by_assume(self, state, cond, positive, pred_idx):
        """Check if assume(cond) implies predicate."""
        try:
            solver = SMTSolver()
            env_vars = {}
            for idx in state.predicates:
                solver.add(self.registry.get_predicate_term(idx))
            cond_smt = _ast_to_smt(cond, env_vars)
            if not positive:
                cond_smt = App(Op.NOT, [cond_smt], Sort(SortKind.BOOL))
            solver.add(cond_smt)
            pred_term = self.registry.get_predicate_term(pred_idx)
            neg = App(Op.NOT, [pred_term], Sort(SortKind.BOOL))
            solver.add(neg)
            result = solver.check()
            return result == SMTResult.UNSAT
        except Exception:
            return False


class PredicateCPA(CPA):
    """CPA using predicate abstraction."""

    def __init__(self, registry: Optional[PredicateRegistry] = None):
        self.registry = registry or PredicateRegistry()
        self._transfer = PredicateTransfer(self.registry)
        self._merge = MergeSep()
        self._stop = StopSep()
        self._prec_adj = NoPrecisionAdjustment()

    def initial_state(self):
        return PredicateState()

    def initial_precision(self):
        return self.registry

    def transfer(self):
        return self._transfer

    def merge_op(self):
        return self._merge

    def stop_op(self):
        return self._stop

    def precision_adjustment(self):
        return self._prec_adj


# ============================================================
# Zone CPA (wraps V104)
# ============================================================

class ZoneState(AbstractState):
    """Abstract state using zone (DBM) domain."""

    def __init__(self, zone: Optional[ZoneDomain] = None, var_names: Optional[List[str]] = None,
                 bottom: bool = False):
        if bottom:
            self.zone = None
            self._bottom = True
        elif zone is not None:
            self.zone = zone
            self._bottom = zone.is_bot()
        else:
            self.zone = ZoneDomain(var_names or [])
            self._bottom = False

    def is_bottom(self) -> bool:
        return self._bottom

    def subsumes(self, other: 'ZoneState') -> bool:
        if self._bottom:
            return other._bottom
        if other._bottom:
            return True
        # self >= other in lattice means self has weaker constraints
        return other.zone.leq(self.zone)

    def join(self, other: 'ZoneState') -> 'ZoneState':
        if self._bottom:
            return other.copy()
        if other._bottom:
            return self.copy()
        # Ensure both have same variables
        z1, z2 = self._align(other)
        return ZoneState(zone=z1.join(z2))

    def widen(self, other: 'ZoneState') -> 'ZoneState':
        if self._bottom:
            return other.copy()
        if other._bottom:
            return self.copy()
        z1, z2 = self._align(other)
        return ZoneState(zone=z1.widen(z2))

    def _align(self, other: 'ZoneState') -> Tuple[ZoneDomain, ZoneDomain]:
        """Ensure both zones have the same variables."""
        z1 = self.zone.copy()
        z2 = other.zone.copy()
        for v in z2.var_names:
            if v not in z1.var_names:
                z1.add_var(v)
        for v in z1.var_names:
            if v not in z2.var_names:
                z2.add_var(v)
        return z1, z2

    def copy(self) -> 'ZoneState':
        if self._bottom:
            return ZoneState(bottom=True)
        return ZoneState(zone=self.zone.copy())

    def get_interval(self, var: str) -> Tuple[float, float]:
        if self._bottom or var not in self.zone.var_names:
            return (NEG_INF, INF)
        return self.zone.get_interval(var)

    def __repr__(self) -> str:
        if self._bottom:
            return "ZoneState(BOT)"
        constraints = self.zone.get_constraints() if self.zone else []
        return f"ZoneState({', '.join(constraints[:5])})"


class ZoneTransfer(TransferRelation):
    """Transfer relation for zone domain."""

    def get_abstract_successors(self, state, edge, precision):
        if state.is_bottom():
            return []

        if edge.edge_type == EdgeType.SKIP:
            return [state.copy()]

        if edge.edge_type == EdgeType.ERROR:
            return [state.copy()]

        if edge.edge_type == EdgeType.ASSIGN:
            var_name, expr = edge.data
            new_state = state.copy()
            self._apply_assign(new_state, var_name, expr)
            if new_state.is_bottom():
                return []
            return [new_state]

        if edge.edge_type == EdgeType.ASSUME:
            return self._apply_assume(state, edge.data, positive=True)

        if edge.edge_type == EdgeType.ASSUME_NOT:
            return self._apply_assume(state, edge.data, positive=False)

        if edge.edge_type == EdgeType.ASSERT:
            return [state.copy()]

        return [state.copy()]

    def _apply_assign(self, state: ZoneState, var_name: str, expr):
        """Apply assignment to zone state."""
        zone = state.zone
        if var_name not in zone.var_names:
            zone.add_var(var_name)

        cls = expr.__class__.__name__
        if cls == 'IntLit':
            zone.assign_const(var_name, expr.value)
        elif cls == 'Var':
            if expr.name in zone.var_names:
                zone.assign_var(var_name, expr.name)
            else:
                zone.forget(var_name)
        elif cls == 'BinOp' and expr.op == '+':
            if expr.left.__class__.__name__ == 'Var' and expr.right.__class__.__name__ == 'IntLit':
                if expr.left.name in zone.var_names:
                    zone.assign_add(var_name, expr.left.name, expr.right.value)
                else:
                    zone.forget(var_name)
            elif expr.left.__class__.__name__ == 'IntLit' and expr.right.__class__.__name__ == 'Var':
                if expr.right.name in zone.var_names:
                    zone.assign_add(var_name, expr.right.name, expr.left.value)
                else:
                    zone.forget(var_name)
            else:
                zone.forget(var_name)
                val = self._eval_interval(expr, state)
                if val[0] != NEG_INF:
                    zone.set_lower(var_name, val[0])
                if val[1] != INF:
                    zone.set_upper(var_name, val[1])
        elif cls == 'BinOp' and expr.op == '-':
            if expr.left.__class__.__name__ == 'Var' and expr.right.__class__.__name__ == 'IntLit':
                if expr.left.name in zone.var_names:
                    zone.assign_add(var_name, expr.left.name, -expr.right.value)
                else:
                    zone.forget(var_name)
            elif (expr.left.__class__.__name__ == 'Var' and
                  expr.right.__class__.__name__ == 'Var'):
                if (expr.left.name in zone.var_names and
                    expr.right.name in zone.var_names):
                    zone.assign_sub_vars(var_name, expr.left.name, expr.right.name)
                else:
                    zone.forget(var_name)
            else:
                zone.forget(var_name)
                val = self._eval_interval(expr, state)
                if val[0] != NEG_INF:
                    zone.set_lower(var_name, val[0])
                if val[1] != INF:
                    zone.set_upper(var_name, val[1])
        else:
            zone.forget(var_name)
            val = self._eval_interval(expr, state)
            if val[0] != NEG_INF:
                zone.set_lower(var_name, val[0])
            if val[1] != INF:
                zone.set_upper(var_name, val[1])
        zone.close()

    def _eval_interval(self, expr, state: ZoneState) -> Tuple[float, float]:
        """Evaluate expression to interval."""
        cls = expr.__class__.__name__
        if cls == 'IntLit':
            return (expr.value, expr.value)
        if cls == 'Var':
            return state.get_interval(expr.name)
        if cls == 'BinOp':
            l = self._eval_interval(expr.left, state)
            r = self._eval_interval(expr.right, state)
            if expr.op == '+':
                return (l[0] + r[0], l[1] + r[1])
            if expr.op == '-':
                return (l[0] - r[1], l[1] - r[0])
            if expr.op == '*':
                products = [l[0]*r[0], l[0]*r[1], l[1]*r[0], l[1]*r[1]]
                return (min(products), max(products))
        return (NEG_INF, INF)

    def _apply_assume(self, state: ZoneState, cond, positive: bool) -> List[ZoneState]:
        """Apply assume to zone state."""
        new_state = state.copy()
        zone = new_state.zone
        cls = cond.__class__.__name__

        if cls == 'BinOp':
            op = cond.op
            if not positive:
                op = {'<': '>=', '<=': '>', '>': '<=', '>=': '<',
                      '==': '!=', '!=': '=='}.get(op, op)

            left_cls = cond.left.__class__.__name__
            right_cls = cond.right.__class__.__name__

            if left_cls == 'Var' and right_cls == 'Var':
                lv, rv = cond.left.name, cond.right.name
                if lv in zone.var_names and rv in zone.var_names:
                    if op == '<':
                        zone.set_diff(lv, rv, -1)  # lv - rv <= -1
                    elif op == '<=':
                        zone.set_diff(lv, rv, 0)
                    elif op == '>':
                        zone.set_diff(rv, lv, -1)
                    elif op == '>=':
                        zone.set_diff(rv, lv, 0)
                    elif op == '==':
                        zone.set_diff(lv, rv, 0)
                        zone.set_diff(rv, lv, 0)
                    zone.close()

            elif left_cls == 'Var' and right_cls == 'IntLit':
                lv = cond.left.name
                c = cond.right.value
                if lv in zone.var_names:
                    if op == '<':
                        zone.set_upper(lv, c - 1)
                    elif op == '<=':
                        zone.set_upper(lv, c)
                    elif op == '>':
                        zone.set_lower(lv, c + 1)
                    elif op == '>=':
                        zone.set_lower(lv, c)
                    elif op == '==':
                        zone.set_upper(lv, c)
                        zone.set_lower(lv, c)
                    zone.close()

            elif left_cls == 'IntLit' and right_cls == 'Var':
                rv = cond.right.name
                c = cond.left.value
                if rv in zone.var_names:
                    if op == '<':
                        zone.set_lower(rv, c + 1)
                    elif op == '<=':
                        zone.set_lower(rv, c)
                    elif op == '>':
                        zone.set_upper(rv, c - 1)
                    elif op == '>=':
                        zone.set_upper(rv, c)
                    elif op == '==':
                        zone.set_upper(rv, c)
                        zone.set_lower(rv, c)
                    zone.close()

        if new_state.zone.is_bot():
            return []
        return [new_state]


class ZoneCPA(CPA):
    """CPA using zone (difference-bound) abstract domain."""

    def __init__(self):
        self._transfer = ZoneTransfer()
        self._merge = MergeSep()
        self._stop = StopSep()
        self._prec_adj = NoPrecisionAdjustment()

    def initial_state(self):
        return ZoneState()

    def initial_precision(self):
        return None

    def transfer(self):
        return self._transfer

    def merge_op(self):
        return self._merge

    def stop_op(self):
        return self._stop

    def precision_adjustment(self):
        return self._prec_adj


# ============================================================
# Composite CPA (product of multiple CPAs)
# ============================================================

class CompositeState(AbstractState):
    """Product of multiple abstract states."""

    def __init__(self, components: List[AbstractState]):
        self.components = components

    def is_bottom(self) -> bool:
        return any(c.is_bottom() for c in self.components)

    def subsumes(self, other: 'CompositeState') -> bool:
        return all(c1.subsumes(c2) for c1, c2 in zip(self.components, other.components))

    def join(self, other: 'CompositeState') -> 'CompositeState':
        return CompositeState([c1.join(c2) for c1, c2 in zip(self.components, other.components)])

    def copy(self) -> 'CompositeState':
        return CompositeState([c.copy() for c in self.components])

    def get_component(self, idx: int) -> AbstractState:
        return self.components[idx]

    def __repr__(self) -> str:
        return f"CompositeState({', '.join(repr(c) for c in self.components)})"


class CompositeTransfer(TransferRelation):
    """Applies each component CPA's transfer independently."""

    def __init__(self, transfers: List[TransferRelation]):
        self.transfers = transfers

    def get_abstract_successors(self, state: CompositeState, edge, precision):
        # Apply each transfer to its component
        component_successors = []
        for i, (transfer, comp_state) in enumerate(zip(self.transfers, state.components)):
            prec = precision[i] if isinstance(precision, list) else precision
            succs = transfer.get_abstract_successors(comp_state, edge, prec)
            if not succs:
                return []  # if any component is infeasible, whole is infeasible
            component_successors.append(succs)

        # Combine: take first successor from each component (product)
        results = []
        first_combo = [succs[0] for succs in component_successors]
        results.append(CompositeState(first_combo))
        return results


class CompositeCPA(CPA):
    """Product CPA combining multiple CPAs."""

    def __init__(self, cpas: List[CPA]):
        self.cpas = cpas
        transfers = [cpa.transfer() for cpa in cpas]
        self._transfer = CompositeTransfer(transfers)
        self._merge = MergeSep()
        self._stop = StopSep()
        self._prec_adj = NoPrecisionAdjustment()

    def initial_state(self):
        return CompositeState([cpa.initial_state() for cpa in self.cpas])

    def initial_precision(self):
        return [cpa.initial_precision() for cpa in self.cpas]

    def transfer(self):
        return self._transfer

    def merge_op(self):
        return self._merge

    def stop_op(self):
        return self._stop

    def precision_adjustment(self):
        return self._prec_adj


# ============================================================
# CPA Algorithm (generic ART exploration)
# ============================================================

@dataclass
class ARTNode:
    """Node in the Abstract Reachability Tree."""
    id: int
    cfg_node: CFGNode
    state: AbstractState
    parent: Optional['ARTNode'] = None
    children: List['ARTNode'] = field(default_factory=list)
    covered_by: Optional['ARTNode'] = None
    depth: int = 0

    @property
    def is_covered(self):
        return self.covered_by is not None


@dataclass
class CPAResult:
    """Result of CPA analysis."""
    safe: bool
    art_nodes: int = 0
    error_paths: List[List[ARTNode]] = field(default_factory=list)
    counterexample: Optional[List[Tuple[str, Any]]] = None
    counterexample_inputs: Optional[Dict[str, int]] = None
    refinement_count: int = 0
    predicates: Optional[List[str]] = None
    covered_count: int = 0
    domain_name: str = ""
    variable_ranges: Optional[Dict[str, Tuple[float, float]]] = None


def _cfg_to_edges(cfg: CFG) -> Dict[int, List[CPAEdge]]:
    """Convert CFG to edge list per source node."""
    edges = {}
    for node in cfg.nodes:
        node_edges = []
        for succ in node.successors:
            edge_type = EdgeType.SKIP
            data = None
            if node.type == CFGNodeType.ASSIGN:
                edge_type = EdgeType.ASSIGN
                data = node.data
            elif node.type == CFGNodeType.ASSUME:
                if succ == node.successors[0]:
                    edge_type = EdgeType.ASSUME
                else:
                    edge_type = EdgeType.ASSUME_NOT
                data = node.data
            elif node.type == CFGNodeType.ASSUME_NOT:
                edge_type = EdgeType.ASSUME_NOT
                data = node.data
            elif node.type == CFGNodeType.ASSERT:
                if succ.type == CFGNodeType.ERROR:
                    edge_type = EdgeType.ASSUME_NOT
                    data = node.data
                else:
                    edge_type = EdgeType.ASSUME
                    data = node.data
            elif node.type == CFGNodeType.ERROR:
                edge_type = EdgeType.ERROR
            node_edges.append(CPAEdge(source=node, target=succ, edge_type=edge_type, data=data))
        edges[node.id] = node_edges
    return edges


def cpa_algorithm(cfg: CFG, cpa: CPA, max_nodes: int = 500) -> CPAResult:
    """
    Generic CPA algorithm: explore ART using the given CPA.

    Returns CPAResult with safety verdict and analysis statistics.
    """
    transfer = cpa.transfer()
    merge = cpa.merge_op()
    stop = cpa.stop_op()
    prec_adj = cpa.precision_adjustment()
    precision = cpa.initial_precision()

    edges = _cfg_to_edges(cfg)

    # Initialize
    root = ARTNode(id=0, cfg_node=cfg.entry, state=cpa.initial_state(), depth=0)
    node_counter = 1
    covered_count = 0

    worklist = [root]
    reached = {cfg.entry.id: [root]}  # cfg_node_id -> list of ARTNodes
    all_nodes = [root]
    error_nodes = []

    while worklist and node_counter < max_nodes:
        current = worklist.pop(0)  # BFS

        if current.is_covered:
            continue

        if current.cfg_node.type == CFGNodeType.ERROR:
            error_nodes.append(current)
            continue

        for edge in edges.get(current.cfg_node.id, []):
            successors = transfer.get_abstract_successors(current.state, edge, precision)

            for succ_state in successors:
                if succ_state.is_bottom():
                    continue

                # Precision adjustment
                succ_state, precision = prec_adj.adjust(succ_state, precision,
                                                         reached.get(edge.target.id, []))

                child = ARTNode(
                    id=node_counter,
                    cfg_node=edge.target,
                    state=succ_state,
                    parent=current,
                    depth=current.depth + 1
                )
                node_counter += 1
                current.children.append(child)
                all_nodes.append(child)

                # Merge with existing reached states
                target_reached = reached.get(edge.target.id, [])
                merged = False
                for i, existing in enumerate(target_reached):
                    merged_state = merge.merge(succ_state, existing.state, precision)
                    if merged_state is not existing.state:
                        existing.state = merged_state
                        merged = True
                        break

                if not merged:
                    # Check stop (coverage)
                    if stop.stop(succ_state, target_reached, precision):
                        child.covered_by = target_reached[0] if target_reached else None
                        covered_count += 1
                    else:
                        if edge.target.id not in reached:
                            reached[edge.target.id] = []
                        reached[edge.target.id].append(child)
                        worklist.append(child)

    # Collect error paths
    error_paths = []
    for en in error_nodes:
        path = []
        node = en
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        error_paths.append(path)

    # Extract variable ranges from final states at exit
    variable_ranges = {}
    exit_states = reached.get(cfg.exit_node.id, [])
    for art_node in exit_states:
        state = art_node.state
        if isinstance(state, IntervalState):
            for var, interval in state.env.items():
                if var not in variable_ranges:
                    variable_ranges[var] = interval
                else:
                    old = variable_ranges[var]
                    variable_ranges[var] = (min(old[0], interval[0]), max(old[1], interval[1]))
        elif isinstance(state, ZoneState) and state.zone:
            for var in state.zone.var_names:
                interval = state.zone.get_interval(var)
                if var not in variable_ranges:
                    variable_ranges[var] = interval
                else:
                    old = variable_ranges[var]
                    variable_ranges[var] = (min(old[0], interval[0]), max(old[1], interval[1]))
        elif isinstance(state, CompositeState):
            for comp in state.components:
                _extract_ranges(comp, variable_ranges)

    return CPAResult(
        safe=len(error_nodes) == 0,
        art_nodes=node_counter,
        error_paths=error_paths,
        covered_count=covered_count,
        variable_ranges=variable_ranges if variable_ranges else None,
        domain_name=cpa.__class__.__name__
    )


def _extract_ranges(state, ranges):
    """Extract variable ranges from a state into dict."""
    if isinstance(state, IntervalState):
        for var, interval in state.env.items():
            if var not in ranges:
                ranges[var] = interval
            else:
                old = ranges[var]
                ranges[var] = (min(old[0], interval[0]), max(old[1], interval[1]))
    elif isinstance(state, ZoneState) and state.zone:
        for var in state.zone.var_names:
            interval = state.zone.get_interval(var)
            if var not in ranges:
                ranges[var] = interval
            else:
                old = ranges[var]
                ranges[var] = (min(old[0], interval[0]), max(old[1], interval[1]))


# ============================================================
# CEGAR with CPA
# ============================================================

def _extract_path_trace(path: List[ARTNode]) -> List[Tuple[str, Any]]:
    """Extract (type, data) trace from ART path."""
    trace = []
    for node in path:
        cfg = node.cfg_node
        if cfg.type == CFGNodeType.ASSIGN:
            trace.append(('assign', cfg.data))
        elif cfg.type == CFGNodeType.ASSUME:
            trace.append(('assume', cfg.data))
        elif cfg.type == CFGNodeType.ASSUME_NOT:
            trace.append(('assume_not', cfg.data))
        elif cfg.type == CFGNodeType.ASSERT:
            trace.append(('assert', cfg.data))
        elif cfg.type == CFGNodeType.ERROR:
            trace.append(('error', None))
        else:
            trace.append(('skip', None))
    return trace


def _check_path_feasibility(path: List[ARTNode]) -> Tuple[bool, Optional[Dict], List]:
    """Check if an error path is feasible using SMT."""
    solver = SMTSolver()
    env_vars = {}
    ssa_counter = {}
    formulas = []

    def get_ssa(name):
        idx = ssa_counter.get(name, 0)
        return f"{name}_{idx}"

    def bump_ssa(name):
        idx = ssa_counter.get(name, 0) + 1
        ssa_counter[name] = idx
        return f"{name}_{idx}"

    def ssa_rename(expr):
        """Rename variables to SSA form."""
        cls = expr.__class__.__name__
        if cls == 'Var':
            ssa_name = get_ssa(expr.name)
            return Var(ssa_name, Sort(SortKind.INT))
        if cls == 'IntLit':
            return IntConst(expr.value)
        if cls == 'BoolLit':
            return IntConst(1 if expr.value else 0)
        if cls == 'BinOp':
            left = ssa_rename(expr.left)
            right = ssa_rename(expr.right)
            op_map = {
                '+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
                '<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE,
                '==': Op.EQ, '!=': Op.NEQ,
            }
            smt_op = op_map.get(expr.op)
            if smt_op is None:
                return IntConst(0)
            if smt_op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NEQ):
                return App(smt_op, [left, right], Sort(SortKind.BOOL))
            return App(smt_op, [left, right], Sort(SortKind.INT))
        if cls == 'UnaryOp':
            if expr.op == '-':
                operand = ssa_rename(expr.operand)
                return App(Op.SUB, [IntConst(0), operand], Sort(SortKind.INT))
            if expr.op == 'not':
                operand = ssa_rename(expr.operand)
                return App(Op.NOT, [operand], Sort(SortKind.BOOL))
        return IntConst(0)

    for node in path:
        cfg = node.cfg_node
        if cfg.type == CFGNodeType.ASSIGN and cfg.data:
            var_name, expr = cfg.data
            rhs = ssa_rename(expr)
            new_name = bump_ssa(var_name)
            lhs = Var(new_name, Sort(SortKind.INT))
            formula = App(Op.EQ, [lhs, rhs], Sort(SortKind.BOOL))
            solver.add(formula)
            formulas.append(formula)
        elif cfg.type == CFGNodeType.ASSUME and cfg.data:
            formula = ssa_rename(cfg.data)
            solver.add(formula)
            formulas.append(formula)
        elif cfg.type == CFGNodeType.ASSUME_NOT and cfg.data:
            inner = ssa_rename(cfg.data)
            formula = App(Op.NOT, [inner], Sort(SortKind.BOOL))
            solver.add(formula)
            formulas.append(formula)
        elif cfg.type == CFGNodeType.ASSERT and cfg.data:
            # Assert the negation (we're checking path TO error)
            inner = ssa_rename(cfg.data)
            formula = App(Op.NOT, [inner], Sort(SortKind.BOOL))
            solver.add(formula)
            formulas.append(formula)
        else:
            formulas.append(BoolConst(True))

    result = solver.check()
    if result == SMTResult.SAT:
        model = solver.model()
        inputs = {}
        for k, v in model.items():
            name = k if isinstance(k, str) else k.name if hasattr(k, 'name') else str(k)
            if '_0' in name:
                orig = name.replace('_0', '')
                inputs[orig] = v
        return True, inputs, formulas
    return False, None, formulas


def _refine_from_path(path: List[ARTNode], formulas: List,
                      registry: PredicateRegistry) -> List[int]:
    """Refine predicate set using Craig interpolation from infeasible path."""
    try:
        from craig_interpolation import interpolate, And as CIAnd
    except ImportError:
        return []

    new_preds = []
    # Try splitting at different points
    for split in range(1, len(formulas)):
        a_formulas = [f for f in formulas[:split] if not isinstance(f, BoolConst)]
        b_formulas = [f for f in formulas[split:] if not isinstance(f, BoolConst)]
        if not a_formulas or not b_formulas:
            continue

        a_conj = a_formulas[0]
        for f in a_formulas[1:]:
            a_conj = App(Op.AND, [a_conj, f], Sort(SortKind.BOOL))
        b_conj = b_formulas[0]
        for f in b_formulas[1:]:
            b_conj = App(Op.AND, [b_conj, f], Sort(SortKind.BOOL))

        try:
            result = interpolate(a_conj, b_conj)
            if result.success and result.interpolant:
                atoms = _extract_atoms(result.interpolant)
                for atom in atoms:
                    name = str(atom)
                    if name not in registry.pred_map:
                        loc = path[split].cfg_node.id if split < len(path) else None
                        idx = registry.add_predicate(atom, name=name, location_id=loc)
                        new_preds.append(idx)
                if new_preds:
                    break
        except Exception:
            continue

    return new_preds


def _extract_atoms(term) -> list:
    """Extract atomic predicates from an SMT term."""
    atoms = []
    if isinstance(term, App):
        if term.op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NEQ):
            atoms.append(term)
        elif term.op in (Op.AND, Op.OR):
            for arg in term.args:
                atoms.extend(_extract_atoms(arg))
        elif term.op == Op.NOT:
            atoms.extend(_extract_atoms(term.args[0]))
    return atoms


def cpa_cegar(source: str, cpa: CPA, max_iterations: int = 15,
              max_nodes: int = 500) -> CPAResult:
    """
    CEGAR loop with configurable CPA.

    For predicate CPAs, refines predicates on spurious counterexamples.
    For numeric CPAs (interval/zone), returns after first analysis
    (no refinement possible without predicate domain).
    """
    cfg = build_cfg_from_source(source)

    # Seed predicates from assertions/conditions in CFG
    registry = None
    if isinstance(cpa, PredicateCPA):
        registry = cpa.registry
        _seed_predicates(cfg, registry)

    for iteration in range(max_iterations):
        result = cpa_algorithm(cfg, cpa, max_nodes=max_nodes)
        result.refinement_count = iteration

        if result.safe:
            return result

        if not result.error_paths:
            return result

        # Check feasibility of first error path
        path = result.error_paths[0]
        feasible, inputs, formulas = _check_path_feasibility(path)

        if feasible:
            result.counterexample = _extract_path_trace(path)
            result.counterexample_inputs = inputs
            return result

        # Refine (only for predicate CPA)
        if registry is not None:
            new_preds = _refine_from_path(path, formulas, registry)
            if not new_preds:
                # Fallback: extract from path conditions
                for node in path:
                    if node.cfg_node.data and node.cfg_node.type in (
                        CFGNodeType.ASSUME, CFGNodeType.ASSERT
                    ):
                        try:
                            env_vars = {}
                            term = _ast_to_smt(node.cfg_node.data, env_vars)
                            name = str(term)
                            if name not in registry.pred_map:
                                registry.add_predicate(term, name=name,
                                                       location_id=node.cfg_node.id)
                                new_preds.append(len(registry.predicates) - 1)
                        except Exception:
                            pass
                if not new_preds:
                    return result  # can't refine further
            result.predicates = [registry.get_predicate_name(i) for i in
                                 registry.get_all_predicate_indices()]
        else:
            # Non-predicate CPA: can't refine
            return result

    return result


def _seed_predicates(cfg: CFG, registry: PredicateRegistry):
    """Extract initial predicates from CFG assertions and conditions."""
    env_vars = {}
    for node in cfg.nodes:
        if node.type in (CFGNodeType.ASSUME, CFGNodeType.ASSERT) and node.data:
            try:
                term = _ast_to_smt(node.data, env_vars)
                registry.add_predicate(term, location_id=node.id)
            except Exception:
                pass


# ============================================================
# Public API
# ============================================================

def verify_with_intervals(source: str, max_nodes: int = 500) -> CPAResult:
    """Verify program using interval CPA."""
    cpa = IntervalCPA()
    cfg = build_cfg_from_source(source)
    result = cpa_algorithm(cfg, cpa, max_nodes=max_nodes)
    result.domain_name = "IntervalCPA"
    return result


def verify_with_zones(source: str, max_nodes: int = 500) -> CPAResult:
    """Verify program using zone CPA."""
    cpa = ZoneCPA()
    cfg = build_cfg_from_source(source)
    result = cpa_algorithm(cfg, cpa, max_nodes=max_nodes)
    result.domain_name = "ZoneCPA"
    return result


def verify_with_predicates(source: str, max_iterations: int = 15,
                           max_nodes: int = 500) -> CPAResult:
    """Verify program using predicate CPA with CEGAR refinement."""
    cpa = PredicateCPA()
    result = cpa_cegar(source, cpa, max_iterations=max_iterations, max_nodes=max_nodes)
    result.domain_name = "PredicateCPA"
    return result


def verify_with_composite(source: str, cpas: Optional[List[str]] = None,
                          max_nodes: int = 500) -> CPAResult:
    """Verify program using composite CPA (product of multiple domains).

    Args:
        source: C10 source code
        cpas: List of CPA names ('interval', 'zone', 'predicate'). Default: ['interval', 'zone']
        max_nodes: Max ART nodes
    """
    cpa_names = cpas or ['interval', 'zone']
    cpa_list = []
    for name in cpa_names:
        if name == 'interval':
            cpa_list.append(IntervalCPA())
        elif name == 'zone':
            cpa_list.append(ZoneCPA())
        elif name == 'predicate':
            cpa_list.append(PredicateCPA())
    composite = CompositeCPA(cpa_list)
    cfg = build_cfg_from_source(source)
    result = cpa_algorithm(cfg, composite, max_nodes=max_nodes)
    result.domain_name = f"CompositeCPA({'+'.join(cpa_names)})"
    return result


def compare_cpas(source: str, max_nodes: int = 500) -> Dict:
    """Compare different CPA configurations on the same program.

    Returns dict with per-CPA results and comparison summary.
    """
    results = {}

    # Interval CPA
    try:
        r = verify_with_intervals(source, max_nodes)
        results['interval'] = {
            'safe': r.safe, 'nodes': r.art_nodes, 'covered': r.covered_count,
            'ranges': r.variable_ranges, 'domain': r.domain_name
        }
    except Exception as e:
        results['interval'] = {'error': str(e)}

    # Zone CPA
    try:
        r = verify_with_zones(source, max_nodes)
        results['zone'] = {
            'safe': r.safe, 'nodes': r.art_nodes, 'covered': r.covered_count,
            'ranges': r.variable_ranges, 'domain': r.domain_name
        }
    except Exception as e:
        results['zone'] = {'error': str(e)}

    # Predicate CPA with CEGAR
    try:
        r = verify_with_predicates(source, max_nodes=max_nodes)
        results['predicate'] = {
            'safe': r.safe, 'nodes': r.art_nodes, 'covered': r.covered_count,
            'refinements': r.refinement_count, 'predicates': r.predicates,
            'domain': r.domain_name
        }
    except Exception as e:
        results['predicate'] = {'error': str(e)}

    # Composite (interval + zone)
    try:
        r = verify_with_composite(source, max_nodes=max_nodes)
        results['composite'] = {
            'safe': r.safe, 'nodes': r.art_nodes, 'covered': r.covered_count,
            'ranges': r.variable_ranges, 'domain': r.domain_name
        }
    except Exception as e:
        results['composite'] = {'error': str(e)}

    # Summary
    verdicts = {}
    for name, r in results.items():
        if 'error' not in r:
            verdicts[name] = r['safe']
    agreement = len(set(verdicts.values())) <= 1 if verdicts else False

    return {
        'results': results,
        'agreement': agreement,
        'verdicts': verdicts
    }


def get_variable_ranges(source: str, cpa_name: str = 'interval') -> Dict[str, Tuple[float, float]]:
    """Get variable ranges computed by the specified CPA.

    Args:
        source: C10 source code
        cpa_name: 'interval' or 'zone'
    """
    if cpa_name == 'zone':
        result = verify_with_zones(source)
    else:
        result = verify_with_intervals(source)
    return result.variable_ranges or {}


def cpa_summary(source: str) -> Dict:
    """Get summary statistics for all CPAs on the given program."""
    comparison = compare_cpas(source)
    cfg = build_cfg_from_source(source)
    return {
        'cfg_nodes': len(cfg.nodes),
        'comparison': comparison
    }
