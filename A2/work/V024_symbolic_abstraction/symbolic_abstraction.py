"""V024: Symbolic Abstraction

Computes optimal abstract transformers for predicate domains using symbolic
execution. Given predicates P1,...,Pk and a program S, symbolic abstraction
computes the BEST abstract post-state by:

1. Representing the pre-state symbolically (conjunction of assumed predicates)
2. Running symbolic execution to explore all paths
3. For each path, using SMT to determine which predicates hold in the post-state
4. Joining results across paths

This is strictly more precise than Cartesian abstraction (V010) because it
preserves predicate correlations.

Composes: C038 (symbolic execution) + C037 (SMT solver) + C010 (parser) +
          V010 (predicate abstraction concepts) + V002 (transition systems)
"""

import os, sys

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
sys.path.insert(0, os.path.join(_az, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C038_symbolic_execution'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_work, 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(_work, 'V010_predicate_abstraction_cegar'))

from smt_solver import (SMTSolver, SMTResult, Var, IntConst, BoolConst,
                         App, Op, BOOL, INT, Term)
from symbolic_execution import SymbolicExecutor, PathState, SymValue, SymType
from stack_vm import lex, Parser, IntLit
from stack_vm import Var as ASTVar
from pdr import TransitionSystem
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Predicate representation
# ---------------------------------------------------------------------------

@dataclass
class Predicate:
    """A named SMT formula over program variables."""
    name: str
    formula: Term  # SMT term over unprimed variables

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Predicate) and self.name == other.name


class PredValue(Enum):
    """Truth value of a predicate in an abstract state."""
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Abstract state: mapping from predicates to truth values
# ---------------------------------------------------------------------------

@dataclass
class PredicateState:
    """Abstract state: a valuation of predicates.

    Unlike Cartesian abstraction, this can represent correlations:
    e.g., "if P1 is true then P2 must be true" is captured by only
    having states where P1=T implies P2=T.
    """
    values: Dict[str, PredValue]  # pred_name -> truth value
    predicates: List[Predicate]   # reference predicate list

    def __repr__(self):
        parts = []
        for p in self.predicates:
            v = self.values.get(p.name, PredValue.UNKNOWN)
            if v == PredValue.TRUE:
                parts.append(f"+{p.name}")
            elif v == PredValue.FALSE:
                parts.append(f"-{p.name}")
        if not parts:
            return "PredicateState(TOP)"
        return f"PredicateState({', '.join(parts)})"

    def is_top(self):
        return all(v == PredValue.UNKNOWN for v in self.values.values())

    def is_bot(self):
        return self.values.get('__bot__') == PredValue.TRUE

    @staticmethod
    def top(predicates):
        return PredicateState(
            {p.name: PredValue.UNKNOWN for p in predicates},
            predicates
        )

    @staticmethod
    def bot(predicates):
        vals = {p.name: PredValue.UNKNOWN for p in predicates}
        vals['__bot__'] = PredValue.TRUE
        return PredicateState(vals, predicates)

    def join(self, other):
        """Least upper bound: unknown if disagreeing."""
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        result = {}
        for p in self.predicates:
            sv = self.values.get(p.name, PredValue.UNKNOWN)
            ov = other.values.get(p.name, PredValue.UNKNOWN)
            if sv == ov:
                result[p.name] = sv
            else:
                result[p.name] = PredValue.UNKNOWN
        return PredicateState(result, self.predicates)

    def meet(self, other):
        """Greatest lower bound: keep known values."""
        if self.is_bot() or other.is_bot():
            return PredicateState.bot(self.predicates)
        if self.is_top():
            return other
        if other.is_top():
            return self
        result = {}
        for p in self.predicates:
            sv = self.values.get(p.name, PredValue.UNKNOWN)
            ov = other.values.get(p.name, PredValue.UNKNOWN)
            if sv != PredValue.UNKNOWN and ov != PredValue.UNKNOWN and sv != ov:
                return PredicateState.bot(self.predicates)
            if sv != PredValue.UNKNOWN:
                result[p.name] = sv
            else:
                result[p.name] = ov
        return PredicateState(result, self.predicates)

    def leq(self, other):
        """self <= other: self is more precise (has more known values)."""
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        for p in self.predicates:
            sv = self.values.get(p.name, PredValue.UNKNOWN)
            ov = other.values.get(p.name, PredValue.UNKNOWN)
            if ov != PredValue.UNKNOWN and sv != ov:
                return False
            if sv != PredValue.UNKNOWN and ov == PredValue.UNKNOWN:
                # self is more precise -- that's OK (self <= other means
                # self is lower/more precise in the lattice)
                pass
        return True

    def known_predicates(self):
        """Return predicates with definite truth values."""
        return {name: val for name, val in self.values.items()
                if val != PredValue.UNKNOWN and name != '__bot__'}

    def definite_true(self):
        """Predicates known to be true."""
        return [p for p in self.predicates
                if self.values.get(p.name) == PredValue.TRUE]

    def definite_false(self):
        """Predicates known to be false."""
        return [p for p in self.predicates
                if self.values.get(p.name) == PredValue.FALSE]


# ---------------------------------------------------------------------------
# SMT helpers
# ---------------------------------------------------------------------------

def _negate(term):
    """Negate an SMT term using complement operators."""
    if isinstance(term, App):
        complements = {
            Op.EQ: Op.NEQ, Op.NEQ: Op.EQ,
            Op.LT: Op.GE, Op.GE: Op.LT,
            Op.LE: Op.GT, Op.GT: Op.LE,
        }
        if term.op in complements:
            return App(complements[term.op], term.args, term.sort)
        if term.op == Op.AND:
            return App(Op.OR, [_negate(a) for a in term.args], BOOL)
        if term.op == Op.OR:
            return App(Op.AND, [_negate(a) for a in term.args], BOOL)
        if term.op == Op.NOT:
            return term.args[0]
    return App(Op.NOT, [term], BOOL)


def _smt_and(terms):
    """Conjoin a list of SMT terms."""
    terms = [t for t in terms if t is not None]
    if not terms:
        return BoolConst(True)
    result = terms[0]
    for t in terms[1:]:
        result = App(Op.AND, [result, t], BOOL)
    return result


def _smt_or(terms):
    """Disjoin a list of SMT terms."""
    terms = [t for t in terms if t is not None]
    if not terms:
        return BoolConst(False)
    result = terms[0]
    for t in terms[1:]:
        result = App(Op.OR, [result, t], BOOL)
    return result


# ---------------------------------------------------------------------------
# Core: Symbolic abstract post computation
# ---------------------------------------------------------------------------

def _evaluate_predicate_in_path(pred, path, var_map, extra_constraints=None):
    """Check if a predicate holds in a path's post-state.

    Args:
        pred: Predicate with formula over program variables
        path: PathState from symbolic execution
        var_map: mapping from variable names to their SMT terms in post-state
        extra_constraints: additional SMT constraints (e.g., from pre-state)

    Returns: PredValue (TRUE, FALSE, or UNKNOWN)
    """
    # Substitute program vars in predicate formula with their post-state values
    substituted = _substitute_formula(pred.formula, var_map)

    base_constraints = list(path.constraints)
    if extra_constraints:
        base_constraints.extend(extra_constraints)

    # Check if predicate definitely holds: base AND NOT(pred) is UNSAT
    s_pos = SMTSolver()
    for c in base_constraints:
        s_pos.add(c)
    s_pos.add(_negate(substituted))
    if s_pos.check() == SMTResult.UNSAT:
        return PredValue.TRUE

    # Check if predicate definitely fails: base AND pred is UNSAT
    s_neg = SMTSolver()
    for c in base_constraints:
        s_neg.add(c)
    s_neg.add(substituted)
    if s_neg.check() == SMTResult.UNSAT:
        return PredValue.FALSE

    return PredValue.UNKNOWN


def _substitute_formula(formula, var_map):
    """Replace Var references in formula with post-state terms from var_map."""
    if isinstance(formula, Var):
        return var_map.get(formula.name, formula)
    if isinstance(formula, (IntConst, BoolConst)):
        return formula
    if isinstance(formula, App):
        new_args = [_substitute_formula(a, var_map) for a in formula.args]
        return App(formula.op, new_args, formula.sort)
    return formula


def _extract_var_map(path):
    """Extract mapping from variable names to their SMT terms in a path's post-state."""
    var_map = {}
    for name, sym_val in path.env.items():
        if name.startswith('__') or name.startswith('_sym'):
            continue
        if isinstance(sym_val, SymValue):
            if sym_val.kind == SymType.SYMBOLIC and sym_val.term is not None:
                var_map[name] = sym_val.term
            elif sym_val.kind == SymType.CONCRETE and sym_val.concrete is not None:
                var_map[name] = IntConst(int(sym_val.concrete))
        elif isinstance(sym_val, (int, float)):
            var_map[name] = IntConst(int(sym_val))
    return var_map


def symbolic_abstract_post(source, predicates, symbolic_inputs,
                           pre_state=None, max_paths=64):
    """Compute the optimal abstract post-state using symbolic execution.

    This is the BEST abstract transformer: alpha . f . gamma.
    - gamma: represent pre-state as SMT constraints on symbolic inputs
    - f: symbolic execution explores all paths
    - alpha: evaluate predicates in each path's post-state, join across paths

    Args:
        source: C010 source code
        predicates: list of Predicate objects
        symbolic_inputs: dict of {var_name: 'int'} for symbolic variables
        pre_state: optional PredicateState constraining the pre-state
        max_paths: max paths for symbolic execution

    Returns: PredicateState (the abstract post)
    """
    executor = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=5)
    result = executor.execute(source, symbolic_inputs)

    # Filter to feasible completed paths
    from symbolic_execution import PathStatus
    feasible_paths = [p for p in result.paths
                      if p.status in (PathStatus.COMPLETED, PathStatus.ASSERTION_FAILED)]

    if not feasible_paths:
        return PredicateState.bot(predicates)

    # For each path, evaluate all predicates; then join across paths
    joined = None
    paths_analyzed = 0

    for path in feasible_paths:
        var_map = _extract_var_map(path)

        # Build extra constraints from pre-state
        extra_constraints = []
        if pre_state and not pre_state.is_top():
            if not _path_consistent_with_pre(path, pre_state, symbolic_inputs):
                continue
            extra_constraints = _pre_state_to_constraints(pre_state, symbolic_inputs)

        # Evaluate each predicate in this path's post-state
        path_values = {}
        for pred in predicates:
            path_values[pred.name] = _evaluate_predicate_in_path(
                pred, path, var_map, extra_constraints=extra_constraints)

        path_state = PredicateState(path_values, predicates)
        paths_analyzed += 1

        if joined is None:
            joined = path_state
        else:
            joined = joined.join(path_state)

    if joined is None:
        return PredicateState.bot(predicates)

    return joined


def _path_consistent_with_pre(path, pre_state, symbolic_inputs):
    """Check if a path's initial constraints are consistent with the pre-state predicates."""
    # The pre-state predicates constrain the initial values of symbolic inputs.
    # We need to check that the path constraints are satisfiable together
    # with the pre-state predicate constraints.
    s = SMTSolver()
    for c in path.constraints:
        s.add(c)

    for pred in pre_state.definite_true():
        # The predicate should hold at the start -- substitute initial vars
        init_map = {}
        for name in symbolic_inputs:
            init_map[name] = Var(name, INT)
        sub = _substitute_formula(pred.formula, init_map)
        s.add(sub)

    for pred in pre_state.definite_false():
        init_map = {}
        for name in symbolic_inputs:
            init_map[name] = Var(name, INT)
        sub = _substitute_formula(pred.formula, init_map)
        s.add(_negate(sub))

    return s.check() != SMTResult.UNSAT


def _pre_state_to_constraints(pre_state, symbolic_inputs):
    """Convert pre-state predicate assumptions to SMT constraints."""
    constraints = []
    init_map = {name: Var(name, INT) for name in symbolic_inputs}

    for p in pre_state.definite_true():
        constraints.append(_substitute_formula(p.formula, init_map))

    for p in pre_state.definite_false():
        constraints.append(_negate(_substitute_formula(p.formula, init_map)))

    return constraints


# ---------------------------------------------------------------------------
# Predicate discovery from symbolic execution
# ---------------------------------------------------------------------------

def discover_predicates(source, symbolic_inputs, max_predicates=20,
                        max_paths=64):
    """Discover interesting predicates by analyzing symbolic execution paths.

    Extracts predicates from:
    1. Path conditions (branch points)
    2. Variable relationships in post-states
    3. Comparison operations in the program

    Args:
        source: C10 source code
        symbolic_inputs: dict of {var_name: 'int'}
        max_predicates: max number of predicates to return
        max_paths: max paths for symbolic execution

    Returns: list of Predicate objects
    """
    executor = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=5)
    result = executor.execute(source, symbolic_inputs)

    predicates = {}  # formula_str -> Predicate

    # Extract from path conditions
    for path in result.paths:
        for constraint in path.constraints:
            _extract_predicates_from_term(constraint, predicates)

    # Extract from AST (comparisons, assignments)
    _extract_predicates_from_source(source, symbolic_inputs, predicates)

    # Deduplicate and limit
    pred_list = list(predicates.values())
    if len(pred_list) > max_predicates:
        pred_list = pred_list[:max_predicates]

    return pred_list


def _extract_predicates_from_term(term, predicates):
    """Extract atomic predicates from an SMT term."""
    if isinstance(term, App):
        if term.op in (Op.EQ, Op.NEQ, Op.LT, Op.LE, Op.GT, Op.GE):
            key = str(term)
            if key not in predicates:
                name = _make_pred_name(term)
                predicates[key] = Predicate(name, term)
        # Recurse into AND/OR
        if term.op in (Op.AND, Op.OR, Op.NOT):
            for arg in term.args:
                _extract_predicates_from_term(arg, predicates)


def _extract_predicates_from_source(source, symbolic_inputs, predicates):
    """Extract predicates from source code AST."""
    try:
        tokens = lex(source)
        program = Parser(tokens).parse()
    except Exception:
        return

    var_names = list(symbolic_inputs.keys())

    # Add non-negativity predicates for all symbolic vars
    for name in var_names:
        formula = App(Op.GE, [Var(name, INT), IntConst(0)], BOOL)
        key = str(formula)
        if key not in predicates:
            predicates[key] = Predicate(f"{name}_ge_0", formula)

    # Add pairwise comparisons for vars
    for i, n1 in enumerate(var_names):
        for n2 in var_names[i+1:]:
            for op, op_name in [(Op.EQ, 'eq'), (Op.LT, 'lt'), (Op.LE, 'le')]:
                formula = App(op, [Var(n1, INT), Var(n2, INT)], BOOL)
                key = str(formula)
                if key not in predicates:
                    predicates[key] = Predicate(f"{n1}_{op_name}_{n2}", formula)


def _make_pred_name(term):
    """Generate a readable name for a predicate term."""
    if isinstance(term, App):
        op_names = {
            Op.EQ: 'eq', Op.NEQ: 'neq', Op.LT: 'lt',
            Op.LE: 'le', Op.GT: 'gt', Op.GE: 'ge'
        }
        if term.op in op_names and len(term.args) == 2:
            left = _term_name(term.args[0])
            right = _term_name(term.args[1])
            return f"{left}_{op_names[term.op]}_{right}"
    return f"pred_{hash(str(term)) % 10000}"


def _term_name(term):
    """Short name for a term."""
    if isinstance(term, Var):
        return term.name
    if isinstance(term, IntConst):
        return str(term.value)
    if isinstance(term, App) and term.op == Op.ADD:
        return f"{_term_name(term.args[0])}_plus_{_term_name(term.args[1])}"
    if isinstance(term, App) and term.op == Op.SUB:
        return f"{_term_name(term.args[0])}_minus_{_term_name(term.args[1])}"
    return "expr"


# ---------------------------------------------------------------------------
# Full program analysis with symbolic abstraction
# ---------------------------------------------------------------------------

@dataclass
class AnalysisPoint:
    """Analysis result at a specific program point."""
    location: str
    state: PredicateState
    description: str = ""


@dataclass
class SymbolicAbstractionResult:
    """Result of full symbolic abstraction analysis."""
    points: List[AnalysisPoint]
    predicates: List[Predicate]
    paths_explored: int
    predicate_correlations: List[Tuple[str, str, str]]  # (p1, relation, p2)
    precision_gains: List[str]  # descriptions of precision over Cartesian


def symbolic_abstraction_analyze(source, predicates, symbolic_inputs,
                                 max_paths=64):
    """Full program analysis using symbolic abstraction.

    Tracks predicate truth values through the entire program using
    optimal (symbolic) abstract transformers at each point.

    Args:
        source: C10 source code
        predicates: list of Predicates
        symbolic_inputs: dict of {var_name: 'int'}
        max_paths: max paths for symbolic execution

    Returns: SymbolicAbstractionResult
    """
    executor = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=5)
    result = executor.execute(source, symbolic_inputs)

    from symbolic_execution import PathStatus
    feasible_paths = [p for p in result.paths
                      if p.status in (PathStatus.COMPLETED, PathStatus.ASSERTION_FAILED)]

    points = []
    correlations = []

    # Compute post-state predicate values for each feasible path
    path_states = []
    for path in feasible_paths:
        var_map = _extract_var_map(path)
        path_values = {}
        for pred in predicates:
            path_values[pred.name] = _evaluate_predicate_in_path(
                pred, path, var_map)
        path_states.append(PredicateState(path_values, predicates))

    # Join across all paths for the overall post-state
    if path_states:
        overall = path_states[0]
        for ps in path_states[1:]:
            overall = overall.join(ps)
    else:
        overall = PredicateState.bot(predicates)

    points.append(AnalysisPoint("program_exit", overall, "After execution"))

    # Detect correlations: do any path-specific states show predicate implications?
    correlations = _detect_correlations(path_states, predicates)

    # Compute precision gains vs Cartesian
    precision_gains = _compute_precision_gains(path_states, predicates)

    return SymbolicAbstractionResult(
        points=points,
        predicates=predicates,
        paths_explored=len(feasible_paths),
        predicate_correlations=correlations,
        precision_gains=precision_gains,
    )


def _detect_correlations(path_states, predicates):
    """Detect implications/correlations between predicates across paths."""
    correlations = []
    pred_names = [p.name for p in predicates]

    for i, p1 in enumerate(pred_names):
        for p2 in pred_names[i+1:]:
            # Check: does p1=TRUE always imply p2=TRUE?
            p1_implies_p2 = True
            p2_implies_p1 = True
            p1_excludes_p2 = True  # p1=T => p2=F
            seen_p1_true = False
            seen_p2_true = False

            for ps in path_states:
                v1 = ps.values.get(p1, PredValue.UNKNOWN)
                v2 = ps.values.get(p2, PredValue.UNKNOWN)

                if v1 == PredValue.TRUE:
                    seen_p1_true = True
                    if v2 != PredValue.TRUE:
                        p1_implies_p2 = False
                    if v2 != PredValue.FALSE:
                        p1_excludes_p2 = False
                if v2 == PredValue.TRUE:
                    seen_p2_true = True
                    if v1 != PredValue.TRUE:
                        p2_implies_p1 = False

            if seen_p1_true and p1_implies_p2:
                correlations.append((p1, "implies", p2))
            if seen_p2_true and p2_implies_p1:
                correlations.append((p2, "implies", p1))
            if seen_p1_true and p1_excludes_p2:
                correlations.append((p1, "excludes", p2))

    return correlations


def _compute_precision_gains(path_states, predicates):
    """Identify where symbolic abstraction is more precise than Cartesian."""
    gains = []

    if len(path_states) < 2:
        return gains

    # Cartesian: join all paths independently per predicate
    cartesian = PredicateState.top(predicates)
    for ps in path_states:
        cartesian = cartesian.join(ps)

    # Check if any path reveals predicate correlations lost in Cartesian
    for i, p1 in enumerate(predicates):
        for p2 in predicates[i+1:]:
            # In all paths, check if p1+p2 have a fixed relationship
            relationships = set()
            for ps in path_states:
                v1 = ps.values.get(p1.name, PredValue.UNKNOWN)
                v2 = ps.values.get(p2.name, PredValue.UNKNOWN)
                if v1 != PredValue.UNKNOWN and v2 != PredValue.UNKNOWN:
                    relationships.add((v1, v2))

            # If all paths agree on a correlation but Cartesian loses it...
            if len(relationships) > 0:
                # Check if all valid (v1,v2) pairs are a proper subset of {T,F}x{T,F}
                cv1 = cartesian.values.get(p1.name, PredValue.UNKNOWN)
                cv2 = cartesian.values.get(p2.name, PredValue.UNKNOWN)

                if cv1 == PredValue.UNKNOWN and cv2 == PredValue.UNKNOWN:
                    # Cartesian lost info about both -- did any path know both?
                    for v1, v2 in relationships:
                        if v1 != PredValue.UNKNOWN and v2 != PredValue.UNKNOWN:
                            gains.append(
                                f"Correlation {p1.name}<->{p2.name}: "
                                f"paths show {v1.value}/{v2.value} but "
                                f"Cartesian loses both to UNKNOWN"
                            )
                            break

    return gains


# ---------------------------------------------------------------------------
# Transition system abstraction via symbolic execution
# ---------------------------------------------------------------------------

@dataclass
class AbstractTransition:
    """A single abstract transition: pre_state -> post_state."""
    pre: PredicateState
    post: PredicateState
    feasible: bool
    path_count: int  # how many concrete paths witness this


@dataclass
class AbstractTransformerResult:
    """Complete abstract transformer for a transition system."""
    transitions: List[AbstractTransition]
    predicates: List[Predicate]
    total_concrete_paths: int
    abstract_states_reachable: int


def compute_abstract_transformer(ts, predicates, max_depth=10):
    """Compute the optimal abstract transformer for a transition system.

    For each abstract pre-state (combination of predicate values), computes
    the best abstract post-state by unrolling the transition relation and
    evaluating predicates via SMT.

    Unlike Cartesian abstraction (V010), this considers predicate combinations,
    not each predicate independently.

    Args:
        ts: TransitionSystem (V002) with init, trans, property formulas
        predicates: list of Predicates
        max_depth: max unrolling depth for BMC-style exploration

    Returns: AbstractTransformerResult
    """
    transitions = []
    total_paths = 0

    # Evaluate predicates in init state
    init_state = _evaluate_predicates_in_formula(predicates, ts.init_formula)

    # For each possible abstract pre-state, compute abstract post
    # Start with the definite init state
    pre_states_to_explore = [init_state]
    explored = set()

    while pre_states_to_explore:
        pre = pre_states_to_explore.pop(0)
        pre_key = tuple(sorted(pre.values.items()))
        if pre_key in explored:
            continue
        explored.add(pre_key)

        # Build SMT formula for this abstract pre-state
        pre_constraints = _state_to_constraints(pre, predicates)

        # Compute abstract post via transition relation
        post, n_paths = _abstract_post_via_trans(
            ts, predicates, pre_constraints)

        total_paths += n_paths

        at = AbstractTransition(
            pre=pre, post=post,
            feasible=(not post.is_bot()),
            path_count=n_paths
        )
        transitions.append(at)

        # Add post-state to exploration queue if new
        if not post.is_bot() and not post.is_top():
            post_key = tuple(sorted(post.values.items()))
            if post_key not in explored:
                pre_states_to_explore.append(post)

        if len(explored) > 2 ** len(predicates):
            break  # Exhausted all abstract states

    return AbstractTransformerResult(
        transitions=transitions,
        predicates=predicates,
        total_concrete_paths=total_paths,
        abstract_states_reachable=len(explored),
    )


def _evaluate_predicates_in_formula(predicates, region_formula):
    """Evaluate all predicates in a given SMT formula region."""
    values = {}
    for pred in predicates:
        s_pos = SMTSolver()
        s_pos.add(region_formula)
        s_pos.add(_negate(pred.formula))
        if s_pos.check() == SMTResult.UNSAT:
            values[pred.name] = PredValue.TRUE
            continue

        s_neg = SMTSolver()
        s_neg.add(region_formula)
        s_neg.add(pred.formula)
        if s_neg.check() == SMTResult.UNSAT:
            values[pred.name] = PredValue.FALSE
            continue

        values[pred.name] = PredValue.UNKNOWN

    return PredicateState(values, predicates)


def _state_to_constraints(state, predicates):
    """Convert a PredicateState to a list of SMT constraints."""
    constraints = []
    for pred in predicates:
        v = state.values.get(pred.name, PredValue.UNKNOWN)
        if v == PredValue.TRUE:
            constraints.append(pred.formula)
        elif v == PredValue.FALSE:
            constraints.append(_negate(pred.formula))
    return constraints


def _abstract_post_via_trans(ts, predicates, pre_constraints):
    """Compute abstract post-state given pre-state constraints and transition."""
    # Check if pre-state + init is satisfiable
    s = SMTSolver()
    for c in pre_constraints:
        s.add(c)
    # Don't add init -- we're computing post from any state matching pre_constraints

    # Check if pre-state is satisfiable at all
    s_check = SMTSolver()
    for c in pre_constraints:
        s_check.add(c)
    if s_check.check() == SMTResult.UNSAT:
        return PredicateState.bot(predicates), 0

    # Apply transition: pre_constraints AND trans => what holds for primed vars?
    s_trans = SMTSolver()
    for c in pre_constraints:
        s_trans.add(c)
    s_trans.add(ts.trans_formula)

    if s_trans.check() == SMTResult.UNSAT:
        return PredicateState.bot(predicates), 0

    # Evaluate each predicate on the primed (post) variables
    values = {}
    paths_checked = 0

    for pred in predicates:
        # Substitute unprimed vars with primed vars in predicate
        primed_pred = _prime_formula(pred.formula, [name for name, _ in ts.state_vars])

        # Check if predicate definitely holds after transition
        s_pos = SMTSolver()
        for c in pre_constraints:
            s_pos.add(c)
        s_pos.add(ts.trans_formula)
        s_pos.add(_negate(primed_pred))
        paths_checked += 1
        if s_pos.check() == SMTResult.UNSAT:
            values[pred.name] = PredValue.TRUE
            continue

        # Check if predicate definitely fails after transition
        s_neg = SMTSolver()
        for c in pre_constraints:
            s_neg.add(c)
        s_neg.add(ts.trans_formula)
        s_neg.add(primed_pred)
        paths_checked += 1
        if s_neg.check() == SMTResult.UNSAT:
            values[pred.name] = PredValue.FALSE
            continue

        values[pred.name] = PredValue.UNKNOWN

    return PredicateState(values, predicates), paths_checked


def _prime_formula(formula, var_names):
    """Replace all var references with primed versions (x -> x')."""
    if isinstance(formula, Var):
        if formula.name in var_names:
            return Var(formula.name + "'", formula.sort)
        return formula
    if isinstance(formula, (IntConst, BoolConst)):
        return formula
    if isinstance(formula, App):
        new_args = [_prime_formula(a, var_names) for a in formula.args]
        return App(formula.op, new_args, formula.sort)
    return formula


# ---------------------------------------------------------------------------
# Comparison with Cartesian abstraction (V010)
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of comparing symbolic vs Cartesian abstraction."""
    symbolic_state: PredicateState
    cartesian_state: PredicateState
    symbolic_more_precise: bool
    precision_gains: List[str]
    correlations_found: List[Tuple[str, str, str]]


def compare_with_cartesian(source, predicates, symbolic_inputs,
                           max_paths=64):
    """Compare symbolic abstraction with Cartesian abstraction.

    Cartesian computes each predicate independently (loses correlations).
    Symbolic considers all predicates together (preserves correlations).

    Args:
        source: C10 source code
        predicates: list of Predicates
        symbolic_inputs: dict of {var_name: 'int'}

    Returns: ComparisonResult
    """
    # Symbolic abstraction
    sym_result = symbolic_abstraction_analyze(
        source, predicates, symbolic_inputs, max_paths)

    # Cartesian: evaluate each predicate independently
    executor = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=5)
    result = executor.execute(source, symbolic_inputs)

    from symbolic_execution import PathStatus
    feasible_paths = [p for p in result.paths
                      if p.status in (PathStatus.COMPLETED, PathStatus.ASSERTION_FAILED)]

    # Cartesian: for each predicate, check all paths independently
    cart_values = {}
    for pred in predicates:
        all_true = True
        all_false = True
        any_path = False

        for path in feasible_paths:
            var_map = _extract_var_map(path)
            pv = _evaluate_predicate_in_path(pred, path, var_map)
            any_path = True

            if pv != PredValue.TRUE:
                all_true = False
            if pv != PredValue.FALSE:
                all_false = False

        if not any_path:
            cart_values[pred.name] = PredValue.UNKNOWN
        elif all_true:
            cart_values[pred.name] = PredValue.TRUE
        elif all_false:
            cart_values[pred.name] = PredValue.FALSE
        else:
            cart_values[pred.name] = PredValue.UNKNOWN

    cartesian_state = PredicateState(cart_values, predicates)

    # The overall symbolic state
    sym_state = sym_result.points[0].state if sym_result.points else \
        PredicateState.top(predicates)

    # Compare precision
    # Note: for the join over all paths, Cartesian and symbolic give the SAME
    # per-predicate truth values. The precision gain is in the CORRELATIONS:
    # symbolic abstraction can express "if P1 then P2" while Cartesian cannot.

    gains = sym_result.precision_gains
    is_more_precise = len(gains) > 0 or len(sym_result.predicate_correlations) > 0

    return ComparisonResult(
        symbolic_state=sym_state,
        cartesian_state=cartesian_state,
        symbolic_more_precise=is_more_precise,
        precision_gains=gains,
        correlations_found=sym_result.predicate_correlations,
    )


# ---------------------------------------------------------------------------
# Transition system comparison with V010
# ---------------------------------------------------------------------------

def compare_ts_abstraction(ts, predicates, max_depth=10):
    """Compare symbolic vs Cartesian abstraction for a transition system.

    Shows that symbolic abstraction computes the best (most precise)
    abstract transformer, while Cartesian may lose predicate correlations.

    Returns: dict with comparison details
    """
    # Symbolic abstraction
    sym = compute_abstract_transformer(ts, predicates, max_depth)

    # Cartesian: compute each predicate's transitions independently
    cart_transitions = _cartesian_abstraction(ts, predicates)

    return {
        'symbolic': {
            'transitions': len(sym.transitions),
            'reachable_states': sym.abstract_states_reachable,
            'total_paths': sym.total_concrete_paths,
        },
        'cartesian': {
            'transitions': len(cart_transitions),
        },
        'symbolic_result': sym,
        'cartesian_transitions': cart_transitions,
    }


def _cartesian_abstraction(ts, predicates):
    """Simple Cartesian abstraction: check each predicate independently."""
    transitions = []

    for pred in predicates:
        # Check: can pred be preserved (T->T)?
        s = SMTSolver()
        s.add(pred.formula)
        s.add(ts.trans_formula)
        primed = _prime_formula(pred.formula, [name for name, _ in ts.state_vars])
        s.add(primed)
        t_to_t = s.check() == SMTResult.SAT

        # Check: can pred go from T->F?
        s = SMTSolver()
        s.add(pred.formula)
        s.add(ts.trans_formula)
        s.add(_negate(primed))
        t_to_f = s.check() == SMTResult.SAT

        # Check: can pred go from F->T?
        s = SMTSolver()
        s.add(_negate(pred.formula))
        s.add(ts.trans_formula)
        s.add(primed)
        f_to_t = s.check() == SMTResult.SAT

        # Check: can pred go from F->F?
        s = SMTSolver()
        s.add(_negate(pred.formula))
        s.add(ts.trans_formula)
        s.add(_negate(primed))
        f_to_f = s.check() == SMTResult.SAT

        transitions.append({
            'predicate': pred.name,
            'T_to_T': t_to_t,
            'T_to_F': t_to_f,
            'F_to_T': f_to_t,
            'F_to_F': f_to_f,
        })

    return transitions


# ---------------------------------------------------------------------------
# Convenience: auto-discover + analyze
# ---------------------------------------------------------------------------

def auto_symbolic_abstraction(source, symbolic_inputs, max_predicates=10,
                              max_paths=64):
    """Discover predicates automatically and run symbolic abstraction analysis.

    Combines predicate discovery with symbolic abstraction for a
    fully automatic analysis pipeline.

    Args:
        source: C10 source code
        symbolic_inputs: dict of {var_name: 'int'}
        max_predicates: max number of predicates to discover
        max_paths: max paths for symbolic execution

    Returns: SymbolicAbstractionResult
    """
    predicates = discover_predicates(
        source, symbolic_inputs, max_predicates, max_paths)

    if not predicates:
        return SymbolicAbstractionResult(
            points=[], predicates=[], paths_explored=0,
            predicate_correlations=[], precision_gains=[])

    return symbolic_abstraction_analyze(
        source, predicates, symbolic_inputs, max_paths)


# ---------------------------------------------------------------------------
# Source-level transition system verification
# ---------------------------------------------------------------------------

def verify_with_symbolic_abstraction(source, property_pred, predicates=None,
                                     symbolic_inputs=None, max_paths=64):
    """Verify a property about a program using symbolic abstraction.

    Extracts predicates, runs symbolic abstraction, and checks if the
    property predicate holds in all reachable states.

    Args:
        source: C10 source code
        property_pred: Predicate representing the property to verify
        predicates: optional additional predicates (auto-discovered if None)
        symbolic_inputs: dict of {var_name: 'int'} (auto-detected if None)
        max_paths: max symbolic execution paths

    Returns: dict with verdict ('HOLDS', 'VIOLATED', 'UNKNOWN') and details
    """
    # Auto-detect symbolic inputs if not provided
    if symbolic_inputs is None:
        symbolic_inputs = _auto_detect_inputs(source)

    # Build predicate set
    if predicates is None:
        predicates = discover_predicates(source, symbolic_inputs, 15, max_paths)

    # Ensure property is in predicate set
    prop_names = {p.name for p in predicates}
    if property_pred.name not in prop_names:
        predicates = [property_pred] + predicates

    # Run symbolic abstraction
    result = symbolic_abstraction_analyze(
        source, predicates, symbolic_inputs, max_paths)

    # Check property in the overall post-state
    if result.points:
        post = result.points[0].state
        prop_val = post.values.get(property_pred.name, PredValue.UNKNOWN)

        if prop_val == PredValue.TRUE:
            verdict = 'HOLDS'
        elif prop_val == PredValue.FALSE:
            verdict = 'VIOLATED'
        else:
            verdict = 'UNKNOWN'
    else:
        verdict = 'UNKNOWN'

    return {
        'verdict': verdict,
        'post_state': result.points[0].state if result.points else None,
        'predicates': predicates,
        'paths_explored': result.paths_explored,
        'correlations': result.predicate_correlations,
        'precision_gains': result.precision_gains,
    }


def _auto_detect_inputs(source):
    """Auto-detect symbolic inputs from source (variables used before assignment)."""
    try:
        tokens = lex(source)
        program = Parser(tokens).parse()
    except Exception:
        return {}

    assigned = set()
    used_before_assign = set()

    def _walk_expr(expr):
        if isinstance(expr, ASTVar):
            if expr.name not in assigned:
                used_before_assign.add(expr.name)

    def _walk_stmts(stmts):
        from stack_vm import LetDecl, Assign, IfStmt, WhileStmt, FnDecl, ReturnStmt
        from stack_vm import BinOp, UnaryOp, CallExpr
        for stmt in stmts:
            if hasattr(stmt, '__class__'):
                cls = stmt.__class__.__name__
                if cls == 'LetDecl':
                    _walk_expr_tree(stmt.value)
                    assigned.add(stmt.name)
                elif cls == 'Assign':
                    _walk_expr_tree(stmt.value)
                    assigned.add(stmt.name)
                elif cls == 'IfStmt':
                    _walk_expr_tree(stmt.cond)
                    _walk_stmts(stmt.then_body if isinstance(stmt.then_body, list)
                                else stmt.then_body.stmts if hasattr(stmt.then_body, 'stmts')
                                else [])
                    if stmt.else_body:
                        _walk_stmts(stmt.else_body if isinstance(stmt.else_body, list)
                                    else stmt.else_body.stmts if hasattr(stmt.else_body, 'stmts')
                                    else [])
                elif cls == 'WhileStmt':
                    _walk_expr_tree(stmt.cond)
                    _walk_stmts(stmt.body if isinstance(stmt.body, list)
                                else stmt.body.stmts if hasattr(stmt.body, 'stmts')
                                else [])

    def _walk_expr_tree(expr):
        if isinstance(expr, ASTVar):
            if expr.name not in assigned:
                used_before_assign.add(expr.name)
        elif hasattr(expr, '__class__'):
            cls = expr.__class__.__name__
            if cls == 'BinOp':
                _walk_expr_tree(expr.left)
                _walk_expr_tree(expr.right)
            elif cls == 'UnaryOp':
                _walk_expr_tree(expr.operand)
            elif cls == 'CallExpr':
                for a in expr.args:
                    _walk_expr_tree(a)

    _walk_stmts(program.stmts)

    return {name: 'int' for name in used_before_assign}
