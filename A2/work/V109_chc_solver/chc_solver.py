"""
V109: Constrained Horn Clause (CHC) Solver

Constrained Horn Clauses unify many verification problems into a single formalism.
A CHC system is a set of clauses of the form:
    P1(x1) AND P2(x2) AND ... AND phi(x) => P(y)
where P1, P2, ..., P are uninterpreted predicates and phi is an SMT constraint.

Safety verification = finding interpretations for predicates that satisfy all clauses.
Special clauses:
  - Fact/Init: phi(x) => P(x)  (no body predicates)
  - Query: P(x) AND phi(x) => false  (conclusion is false)

CHC solving subsumes:
  - Loop invariant inference (invariant = predicate interpretation)
  - CEGAR (counterexample = derivation tree, refinement = new predicate interpretation)
  - PDR/IC3 (frames = candidate interpretations, propagation = inductive strengthening)

Composes: C037 (SMT solver), V107 (Craig interpolation), V002 (PDR/IC3)

Solving strategies:
  1. PDR-based: frame-based incremental solving (adapts V002)
  2. CEGAR-based: abstract-refine with interpolation (uses V107)
  3. BMC-based: bounded unrolling for counterexample finding
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set, FrozenSet
from enum import Enum
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V107_craig_interpolation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))

from smt_solver import (
    SMTSolver, SMTResult, Term, Var, IntConst, BoolConst, App, Op,
    Sort, SortKind, INT, BOOL
)
from craig_interpolation import (
    CraigInterpolator, InterpolantResult, collect_vars, substitute_term, simplify_term
)
from pdr import TransitionSystem, check_ts, PDRResult, PDROutput


# --- Formula utilities ---

def _and(*terms):
    """Flatten conjunction."""
    flat = []
    for t in terms:
        if isinstance(t, BoolConst) and t.value is True:
            continue
        if isinstance(t, BoolConst) and t.value is False:
            return BoolConst(False)
        if isinstance(t, App) and t.op == Op.AND:
            flat.extend(t.args)
        else:
            flat.append(t)
    if not flat:
        return BoolConst(True)
    if len(flat) == 1:
        return flat[0]
    return App(Op.AND, flat, BOOL)


def _or(*terms):
    """Flatten disjunction."""
    flat = []
    for t in terms:
        if isinstance(t, BoolConst) and t.value is False:
            continue
        if isinstance(t, BoolConst) and t.value is True:
            return BoolConst(True)
        if isinstance(t, App) and t.op == Op.OR:
            flat.extend(t.args)
        else:
            flat.append(t)
    if not flat:
        return BoolConst(False)
    if len(flat) == 1:
        return flat[0]
    return App(Op.OR, flat, BOOL)


def _not(t):
    """Smart negation using complement operators."""
    if isinstance(t, BoolConst):
        return BoolConst(not t.value)
    if isinstance(t, App) and t.op == Op.NOT:
        return t.args[0]
    complement = {
        Op.EQ: Op.NEQ, Op.NEQ: Op.EQ,
        Op.LT: Op.GE, Op.GE: Op.LT,
        Op.LE: Op.GT, Op.GT: Op.LE,
    }
    if isinstance(t, App) and t.op in complement:
        return App(complement[t.op], t.args, BOOL)
    return App(Op.NOT, [t], BOOL)


def _implies(a, b):
    return App(Op.IMPLIES, [a, b], BOOL)


def _eq(a, b):
    if hasattr(a, 'sort') and a.sort == BOOL:
        return App(Op.IFF, [a, b], BOOL)
    return App(Op.EQ, [a, b], BOOL)


def _substitute(term, mapping):
    """Substitute variables by name."""
    if isinstance(term, Var):
        return mapping.get(term.name, term)
    if isinstance(term, (IntConst, BoolConst)):
        return term
    if isinstance(term, App):
        new_args = [_substitute(a, mapping) for a in term.args]
        return App(term.op, new_args, term.sort)
    return term


def _smt_check(formula):
    """Check satisfiability, return (result, model)."""
    s = SMTSolver()
    s.add(formula)
    result = s.check()
    if result == SMTResult.SAT:
        return result, s.model()
    return result, None


def _check_implication(a, b):
    """Check if a => b is valid (a AND NOT(b) is UNSAT)."""
    result, _ = _smt_check(_and(a, _not(b)))
    return result == SMTResult.UNSAT


# --- CHC Data Structures ---

class CHCResult(Enum):
    SAT = "sat"          # Predicates can be interpreted to satisfy all clauses
    UNSAT = "unsat"      # No interpretation exists (property violation)
    UNKNOWN = "unknown"  # Resource limit


@dataclass
class Predicate:
    """An uninterpreted predicate symbol with arity."""
    name: str
    params: List[Tuple[str, Sort]]  # [(param_name, sort), ...]

    @property
    def arity(self):
        return len(self.params)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Predicate) and self.name == other.name


@dataclass
class PredicateApp:
    """Application of a predicate to arguments: P(t1, t2, ...)."""
    predicate: Predicate
    args: List[Term]

    def __repr__(self):
        arg_strs = [str(a) for a in self.args]
        return f"{self.predicate.name}({', '.join(arg_strs)})"


@dataclass
class HornClause:
    """
    A constrained Horn clause:
        body_preds[0] AND body_preds[1] AND ... AND constraint => head

    head is either:
      - PredicateApp (regular clause)
      - None (query clause, meaning conclusion is False)

    body_preds: list of PredicateApp in the body
    constraint: SMT formula (the constraint phi)
    """
    head: Optional[PredicateApp]  # None = query (conclusion is False)
    body_preds: List[PredicateApp]
    constraint: Term  # SMT constraint

    @property
    def is_fact(self):
        """Fact clause: no body predicates."""
        return len(self.body_preds) == 0 and self.head is not None

    @property
    def is_query(self):
        """Query clause: head is False."""
        return self.head is None

    @property
    def is_linear(self):
        """Linear clause: at most one body predicate."""
        return len(self.body_preds) <= 1

    def __repr__(self):
        parts = []
        for bp in self.body_preds:
            parts.append(str(bp))
        parts.append(f"phi")
        body = " AND ".join(parts)
        head = str(self.head) if self.head else "false"
        return f"{body} => {head}"


@dataclass
class CHCSystem:
    """A system of constrained Horn clauses."""
    predicates: Dict[str, Predicate]  # name -> Predicate
    clauses: List[HornClause]

    def __init__(self):
        self.predicates = {}
        self.clauses = []

    def add_predicate(self, name, params):
        """Add an uninterpreted predicate. params: [(name, sort), ...]"""
        pred = Predicate(name, params)
        self.predicates[name] = pred
        return pred

    def add_clause(self, head, body_preds, constraint):
        """Add a Horn clause."""
        clause = HornClause(head=head, body_preds=body_preds, constraint=constraint)
        self.clauses.append(clause)
        return clause

    def add_fact(self, pred_app, constraint):
        """Add a fact clause: constraint => pred_app."""
        return self.add_clause(head=pred_app, body_preds=[], constraint=constraint)

    def add_query(self, body_preds, constraint):
        """Add a query clause: body_preds AND constraint => false."""
        return self.add_clause(head=None, body_preds=body_preds, constraint=constraint)

    @property
    def is_linear(self):
        """All clauses have at most one body predicate."""
        return all(c.is_linear for c in self.clauses)

    def get_facts(self):
        return [c for c in self.clauses if c.is_fact]

    def get_queries(self):
        return [c for c in self.clauses if c.is_query]

    def get_rules(self):
        """Non-fact, non-query clauses."""
        return [c for c in self.clauses if not c.is_fact and not c.is_query]

    def predicates_in_body(self):
        """Set of predicate names appearing in clause bodies."""
        result = set()
        for c in self.clauses:
            for bp in c.body_preds:
                result.add(bp.predicate.name)
        return result

    def predicates_in_head(self):
        """Set of predicate names appearing in clause heads."""
        result = set()
        for c in self.clauses:
            if c.head is not None:
                result.add(c.head.predicate.name)
        return result


@dataclass
class Interpretation:
    """Maps predicate names to SMT formulas (over predicate parameters)."""
    mapping: Dict[str, Term]  # pred_name -> formula over pred's params

    def get(self, pred_name):
        return self.mapping.get(pred_name, BoolConst(True))

    def set(self, pred_name, formula):
        self.mapping[pred_name] = formula

    def __repr__(self):
        lines = []
        for name, formula in self.mapping.items():
            lines.append(f"  {name} -> {formula}")
        return "Interpretation:\n" + "\n".join(lines)


@dataclass
class Derivation:
    """A derivation tree (counterexample) showing how a query is satisfied."""
    clause: HornClause
    children: List  # List of Derivation (one per body predicate)
    model: Optional[Dict]  # Variable assignment at this step

    def depth(self):
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def __repr__(self):
        return f"Derivation(depth={self.depth()})"


@dataclass
class CHCStats:
    """Statistics about a CHC solving run."""
    smt_queries: int = 0
    iterations: int = 0
    interpolations: int = 0
    predicates_discovered: int = 0
    clauses_processed: int = 0
    strategy: str = ""


@dataclass
class CHCOutput:
    """Result of CHC solving."""
    result: CHCResult
    interpretation: Optional[Interpretation] = None  # If SAT
    derivation: Optional[Derivation] = None  # If UNSAT (counterexample)
    stats: CHCStats = field(default_factory=CHCStats)


# --- Predicate Application Helper ---

def apply_pred(pred, args):
    """Create a predicate application."""
    return PredicateApp(predicate=pred, args=args)


# --- Instantiation: replace predicate params with actual args ---

def instantiate_pred(interpretation, pred_app):
    """
    Given an interpretation for pred and a PredicateApp, substitute
    the predicate's formal parameters with the actual arguments.
    """
    pred = pred_app.predicate
    formula = interpretation.get(pred.name)
    if formula is None:
        return BoolConst(True)
    mapping = {}
    for (param_name, _sort), arg in zip(pred.params, pred_app.args):
        mapping[param_name] = arg
    return _substitute(formula, mapping)


# --- Strategy 1: BMC-based solving (counterexample finding) ---

class BMCSolver:
    """
    Bounded model checking for CHC systems.
    Unrolls the clause system up to a fixed depth to find counterexamples.
    """

    def __init__(self, system, max_depth=20):
        self.system = system
        self.max_depth = max_depth
        self.stats = CHCStats(strategy="bmc")

    def solve(self):
        """Try to find a counterexample by bounded unrolling."""
        for depth in range(self.max_depth + 1):
            result = self._check_depth(depth)
            if result is not None:
                return result
        return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

    def _check_depth(self, depth):
        """Check if a query can be derived in exactly `depth` steps."""
        queries = self.system.get_queries()
        for query in queries:
            derivation = self._try_derive_query(query, depth)
            if derivation is not None:
                return CHCOutput(
                    result=CHCResult.UNSAT,
                    derivation=derivation,
                    stats=self.stats
                )
        return None

    def _try_derive_query(self, query, max_depth):
        """Try to build a derivation tree for a query clause."""
        # Collect all formulas needed along the derivation
        formulas = []
        var_counter = [0]

        def fresh_var(name, sort):
            var_counter[0] += 1
            return Var(f"{name}_{var_counter[0]}", sort)

        derivation = self._build_derivation(query, max_depth, formulas, var_counter)
        if derivation is None:
            return None

        # Check if the conjunction of all formulas is satisfiable
        if not formulas:
            formulas = [BoolConst(True)]
        conjunction = _and(*formulas)
        self.stats.smt_queries += 1
        result, model = _smt_check(conjunction)
        if result == SMTResult.SAT:
            self._annotate_derivation(derivation, model)
            return derivation
        return None

    def _build_derivation(self, clause, remaining_depth, formulas, var_counter):
        """
        Recursively build a derivation tree.
        Returns Derivation or None if no derivation exists at this depth.
        """
        # Create fresh variables for this clause instance
        fresh_mapping = {}
        all_vars_in_clause = set()
        self._collect_clause_vars(clause, all_vars_in_clause)
        for vname in all_vars_in_clause:
            var_counter[0] += 1
            fresh_mapping[vname] = Var(f"{vname}_{var_counter[0]}", INT)

        # Add the constraint (with fresh variables)
        fresh_constraint = _substitute(clause.constraint, fresh_mapping)
        formulas.append(fresh_constraint)

        # For the head, bind its args to the fresh versions of head pred params
        if clause.head is not None:
            for (param_name, _sort), arg in zip(
                clause.head.predicate.params, clause.head.args
            ):
                fresh_param = fresh_mapping.get(param_name, Var(param_name, INT))
                fresh_arg = _substitute(arg, fresh_mapping)
                formulas.append(_eq(fresh_param, fresh_arg))

        # Process body predicates
        children = []
        for body_pred_app in clause.body_preds:
            if remaining_depth <= 0:
                # Cannot go deeper, try fact clauses only
                child = self._try_resolve_with_fact(body_pred_app, fresh_mapping, formulas, var_counter)
                if child is None:
                    return None
                children.append(child)
            else:
                # Try to resolve this body predicate
                child = self._resolve_body_pred(body_pred_app, remaining_depth - 1,
                                                fresh_mapping, formulas, var_counter)
                if child is None:
                    return None
                children.append(child)

        return Derivation(clause=clause, children=children, model=None)

    def _resolve_body_pred(self, pred_app, remaining_depth, parent_mapping, formulas, var_counter):
        """Try to resolve a body predicate using matching clauses."""
        pred_name = pred_app.predicate.name
        # Find clauses whose head matches this predicate
        matching = [c for c in self.system.clauses
                    if c.head is not None and c.head.predicate.name == pred_name]

        # Try facts first (depth 0), then deeper clauses
        for clause in sorted(matching, key=lambda c: 0 if c.is_fact else 1):
            if not clause.is_fact and remaining_depth <= 0:
                continue

            saved_len = len(formulas)
            # Create argument binding
            fresh_args = [_substitute(a, parent_mapping) for a in pred_app.args]

            sub_derivation = self._build_derivation(clause, remaining_depth, formulas, var_counter)
            if sub_derivation is not None:
                # Bind the head's args to match the body pred's args
                for (param_name, _sort), fresh_arg in zip(clause.head.predicate.params, fresh_args):
                    # Find the fresh version of this param in the sub-derivation
                    # The last set of fresh vars created by _build_derivation
                    formulas.append(_eq(fresh_arg, _substitute(Var(param_name, INT), {})))
                return sub_derivation

            # Backtrack
            del formulas[saved_len:]

        return None

    def _try_resolve_with_fact(self, pred_app, parent_mapping, formulas, var_counter):
        """Try to resolve a body predicate using only fact clauses."""
        pred_name = pred_app.predicate.name
        facts = [c for c in self.system.clauses
                 if c.is_fact and c.head.predicate.name == pred_name]

        for fact in facts:
            saved_len = len(formulas)
            sub_derivation = self._build_derivation(fact, 0, formulas, var_counter)
            if sub_derivation is not None:
                return sub_derivation
            del formulas[saved_len:]

        return None

    def _collect_clause_vars(self, clause, result):
        """Collect all variable names used in a clause."""
        if clause.head is not None:
            for (pname, _) in clause.head.predicate.params:
                result.add(pname)
            for arg in clause.head.args:
                result.update(collect_vars(arg))
        for bp in clause.body_preds:
            for arg in bp.args:
                result.update(collect_vars(arg))
        result.update(collect_vars(clause.constraint))

    def _annotate_derivation(self, derivation, model):
        """Annotate derivation nodes with concrete values from SMT model."""
        if model:
            derivation.model = dict(model)
        for child in derivation.children:
            self._annotate_derivation(child, model)


# --- Strategy 2: PDR-based solving (for linear CHC) ---

class PDRCHCSolver:
    """
    Solves linear CHC systems by reducing to transition system verification.

    For a linear CHC system with a single predicate P:
      Fact: phi_init(x) => P(x)
      Rule: P(x) AND phi_trans(x, x') => P(x')
      Query: P(x) AND phi_bad(x) => false

    This maps directly to:
      Init = phi_init
      Trans = phi_trans
      Property = NOT(phi_bad)

    For multi-predicate linear systems, we chain predicates.
    """

    def __init__(self, system, max_frames=100):
        self.system = system
        self.max_frames = max_frames
        self.stats = CHCStats(strategy="pdr")

    def solve(self):
        """Solve by reducing to PDR."""
        if not self.system.is_linear:
            return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

        # Identify the structure
        pred_names = list(self.system.predicates.keys())

        if len(pred_names) == 1:
            return self._solve_single_predicate(pred_names[0])
        else:
            return self._solve_multi_predicate()

    def _solve_single_predicate(self, pred_name):
        """Single-predicate linear CHC -> direct PDR."""
        pred = self.system.predicates[pred_name]
        facts = [c for c in self.system.clauses
                 if c.is_fact and c.head.predicate.name == pred_name]
        rules = [c for c in self.system.clauses
                 if not c.is_fact and not c.is_query
                 and c.head is not None and c.head.predicate.name == pred_name
                 and any(bp.predicate.name == pred_name for bp in c.body_preds)]
        queries = [c for c in self.system.clauses if c.is_query]

        if not facts or not queries:
            return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

        # Build transition system
        ts = TransitionSystem()
        var_map = {}
        for pname, psort in pred.params:
            v = ts.add_var(pname, psort)
            var_map[pname] = v

        # Init: disjunction of fact constraints (with arg binding)
        init_parts = []
        for fact in facts:
            binding = {}
            for (pname, _), arg in zip(pred.params, fact.head.args):
                if isinstance(arg, Var) and arg.name != pname:
                    binding[arg.name] = var_map[pname]
            constraint = _substitute(fact.constraint, binding)
            init_parts.append(constraint)
        ts.set_init(_or(*init_parts) if len(init_parts) > 1 else init_parts[0])

        # Trans: disjunction of rule transitions
        trans_parts = []
        for rule in rules:
            # Body pred binds current-state params
            body_app = [bp for bp in rule.body_preds if bp.predicate.name == pred_name][0]
            binding = {}
            for (pname, _), arg in zip(pred.params, body_app.args):
                if isinstance(arg, Var) and arg.name != pname:
                    binding[arg.name] = var_map[pname]

            # Head pred binds next-state params
            head_binding = {}
            for (pname, _), arg in zip(pred.params, rule.head.args):
                primed = ts.prime(pname)
                if isinstance(arg, Var) and arg.name != pname:
                    head_binding[arg.name] = primed
                elif isinstance(arg, Var) and arg.name == pname:
                    head_binding[pname] = primed

            full_binding = {**binding, **head_binding}
            constraint = _substitute(rule.constraint, full_binding)

            # Bind head args to primed vars
            head_eqs = []
            for (pname, _), arg in zip(pred.params, rule.head.args):
                primed = ts.prime(pname)
                arg_sub = _substitute(arg, full_binding)
                if not (isinstance(arg_sub, Var) and arg_sub.name == primed.name):
                    head_eqs.append(_eq(primed, arg_sub))

            trans_parts.append(_and(constraint, *head_eqs) if head_eqs else constraint)

        if not trans_parts:
            # No rules: identity transition
            id_parts = [_eq(ts.prime(pname), var_map[pname]) for pname, _ in pred.params]
            ts.set_trans(_and(*id_parts))
        else:
            ts.set_trans(_or(*trans_parts) if len(trans_parts) > 1 else trans_parts[0])

        # Property: negation of query constraints
        prop_parts = []
        for query in queries:
            if query.body_preds:
                body_app = query.body_preds[0]
                binding = {}
                for (pname, _), arg in zip(pred.params, body_app.args):
                    if isinstance(arg, Var) and arg.name != pname:
                        binding[arg.name] = var_map[pname]
                bad = _substitute(query.constraint, binding)
            else:
                bad = query.constraint
            prop_parts.append(bad)

        bad_formula = _or(*prop_parts) if len(prop_parts) > 1 else prop_parts[0]
        ts.set_property(_not(bad_formula))

        # Run PDR
        pdr_output = check_ts(ts, max_frames=self.max_frames)
        self.stats.smt_queries = pdr_output.stats.smt_queries
        self.stats.iterations = pdr_output.stats.frames_created

        if pdr_output.result == PDRResult.SAFE:
            # Extract interpretation from invariant
            interp = Interpretation(mapping={})
            if pdr_output.invariant:
                inv_formula = _and(*pdr_output.invariant)
                interp.set(pred_name, inv_formula)
            else:
                interp.set(pred_name, ts.prop_formula)
            return CHCOutput(result=CHCResult.SAT, interpretation=interp, stats=self.stats)

        elif pdr_output.result == PDRResult.UNSAFE:
            # Build derivation from counterexample
            derivation = self._build_derivation_from_cex(pdr_output.counterexample)
            return CHCOutput(result=CHCResult.UNSAT, derivation=derivation, stats=self.stats)

        return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

    def _solve_multi_predicate(self):
        """
        Multi-predicate linear CHC: chain predicates.
        For P1 -> P2 -> ... -> Pk -> query, encode as multi-phase TS.
        """
        # Build dependency graph
        deps = defaultdict(set)  # head_pred -> set of body_preds
        for clause in self.system.clauses:
            if clause.head is not None:
                head_name = clause.head.predicate.name
                for bp in clause.body_preds:
                    deps[head_name].add(bp.predicate.name)

        # For query clauses, track which predicates feed queries
        query_preds = set()
        for q in self.system.get_queries():
            for bp in q.body_preds:
                query_preds.add(bp.predicate.name)

        # Find entry predicates (appear in facts but not in rule bodies of other preds)
        fact_preds = {c.head.predicate.name for c in self.system.get_facts()}
        body_preds_all = set()
        for clause in self.system.clauses:
            for bp in clause.body_preds:
                body_preds_all.add(bp.predicate.name)

        # Simple case: sequential chain P1 -> P2 -> query
        # For now, handle this by encoding as a single expanded TS
        # with phase variable
        if len(self.system.predicates) <= 3:
            return self._encode_chained_ts()

        return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

    def _encode_chained_ts(self):
        """Encode a multi-predicate CHC as a single TS with phase variable."""
        # Topological ordering of predicates
        pred_order = self._topo_sort_predicates()
        if pred_order is None:
            return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

        # Assign phase numbers
        phase_map = {name: i for i, name in enumerate(pred_order)}

        # Build TS with all vars from all predicates + phase var
        ts = TransitionSystem()
        phase = ts.add_int_var("__phase")
        all_param_vars = {}  # (pred_name, param_idx) -> Var

        # Collect unique param names across all predicates
        param_names = set()
        for pred in self.system.predicates.values():
            for pname, _ in pred.params:
                param_names.add(pname)

        var_map = {}
        for pname in sorted(param_names):
            v = ts.add_int_var(pname)
            var_map[pname] = v

        # Init: phase=0 AND fact constraints for first predicate
        first_pred = pred_order[0]
        facts = [c for c in self.system.get_facts()
                 if c.head.predicate.name == first_pred]
        init_parts = [_eq(phase, IntConst(0))]
        for fact in facts:
            pred = fact.head.predicate
            binding = {}
            for (pname, _), arg in zip(pred.params, fact.head.args):
                if isinstance(arg, Var) and arg.name != pname:
                    binding[arg.name] = var_map[pname]
            init_parts.append(_substitute(fact.constraint, binding))
        ts.set_init(_and(*init_parts))

        # Trans: union of all rule transitions with phase guards
        trans_parts = []
        for clause in self.system.get_rules():
            if clause.head is None:
                continue
            head_name = clause.head.predicate.name
            if head_name not in phase_map:
                continue
            head_phase = phase_map[head_name]

            for bp in clause.body_preds:
                body_name = bp.predicate.name
                if body_name not in phase_map:
                    continue
                body_phase = phase_map[body_name]

                # Guard: current phase matches body pred
                guard = _eq(phase, IntConst(body_phase))
                # Next phase matches head pred
                next_phase = _eq(ts.prime("__phase"), IntConst(head_phase))

                # Build binding
                binding = {}
                body_pred = self.system.predicates[body_name]
                for (pname, _), arg in zip(body_pred.params, bp.args):
                    if isinstance(arg, Var) and arg.name != pname:
                        binding[arg.name] = var_map[pname]

                head_pred = self.system.predicates[head_name]
                head_binding = {}
                for (pname, _), arg in zip(head_pred.params, clause.head.args):
                    primed = ts.prime(pname)
                    if isinstance(arg, Var):
                        head_binding[arg.name] = primed

                full_binding = {**binding, **head_binding}
                constraint = _substitute(clause.constraint, full_binding)

                trans_parts.append(_and(guard, next_phase, constraint))

        # Self-loop transitions (same-predicate rules)
        for clause in self.system.get_rules():
            if clause.head is None:
                continue
            head_name = clause.head.predicate.name
            for bp in clause.body_preds:
                if bp.predicate.name == head_name:
                    ph = phase_map[head_name]
                    guard = _eq(phase, IntConst(ph))
                    next_phase = _eq(ts.prime("__phase"), IntConst(ph))

                    pred = self.system.predicates[head_name]
                    binding = {}
                    for (pname, _), arg in zip(pred.params, bp.args):
                        if isinstance(arg, Var) and arg.name != pname:
                            binding[arg.name] = var_map[pname]
                    head_binding = {}
                    for (pname, _), arg in zip(pred.params, clause.head.args):
                        primed = ts.prime(pname)
                        if isinstance(arg, Var):
                            head_binding[arg.name] = primed
                    full_binding = {**binding, **head_binding}
                    constraint = _substitute(clause.constraint, full_binding)
                    trans_parts.append(_and(guard, next_phase, constraint))

        if trans_parts:
            ts.set_trans(_or(*trans_parts) if len(trans_parts) > 1 else trans_parts[0])
        else:
            return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

        # Property: NOT(query_phase AND query_constraint)
        query_parts = []
        for query in self.system.get_queries():
            for bp in query.body_preds:
                body_name = bp.predicate.name
                if body_name in phase_map:
                    ph = phase_map[body_name]
                    pred = self.system.predicates[body_name]
                    binding = {}
                    for (pname, _), arg in zip(pred.params, bp.args):
                        if isinstance(arg, Var) and arg.name != pname:
                            binding[arg.name] = var_map[pname]
                    bad = _and(_eq(phase, IntConst(ph)),
                               _substitute(query.constraint, binding))
                    query_parts.append(bad)

        if not query_parts:
            return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

        bad_formula = _or(*query_parts) if len(query_parts) > 1 else query_parts[0]
        ts.set_property(_not(bad_formula))

        # Run PDR
        pdr_output = check_ts(ts, max_frames=self.max_frames)
        self.stats.smt_queries = pdr_output.stats.smt_queries
        self.stats.iterations = pdr_output.stats.frames_created

        if pdr_output.result == PDRResult.SAFE:
            interp = Interpretation(mapping={})
            for name in pred_order:
                if pdr_output.invariant:
                    interp.set(name, _and(*pdr_output.invariant))
                else:
                    interp.set(name, BoolConst(True))
            return CHCOutput(result=CHCResult.SAT, interpretation=interp, stats=self.stats)
        elif pdr_output.result == PDRResult.UNSAFE:
            derivation = self._build_derivation_from_cex(pdr_output.counterexample)
            return CHCOutput(result=CHCResult.UNSAT, derivation=derivation, stats=self.stats)

        return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

    def _topo_sort_predicates(self):
        """Topological sort of predicates by dependency. Returns list or None if cyclic."""
        # Build dep graph: if P appears in body and Q in head of same clause, Q depends on P
        deps = defaultdict(set)
        all_preds = set(self.system.predicates.keys())
        for clause in self.system.clauses:
            if clause.head is not None:
                head_name = clause.head.predicate.name
                for bp in clause.body_preds:
                    if bp.predicate.name != head_name:
                        deps[head_name].add(bp.predicate.name)

        # Kahn's algorithm
        in_degree = {name: 0 for name in all_preds}
        for name, dep_set in deps.items():
            for dep in dep_set:
                if dep in in_degree:
                    pass
            in_degree[name] = len(dep_set & all_preds)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        result = []
        visited = set()

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            result.append(node)
            for name in all_preds:
                if node in deps.get(name, set()):
                    in_degree[name] -= 1
                    if in_degree[name] <= 0 and name not in visited:
                        queue.append(name)

        if len(result) != len(all_preds):
            return None  # Cyclic
        return result

    def _build_derivation_from_cex(self, cex):
        """Build a Derivation from a PDR counterexample."""
        if cex is None:
            return None
        return Derivation(
            clause=self.system.clauses[0] if self.system.clauses else None,
            children=[],
            model=cex.trace[0] if cex.trace else None
        )


# --- Strategy 3: Interpolation-based CEGAR for CHC ---

class InterpCHCSolver:
    """
    CEGAR-based CHC solver using Craig interpolation for refinement.

    1. Start with coarse predicate interpretation (True for all)
    2. Check if interpretation satisfies all clauses
    3. If not, find a counterexample (derivation tree)
    4. Check feasibility of counterexample
    5. If spurious, use interpolation to refine interpretation
    6. Repeat
    """

    def __init__(self, system, max_iterations=20):
        self.system = system
        self.max_iterations = max_iterations
        self.stats = CHCStats(strategy="interp_cegar")
        self.interpolator = CraigInterpolator()

    def solve(self):
        """Solve using CEGAR with interpolation-based refinement."""
        # Start with True interpretation for all predicates
        interp = Interpretation(mapping={})
        for name in self.system.predicates:
            interp.set(name, BoolConst(True))

        for iteration in range(self.max_iterations):
            self.stats.iterations = iteration + 1

            # Check if current interpretation satisfies all clauses
            violation = self._find_violation(interp)
            if violation is None:
                # Check queries too
                query_violation = self._check_queries(interp)
                if query_violation is None:
                    return CHCOutput(result=CHCResult.SAT, interpretation=interp,
                                     stats=self.stats)
                violation = query_violation

            # We have a violated clause -- try to find a concrete counterexample
            cex = self._extract_counterexample(violation, interp)
            if cex is not None:
                # Check feasibility
                feasible = self._check_feasibility(cex)
                if feasible:
                    return CHCOutput(result=CHCResult.UNSAT, derivation=cex,
                                     stats=self.stats)

                # Spurious -- refine via interpolation
                new_formulas = self._refine_with_interpolation(cex, interp)
                if new_formulas:
                    for pred_name, formula in new_formulas.items():
                        old = interp.get(pred_name)
                        interp.set(pred_name, _and(old, formula))
                    self.stats.predicates_discovered += len(new_formulas)
                    continue

            # Try strengthening directly
            strengthened = self._try_strengthen(interp, violation)
            if strengthened:
                continue

            # Cannot make progress
            break

        return CHCOutput(result=CHCResult.UNKNOWN, stats=self.stats)

    def _find_violation(self, interp):
        """Find a clause that is violated by the current interpretation."""
        for clause in self.system.clauses:
            if clause.is_query:
                continue  # Handled separately
            if not self._clause_satisfied(clause, interp):
                return clause
        return None

    def _clause_satisfied(self, clause, interp):
        """Check if a clause is satisfied by the interpretation."""
        # Build: body_preds_instantiated AND constraint => head_instantiated
        # This should be valid (UNSAT when negated)
        body_parts = [clause.constraint]
        for bp in clause.body_preds:
            body_parts.append(instantiate_pred(interp, bp))

        body = _and(*body_parts)

        if clause.head is None:
            # Query: body should be UNSAT
            self.stats.smt_queries += 1
            result, _ = _smt_check(body)
            return result == SMTResult.UNSAT

        head = instantiate_pred(interp, clause.head)

        # Check body => head (body AND NOT(head) is UNSAT)
        self.stats.smt_queries += 1
        check_formula = _and(body, _not(head))
        result, _ = _smt_check(check_formula)
        return result == SMTResult.UNSAT

    def _check_queries(self, interp):
        """Check all query clauses. Return first violated query or None."""
        for query in self.system.get_queries():
            body_parts = [query.constraint]
            for bp in query.body_preds:
                body_parts.append(instantiate_pred(interp, bp))
            body = _and(*body_parts)
            self.stats.smt_queries += 1
            result, _ = _smt_check(body)
            if result == SMTResult.SAT:
                return query
        return None

    def _extract_counterexample(self, violated_clause, interp):
        """Extract a concrete counterexample from a violated clause."""
        # Build the violation formula
        body_parts = [violated_clause.constraint]
        for bp in violated_clause.body_preds:
            body_parts.append(instantiate_pred(interp, bp))
        body = _and(*body_parts)

        if violated_clause.head is not None:
            head = instantiate_pred(interp, violated_clause.head)
            formula = _and(body, _not(head))
        else:
            formula = body

        self.stats.smt_queries += 1
        result, model = _smt_check(formula)
        if result == SMTResult.SAT:
            return Derivation(clause=violated_clause, children=[], model=model)
        return None

    def _check_feasibility(self, derivation):
        """Check if a counterexample derivation is feasible."""
        # For a leaf derivation, check the constraint alone
        clause = derivation.clause
        self.stats.smt_queries += 1
        result, _ = _smt_check(clause.constraint)
        return result == SMTResult.SAT

    def _refine_with_interpolation(self, cex, interp):
        """
        Use Craig interpolation to refine predicate interpretations.
        Given a spurious counterexample, find new constraints to add.
        """
        clause = cex.clause
        if clause.is_query and clause.body_preds:
            # Query violation: need to strengthen body predicates
            # A = body predicate interpretations, B = query constraint
            # If A AND B is UNSAT, interpolant strengthens A
            bp = clause.body_preds[0]
            pred_name = bp.predicate.name
            current_interp = interp.get(pred_name)
            a_formula = instantiate_pred(interp, bp)
            b_formula = clause.constraint

            self.stats.smt_queries += 1
            check_result, _ = _smt_check(_and(a_formula, b_formula))
            if check_result == SMTResult.UNSAT:
                # Can interpolate
                self.stats.interpolations += 1
                itp_result = self.interpolator.interpolate(a_formula, b_formula)
                if itp_result.success and itp_result.interpolant is not None:
                    # Map back to predicate parameters
                    new_constraint = itp_result.interpolant
                    return {pred_name: new_constraint}

        elif not clause.is_query and clause.body_preds:
            # Rule violation: body predicates too weak
            bp = clause.body_preds[0]
            pred_name = bp.predicate.name

            # A = constraint AND body, B = NOT(head)
            a_formula = _and(clause.constraint, instantiate_pred(interp, bp))
            head_formula = instantiate_pred(interp, clause.head)
            b_formula = _not(head_formula)

            self.stats.smt_queries += 1
            check_result, _ = _smt_check(_and(a_formula, b_formula))
            if check_result == SMTResult.UNSAT:
                self.stats.interpolations += 1
                itp_result = self.interpolator.interpolate(a_formula, b_formula)
                if itp_result.success and itp_result.interpolant is not None:
                    head_pred_name = clause.head.predicate.name
                    return {head_pred_name: itp_result.interpolant}

        return {}

    def _try_strengthen(self, interp, violated_clause):
        """Try to strengthen interpretation by analyzing the violated clause."""
        if violated_clause.is_query:
            # Need to make body predicates exclude the query constraint
            for bp in violated_clause.body_preds:
                pred_name = bp.predicate.name
                # Add NOT(query_constraint) to the predicate
                binding = {}
                for (pname, _), arg in zip(bp.predicate.params, bp.args):
                    if isinstance(arg, Var):
                        binding[arg.name] = Var(pname, INT)
                neg_constraint = _not(_substitute(violated_clause.constraint, binding))
                old = interp.get(pred_name)
                interp.set(pred_name, _and(old, neg_constraint))
                return True

        elif violated_clause.head is not None and not violated_clause.body_preds:
            # Fact violation: head interpretation is too restrictive
            # Weaken by joining with fact constraint
            head_name = violated_clause.head.predicate.name
            binding = {}
            for (pname, _), arg in zip(violated_clause.head.predicate.params,
                                        violated_clause.head.args):
                if isinstance(arg, Var):
                    binding[arg.name] = Var(pname, INT)
            fact_formula = _substitute(violated_clause.constraint, binding)
            old = interp.get(head_name)
            interp.set(head_name, _or(old, fact_formula))
            return True

        return False


# --- CHC from Transition System (convenience) ---

def chc_from_ts(ts):
    """
    Convert a V002 TransitionSystem to a CHC system.

    TS(Init, Trans, Prop) becomes:
      Init(x) => Inv(x)                     [fact]
      Inv(x) AND Trans(x,x') => Inv(x')     [rule]
      Inv(x) AND NOT(Prop(x)) => false       [query]
    """
    system = CHCSystem()

    params = [(name, sort) for name, sort in ts.state_vars]
    inv = system.add_predicate("Inv", params)

    # Fact: Init => Inv
    args = [Var(name, sort) for name, sort in params]
    system.add_fact(apply_pred(inv, args), ts.init_formula)

    # Rule: Inv(x) AND Trans(x,x') => Inv(x')
    primed_args = [Var(f"{name}'", sort) for name, sort in params]
    body_app = apply_pred(inv, args)
    head_app = apply_pred(inv, primed_args)
    system.add_clause(head=head_app, body_preds=[body_app], constraint=ts.trans_formula)

    # Query: Inv(x) AND NOT(Prop(x)) => false
    system.add_query([apply_pred(inv, args)], _not(ts.prop_formula))

    return system


def chc_from_loop(init_constraint, trans_constraint, bad_constraint, var_names):
    """
    Build a CHC system from loop components.

    init_constraint: formula over var_names (initial states)
    trans_constraint: formula over var_names and primed var_names (transition)
    bad_constraint: formula over var_names (bad states)
    var_names: list of variable name strings
    """
    system = CHCSystem()

    params = [(name, INT) for name in var_names]
    inv = system.add_predicate("Inv", params)

    args = [Var(name, INT) for name in var_names]
    primed_args = [Var(f"{name}'", INT) for name in var_names]

    system.add_fact(apply_pred(inv, args), init_constraint)
    system.add_clause(
        head=apply_pred(inv, primed_args),
        body_preds=[apply_pred(inv, args)],
        constraint=trans_constraint
    )
    system.add_query([apply_pred(inv, args)], bad_constraint)

    return system


# --- Unified solving API ---

def solve_chc(system, strategy="auto", **kwargs):
    """
    Solve a CHC system using the specified strategy.

    strategy: "auto", "pdr", "bmc", "interp_cegar"
    Returns: CHCOutput
    """
    if strategy == "auto":
        # Try PDR first for linear systems, then CEGAR
        if system.is_linear:
            result = PDRCHCSolver(system, **kwargs).solve()
            if result.result != CHCResult.UNKNOWN:
                return result
        result = InterpCHCSolver(system, **kwargs).solve()
        if result.result != CHCResult.UNKNOWN:
            return result
        return BMCSolver(system, **kwargs).solve()

    elif strategy == "pdr":
        return PDRCHCSolver(system, **kwargs).solve()

    elif strategy == "bmc":
        return BMCSolver(system, **kwargs).solve()

    elif strategy == "interp_cegar":
        return InterpCHCSolver(system, **kwargs).solve()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def verify_safety(init, trans, prop, var_names, strategy="auto"):
    """
    Convenience: verify safety property for a loop.

    init: initial state formula
    trans: transition formula (over x and x')
    prop: property formula (should hold in all reachable states)
    var_names: list of variable name strings

    Returns: CHCOutput
    """
    system = chc_from_loop(init, trans, _not(prop), var_names)
    return solve_chc(system, strategy=strategy)


def compare_strategies(system):
    """Run all strategies on a CHC system and compare results."""
    results = {}
    for strategy in ["pdr", "interp_cegar", "bmc"]:
        try:
            output = solve_chc(system, strategy=strategy)
            results[strategy] = {
                "result": output.result.value,
                "iterations": output.stats.iterations,
                "smt_queries": output.stats.smt_queries,
                "interpolations": output.stats.interpolations,
            }
        except Exception as e:
            results[strategy] = {"result": "error", "error": str(e)}
    return results


def chc_summary():
    """Summary of CHC solver capabilities."""
    return {
        "name": "V109: Constrained Horn Clause Solver",
        "strategies": ["pdr", "interp_cegar", "bmc", "auto"],
        "features": [
            "Linear CHC to transition system reduction",
            "Multi-predicate CHC with phase encoding",
            "PDR-based solving (via V002)",
            "Interpolation-based CEGAR (via V107)",
            "BMC-based counterexample finding",
            "Automatic strategy selection",
            "CHC from transition system conversion",
            "CHC from loop components",
            "Counterexample derivation trees",
            "Predicate interpretation extraction",
        ],
        "composition": ["C037 (SMT solver)", "V002 (PDR/IC3)", "V107 (Craig interpolation)"],
    }
