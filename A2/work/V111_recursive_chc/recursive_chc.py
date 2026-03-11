"""
V111: Recursive Horn Clause Solving

Extends V109 CHC solver with:
  1. Recursive predicate handling -- predicates appearing in head AND body
  2. Nonlinear clause solving -- clauses with multiple body predicates
  3. Lemma learning -- cache and reuse intermediate invariants
  4. Dependency graph analysis -- topological ordering, SCC detection
  5. Modular solving -- decompose multi-predicate systems by SCC

Composes: V109 (CHC solver), C037 (SMT solver), V107 (Craig interpolation), V002 (PDR)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set, FrozenSet
from enum import Enum
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V109_chc_solver'))
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
from chc_solver import (
    CHCSystem, Predicate, PredicateApp, HornClause, Interpretation,
    Derivation, CHCStats, CHCOutput, CHCResult,
    apply_pred, instantiate_pred,
    _and, _or, _not, _implies, _eq, _substitute, _smt_check, _check_implication,
    BMCSolver, PDRCHCSolver, InterpCHCSolver
)


# --- Dependency Graph ---

@dataclass
class DepNode:
    """Node in the predicate dependency graph."""
    name: str
    depends_on: Set[str] = field(default_factory=set)  # predicates this one uses
    used_by: Set[str] = field(default_factory=set)      # predicates that use this one
    is_recursive: bool = False  # appears in own body


@dataclass
class SCC:
    """Strongly connected component in the dependency graph."""
    predicates: List[str]
    is_recursive: bool = False  # any self-loops or mutual recursion

    @property
    def size(self):
        return len(self.predicates)


class DependencyGraph:
    """
    Analyzes predicate dependencies in a CHC system.
    Computes SCCs and topological order for modular solving.
    """

    def __init__(self, system):
        self.system = system
        self.nodes = {}  # name -> DepNode
        self._build()

    def _build(self):
        """Build dependency graph from clauses."""
        # Initialize nodes for all predicates
        for name in self.system.predicates:
            self.nodes[name] = DepNode(name=name)

        # Analyze clauses
        for clause in self.system.clauses:
            if clause.head is None:
                continue
            head_name = clause.head.predicate.name
            for bp in clause.body_preds:
                body_name = bp.predicate.name
                if body_name in self.nodes:
                    self.nodes[head_name].depends_on.add(body_name)
                    self.nodes[body_name].used_by.add(head_name)
                    if body_name == head_name:
                        self.nodes[head_name].is_recursive = True

    def get_sccs(self):
        """Compute SCCs using Tarjan's algorithm. Returns in reverse topological order."""
        index_counter = [0]
        stack = []
        on_stack = set()
        indices = {}
        lowlinks = {}
        result = []

        def strongconnect(v):
            indices[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in self.nodes[v].depends_on:
                if w not in self.nodes:
                    continue
                if w not in indices:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                scc = SCC(predicates=component)
                # SCC is recursive if it has >1 member or a self-loop
                if len(component) > 1:
                    scc.is_recursive = True
                elif component[0] in self.nodes and self.nodes[component[0]].is_recursive:
                    scc.is_recursive = True
                result.append(scc)

        for v in self.nodes:
            if v not in indices:
                strongconnect(v)

        return result  # reverse topological order (leaves first)

    def get_topological_order(self):
        """Return predicate names in topological order (dependencies first)."""
        sccs = self.get_sccs()
        order = []
        for scc in sccs:
            order.extend(scc.predicates)
        return order

    def get_recursive_predicates(self):
        """Return names of predicates that are recursive."""
        return {name for name, node in self.nodes.items() if node.is_recursive}

    def get_nonlinear_clauses(self):
        """Return clauses with multiple body predicates."""
        return [c for c in self.system.clauses if len(c.body_preds) > 1]


# --- Lemma Store ---

@dataclass
class Lemma:
    """A learned invariant for a predicate."""
    predicate_name: str
    formula: Term           # invariant formula over predicate parameters
    source: str             # how it was learned: "base", "inductive", "interpolation", "widening"
    strength: float = 1.0   # 1.0 = strongest (exact), 0.0 = weakest (True)


class LemmaStore:
    """Caches learned predicates across solving iterations."""

    def __init__(self):
        self.lemmas = defaultdict(list)  # pred_name -> [Lemma]
        self._interp_cache = {}          # pred_name -> current best interpretation

    def add_lemma(self, pred_name, formula, source="inductive"):
        """Add a learned lemma. Avoids duplicates."""
        # Check for duplicates (structural equality)
        for existing in self.lemmas[pred_name]:
            if str(existing.formula) == str(formula):
                return
        self.lemmas[pred_name].append(Lemma(
            predicate_name=pred_name,
            formula=formula,
            source=source
        ))

    def get_lemmas(self, pred_name):
        """Get all learned lemmas for a predicate."""
        return self.lemmas.get(pred_name, [])

    def get_conjunction(self, pred_name):
        """Get conjunction of all lemmas for a predicate."""
        lemmas = self.get_lemmas(pred_name)
        if not lemmas:
            return BoolConst(True)
        return _and(*[l.formula for l in lemmas])

    def update_interpretation(self, pred_name, formula):
        """Update the current best interpretation for a predicate."""
        self._interp_cache[pred_name] = formula

    def get_interpretation(self, pred_name):
        """Get current best interpretation."""
        return self._interp_cache.get(pred_name, BoolConst(True))

    def to_interpretation(self):
        """Convert cached interpretations to an Interpretation object."""
        return Interpretation(mapping=dict(self._interp_cache))

    def total_lemmas(self):
        return sum(len(v) for v in self.lemmas.values())


# --- Recursive CHC Solver ---

class RecursiveCHCSolver:
    """
    Solves CHC systems with recursive predicates via fixed-point iteration.

    Algorithm:
    1. Build dependency graph, compute SCCs
    2. Process SCCs in topological order (leaves first)
    3. For non-recursive SCCs: single-pass interpretation via BMC/PDR
    4. For recursive SCCs: iterative widening until fixed point
    5. For nonlinear clauses: product construction + decomposition
    """

    def __init__(self, system, max_iterations=30, max_depth=15):
        self.system = system
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.dep_graph = DependencyGraph(system)
        self.lemma_store = LemmaStore()
        self.stats = CHCStats(strategy="recursive")

    def solve(self):
        """Main solver entry point."""
        sccs = self.dep_graph.get_sccs()

        # Process each SCC bottom-up
        for scc in sccs:
            result = self._solve_scc(scc)
            if result is not None and result.result == CHCResult.UNSAT:
                # Found a real counterexample
                result.stats = self.stats
                return result

        # All SCCs solved -- check queries
        interp = self.lemma_store.to_interpretation()
        query_result = self._check_queries(interp)
        if query_result is not None:
            return query_result

        return CHCOutput(
            result=CHCResult.SAT,
            interpretation=interp,
            stats=self.stats
        )

    def _solve_scc(self, scc):
        """Solve a single SCC."""
        if not scc.is_recursive:
            return self._solve_non_recursive(scc)
        else:
            return self._solve_recursive_scc(scc)

    def _solve_non_recursive(self, scc):
        """
        Non-recursive SCC: compute interpretation in one pass.
        Use facts + already-solved dependencies to compute this predicate's meaning.
        """
        for pred_name in scc.predicates:
            interp = self._compute_base_interpretation(pred_name)
            self.lemma_store.update_interpretation(pred_name, interp)
            self.lemma_store.add_lemma(pred_name, interp, source="base")
        return None  # No counterexample at this level

    def _compute_base_interpretation(self, pred_name):
        """
        Compute interpretation for a non-recursive predicate from its fact clauses.
        Substitutes already-known interpretations for body predicates.
        """
        pred = self.system.predicates.get(pred_name)
        if pred is None:
            return BoolConst(True)

        disjuncts = []
        for clause in self.system.clauses:
            if clause.head is None or clause.head.predicate.name != pred_name:
                continue

            # Build the clause body with current interpretations
            body_formula = clause.constraint
            for bp in clause.body_preds:
                bp_interp = self.lemma_store.get_interpretation(bp.predicate.name)
                bp_inst = self._instantiate(bp, bp_interp)
                body_formula = _and(body_formula, bp_inst)

            # Map clause head args back to predicate params
            head_mapping = self._head_to_param_formula(clause, pred, body_formula)
            if head_mapping is not None:
                disjuncts.append(head_mapping)

        if not disjuncts:
            return BoolConst(False)  # No clauses define this predicate
        return _or(*disjuncts)

    def _head_to_param_formula(self, clause, pred, body_formula):
        """
        Given a clause body formula and head P(t1,...,tn), produce a formula
        over predicate's formal parameters by existentially quantifying
        intermediate variables.

        For simple cases (head args are variables), just rename.
        For complex cases, add equality constraints.
        """
        param_names = [p[0] for p in pred.params]
        head_args = clause.head.args

        # Build renaming: head_arg_name -> param_name
        eq_constraints = []
        for param_name, head_arg in zip(param_names, head_args):
            if isinstance(head_arg, Var):
                # Simple variable: add equality constraint
                eq_constraints.append(
                    App(Op.EQ, [Var(param_name, INT), head_arg], BOOL)
                )
            elif isinstance(head_arg, IntConst):
                eq_constraints.append(
                    App(Op.EQ, [Var(param_name, INT), head_arg], BOOL)
                )

        # Combine: exists intermediates. body AND head_arg == param
        combined = _and(body_formula, *eq_constraints)

        # Try to simplify by substitution for direct variable mappings
        result = self._try_simplify_exists(combined, param_names, clause, body_formula)
        return result

    def _try_simplify_exists(self, combined, param_names, clause, body_formula):
        """
        Try to eliminate existential variables by direct substitution.
        If head args are just variables, substitute directly.
        """
        pred = clause.head.predicate
        head_args = clause.head.args
        param_set = set(param_names)

        # Collect all variables in the body
        body_vars = self._collect_var_names(body_formula)
        for bp in clause.body_preds:
            for a in bp.args:
                body_vars.update(self._collect_var_names(a))

        # If all head args are distinct variables, just rename
        if all(isinstance(a, Var) for a in head_args):
            head_var_names = [a.name for a in head_args]
            if len(set(head_var_names)) == len(head_var_names):
                # Simple renaming: substitute body vars to param names
                mapping = {}
                for param_name, head_arg in zip(param_names, head_args):
                    mapping[head_arg.name] = Var(param_name, INT)
                return _substitute(body_formula, mapping)

        # Fall back to conjunction with equalities
        return combined

    def _collect_var_names(self, term):
        """Collect all variable names in a term."""
        names = set()
        if isinstance(term, Var):
            names.add(term.name)
        elif isinstance(term, App):
            for a in term.args:
                names.update(self._collect_var_names(a))
        return names

    def _instantiate(self, pred_app, interp_formula):
        """Instantiate a predicate interpretation with actual arguments."""
        pred = pred_app.predicate
        mapping = {}
        for (param_name, _sort), arg in zip(pred.params, pred_app.args):
            mapping[param_name] = arg
        return _substitute(interp_formula, mapping)

    def _solve_recursive_scc(self, scc):
        """
        Solve a recursive SCC using a multi-strategy approach:

        1. Try to reduce to a transition system and use PDR (best for linear single-pred)
        2. If PDR can't handle it, try bounded Kleene iteration
        3. Fall back to BMC for counterexample finding
        """
        pred_names = scc.predicates

        # Strategy 1: Try PDR reduction for single-predicate linear recursive SCCs
        if len(pred_names) == 1:
            pdr_result = self._try_pdr_for_recursive(pred_names[0])
            if pdr_result is not None:
                if pdr_result == "safe":
                    return None  # Safe, interpretations already set
                return pdr_result  # CHCOutput (UNSAT)

        # Strategy 2: Bounded Kleene iteration (under-approximation)
        # Compute reachable states layer by layer, check queries at each layer
        kleene_result = self._kleene_iteration(pred_names)
        if kleene_result is not None:
            return kleene_result

        # Strategy 3: Over-approximation -- start from True, narrow down
        over_result = self._over_approximation(pred_names)
        if over_result is not None:
            return over_result

        # Fallback: BMC
        bmc = BMCSolver(self.system, max_depth=self.max_depth)
        bmc_result = bmc.solve()
        if bmc_result.result == CHCResult.UNSAT:
            return bmc_result

        # Could not determine -- return SAT with True interpretation (sound over-approx)
        for pn in pred_names:
            self.lemma_store.update_interpretation(pn, BoolConst(True))
        return None

    def _try_pdr_for_recursive(self, pred_name):
        """
        Try to reduce a single recursive predicate to a transition system
        and solve with PDR. This handles the common case of loop invariant inference.
        """
        pred = self.system.predicates[pred_name]
        params = pred.params

        # Find fact clauses (init) and recursive clauses (transition)
        facts = []
        recursive_rules = []

        for clause in self.system.clauses:
            if clause.head is None:
                continue
            if clause.head.predicate.name != pred_name:
                continue
            if clause.is_fact:
                facts.append(clause)
            elif any(bp.predicate.name == pred_name for bp in clause.body_preds):
                recursive_rules.append(clause)

        if not facts or not recursive_rules:
            return None  # Not a standard loop pattern

        # Build TransitionSystem using its builder API
        ts = TransitionSystem()
        param_vars = {}
        for name, sort in params:
            param_vars[name] = ts.add_var(name, sort)

        # Build init: disjunction of fact constraints (mapped to param vars)
        init_disjuncts = []
        for fact in facts:
            mapped = self._head_to_param_formula(fact, pred, fact.constraint)
            if mapped is not None:
                init_disjuncts.append(mapped)

        if not init_disjuncts:
            return None
        ts.set_init(_or(*init_disjuncts))

        # Build transition: PDR uses x' (prime suffix) for next-state vars
        trans_disjuncts = []

        for rule in recursive_rules:
            body_pred = None
            for bp in rule.body_preds:
                if bp.predicate.name == pred_name:
                    body_pred = bp
                    break
            if body_pred is None:
                continue

            # Build mapping from body pred args to param names (unprimed)
            body_mapping = {}
            for (param_name, _sort), arg in zip(params, body_pred.args):
                if isinstance(arg, Var):
                    body_mapping[arg.name] = Var(param_name, INT)

            # Build mapping from head args to primed names (x')
            head_mapping = {}
            for (param_name, _sort), arg in zip(params, rule.head.args):
                if isinstance(arg, Var):
                    head_mapping[arg.name] = Var(param_name + "'", INT)

            combined_mapping = {**body_mapping, **head_mapping}
            trans_constraint = _substitute(rule.constraint, combined_mapping)

            # Add equalities for complex head args (e.g., x-1)
            for (param_name, _sort), head_arg in zip(params, rule.head.args):
                if not isinstance(head_arg, Var):
                    mapped_expr = _substitute(head_arg, body_mapping)
                    trans_constraint = _and(
                        trans_constraint,
                        App(Op.EQ, [Var(param_name + "'", INT), mapped_expr], BOOL)
                    )

            trans_disjuncts.append(trans_constraint)

        if not trans_disjuncts:
            return None
        ts.set_trans(_or(*trans_disjuncts))

        # Build property from queries
        queries = self.system.get_queries()
        prop_disjuncts = []
        for query in queries:
            for bp in query.body_preds:
                if bp.predicate.name == pred_name:
                    q_mapping = {}
                    for (param_name, _sort), arg in zip(params, bp.args):
                        if isinstance(arg, Var):
                            q_mapping[arg.name] = Var(param_name, INT)
                    q_constraint = _substitute(query.constraint, q_mapping)
                    prop_disjuncts.append(q_constraint)

        if not prop_disjuncts:
            return None

        ts.set_property(_not(_or(*prop_disjuncts)))

        try:
            pdr_output = check_ts(ts)
            self.stats.smt_queries += pdr_output.stats.smt_queries

            if pdr_output.result == PDRResult.SAFE:
                # Use the invariant from PDR if available
                if pdr_output.invariant:
                    inv_formula = _and(*pdr_output.invariant) if isinstance(pdr_output.invariant, list) else pdr_output.invariant
                    self.lemma_store.update_interpretation(pred_name, inv_formula)
                    self.lemma_store.add_lemma(pred_name, inv_formula, source="pdr")
                else:
                    self.lemma_store.update_interpretation(pred_name, BoolConst(True))
                return "safe"  # Signal: PDR proved safe
            elif pdr_output.result == PDRResult.UNSAFE:
                return CHCOutput(
                    result=CHCResult.UNSAT,
                    derivation=Derivation(clause=queries[0] if queries else None,
                                         children=[], model=None),
                    stats=self.stats
                )
        except Exception:
            pass  # PDR failed, try next strategy

        return None

    def _kleene_iteration(self, pred_names):
        """
        Bounded Kleene iteration: compute reachable states layer by layer.
        At each layer, check if queries are satisfiable.
        If queries stay UNSAT for all layers up to max_depth, return SAT.
        """
        # Initialize: layer 0 = facts only
        layers = {pn: [] for pn in pred_names}

        for pn in pred_names:
            self.lemma_store.update_interpretation(pn, BoolConst(False))

        # Compute fact interpretations first
        for pn in pred_names:
            fact_interp = self._compute_fact_interpretation(pn)
            if not (isinstance(fact_interp, BoolConst) and fact_interp.value is False):
                layers[pn].append(fact_interp)
                self.lemma_store.update_interpretation(pn, fact_interp)

        # Check queries with initial layer
        interp = self.lemma_store.to_interpretation()
        query_result = self._check_queries(interp)
        if query_result is not None:
            return query_result  # Counterexample found at init

        # Iterate: compute new layers
        for depth in range(1, self.max_depth + 1):
            self.stats.iterations += 1
            changed = False

            for pn in pred_names:
                new_layer = self._one_step_post(pn)
                if isinstance(new_layer, BoolConst) and new_layer.value is False:
                    continue

                # Combine with previous layers
                prev = self.lemma_store.get_interpretation(pn)
                combined = _or(prev, new_layer)

                # Check if we added anything new
                if not self._check_subsumption(new_layer, prev):
                    changed = True
                    self.lemma_store.update_interpretation(pn, combined)
                    layers[pn].append(new_layer)

            if not changed:
                # Fixed point reached -- all reachable states computed
                interp = self.lemma_store.to_interpretation()
                # Verify this is truly a fixed point
                if self._verify_interpretation(interp, pred_names):
                    return CHCOutput(
                        result=CHCResult.SAT,
                        interpretation=interp,
                        stats=self.stats
                    )
                break

            # Check queries with expanded interpretation
            interp = self.lemma_store.to_interpretation()
            query_result = self._check_queries(interp)
            if query_result is not None:
                return query_result  # Counterexample found at this depth

        return None  # Inconclusive

    def _compute_fact_interpretation(self, pred_name):
        """Compute interpretation from fact clauses only."""
        pred = self.system.predicates.get(pred_name)
        if pred is None:
            return BoolConst(False)

        disjuncts = []
        for clause in self.system.clauses:
            if not clause.is_fact:
                continue
            if clause.head.predicate.name != pred_name:
                continue
            mapped = self._head_to_param_formula(clause, pred, clause.constraint)
            if mapped is not None:
                disjuncts.append(mapped)

        if not disjuncts:
            return BoolConst(False)
        return _or(*disjuncts)

    def _over_approximation(self, pred_names):
        """
        Over-approximation approach: start from True, check if it satisfies
        all clauses. If yes, return SAT. This works when the system is simple
        enough that True is a valid invariant (property holds for all states).
        """
        # Set all predicates to True
        for pn in pred_names:
            self.lemma_store.update_interpretation(pn, BoolConst(True))

        interp = self.lemma_store.to_interpretation()

        # Check if True satisfies all clauses
        if self._verify_interpretation(interp, pred_names):
            # True works -- but does it satisfy queries?
            query_result = self._check_queries(interp)
            if query_result is None:
                return CHCOutput(
                    result=CHCResult.SAT,
                    interpretation=interp,
                    stats=self.stats
                )
            # True doesn't block queries -- need tighter invariant
            return query_result

        return None

    def _one_step_post(self, pred_name):
        """
        Compute one-step post-image for a predicate:
        The disjunction of all clause bodies that conclude pred_name,
        with current interpretations for body predicates.
        """
        pred = self.system.predicates.get(pred_name)
        if pred is None:
            return BoolConst(False)

        disjuncts = []
        for clause in self.system.clauses:
            if clause.head is None or clause.head.predicate.name != pred_name:
                continue

            # Build body formula
            body = clause.constraint
            feasible = True
            for bp in clause.body_preds:
                bp_interp = self.lemma_store.get_interpretation(bp.predicate.name)
                # If body pred is False and this is the only option, skip
                if isinstance(bp_interp, BoolConst) and bp_interp.value is False:
                    if bp.predicate.name != pred_name:
                        # Dependency not yet solved -- this clause is inactive
                        feasible = False
                        break
                bp_inst = self._instantiate(bp, bp_interp)
                body = _and(body, bp_inst)

            if not feasible:
                continue

            # Map to predicate parameters
            mapped = self._head_to_param_formula(clause, pred, body)
            if mapped is not None:
                disjuncts.append(mapped)

        if not disjuncts:
            return BoolConst(False)
        return _or(*disjuncts)

    def _check_subsumption(self, new_formula, old_formula):
        """Check if new => old (new is subsumed by old)."""
        # Special cases for efficiency
        if isinstance(old_formula, BoolConst) and old_formula.value is True:
            return True
        if isinstance(new_formula, BoolConst) and new_formula.value is False:
            return True
        if isinstance(old_formula, BoolConst) and old_formula.value is False:
            if isinstance(new_formula, BoolConst) and new_formula.value is False:
                return True
            return False
        # SMT check: new AND NOT(old) is UNSAT?
        self.stats.smt_queries += 1
        result, _ = _smt_check(_and(new_formula, _not(old_formula)))
        return result == SMTResult.UNSAT

    def _widen(self, old, new, iteration):
        """
        Widening operator for convergence.
        Strategy: after initial iterations, drop constraints not preserved.
        """
        if isinstance(old, BoolConst) and old.value is False:
            return new  # First iteration: just take new

        # Union: old OR new
        combined = _or(old, new)

        # After enough iterations, widen aggressively to True
        if iteration >= 5:
            # Check if the disjunction is getting complex
            if self._formula_size(combined) > 20:
                return BoolConst(True)

        return combined

    def _formula_size(self, term):
        """Estimate formula size."""
        if isinstance(term, (Var, IntConst, BoolConst)):
            return 1
        if isinstance(term, App):
            return 1 + sum(self._formula_size(a) for a in term.args)
        return 1

    def _verify_interpretation(self, interp, pred_names=None):
        """Verify that an interpretation satisfies all relevant clauses."""
        for clause in self.system.clauses:
            # Only check clauses involving our predicates
            if pred_names is not None:
                involved = False
                if clause.head is not None and clause.head.predicate.name in pred_names:
                    involved = True
                for bp in clause.body_preds:
                    if bp.predicate.name in pred_names:
                        involved = True
                if not involved:
                    continue

            if not self._check_clause(clause, interp):
                return False
        return True

    def _check_clause(self, clause, interp):
        """Check if a single clause is satisfied by the interpretation."""
        # Build body: conjunction of body pred interpretations + constraint
        body = clause.constraint
        for bp in clause.body_preds:
            bp_inst = instantiate_pred(interp, bp)
            body = _and(body, bp_inst)

        if clause.head is None:
            # Query: body should be UNSAT
            self.stats.smt_queries += 1
            result, _ = _smt_check(body)
            return result == SMTResult.UNSAT
        else:
            # Regular: body => head interpretation
            head_inst = instantiate_pred(interp, clause.head)
            self.stats.smt_queries += 1
            return _check_implication(body, head_inst)

    def _check_queries(self, interp):
        """Check all query clauses against the interpretation."""
        for clause in self.system.get_queries():
            body = clause.constraint
            for bp in clause.body_preds:
                bp_inst = instantiate_pred(interp, bp)
                body = _and(body, bp_inst)

            self.stats.smt_queries += 1
            result, model = _smt_check(body)
            if result == SMTResult.SAT:
                # Query is satisfiable -- property violation
                return CHCOutput(
                    result=CHCResult.UNSAT,
                    derivation=Derivation(clause=clause, children=[], model=model),
                    stats=self.stats
                )
        return None


# --- Nonlinear CHC Solver ---

class NonlinearCHCSolver:
    """
    Handles CHC systems with nonlinear clauses (multiple body predicates).

    Strategy: product construction.
    Given P(x) AND Q(y) AND phi(x,y) => R(z),
    create a product predicate PQ(x,y) with:
      - Fact: P_fact AND Q_fact => PQ(x,y)
      - Then: PQ(x,y) AND phi(x,y) => R(z)
    This linearizes the clause.
    """

    def __init__(self, system, max_iterations=30):
        self.system = system
        self.max_iterations = max_iterations
        self.stats = CHCStats(strategy="nonlinear")

    def solve(self):
        """Solve by linearizing nonlinear clauses then using RecursiveCHCSolver."""
        nonlinear = [c for c in self.system.clauses if len(c.body_preds) > 1]

        if not nonlinear:
            # Already linear -- delegate to recursive solver
            solver = RecursiveCHCSolver(self.system, max_iterations=self.max_iterations)
            return solver.solve()

        # Linearize
        linearized = self._linearize(nonlinear)

        # Solve linearized system
        solver = RecursiveCHCSolver(linearized, max_iterations=self.max_iterations)
        result = solver.solve()
        result.stats.strategy = "nonlinear"
        return result

    def _linearize(self, nonlinear_clauses):
        """Create a new CHC system where nonlinear clauses are linearized."""
        new_system = CHCSystem()

        # Copy all predicates
        for name, pred in self.system.predicates.items():
            new_system.add_predicate(name, pred.params)

        product_counter = 0

        for clause in self.system.clauses:
            if len(clause.body_preds) <= 1:
                # Linear clause -- copy directly
                new_system.clauses.append(clause)
                continue

            # Nonlinear clause: P1(x1) AND P2(x2) AND ... AND phi => H(y)
            # Create product predicate for pairs
            current_preds = list(clause.body_preds)

            while len(current_preds) > 1:
                # Take first two, create product
                bp1 = current_preds.pop(0)
                bp2 = current_preds.pop(0)

                # Product predicate params = union of both params
                p1_params = bp1.predicate.params
                p2_params = bp2.predicate.params

                # Rename to avoid clashes
                p2_renamed = [(f"_p2_{n}", s) for n, s in p2_params]
                product_params = list(p1_params) + list(p2_renamed)
                product_name = f"__product_{product_counter}"
                product_counter += 1

                product_pred = new_system.add_predicate(product_name, product_params)

                # Create product fact clauses from original predicates' definitions
                # P1(x1) AND P2(x2') => Product(x1, x2')
                rename_map = {}
                p2_args_renamed = []
                for (orig_name, sort), (new_name, _) in zip(p2_params, p2_renamed):
                    rename_map[orig_name] = Var(new_name, sort)
                    p2_args_renamed.append(Var(new_name, sort))

                product_args = list(bp1.args) + p2_args_renamed
                product_app = apply_pred(product_pred, product_args)

                bp2_renamed_app = PredicateApp(
                    predicate=bp2.predicate,
                    args=p2_args_renamed
                )

                new_system.add_clause(
                    head=product_app,
                    body_preds=[bp1, bp2_renamed_app],
                    constraint=BoolConst(True)
                )

                # Replace the two preds with the product
                current_preds.insert(0, PredicateApp(
                    predicate=product_pred,
                    args=product_args
                ))

            # Now current_preds has exactly one element
            new_system.add_clause(
                head=clause.head,
                body_preds=current_preds,
                constraint=clause.constraint
            )

        return new_system


# --- Modular CHC Solver ---

class ModularCHCSolver:
    """
    Decomposes a CHC system by SCC and solves each component
    using the most appropriate strategy.
    """

    def __init__(self, system, max_iterations=30, max_depth=15):
        self.system = system
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.stats = CHCStats(strategy="modular")
        self.lemma_store = LemmaStore()
        self.dep_graph = DependencyGraph(system)

    def solve(self):
        """Solve by decomposing into SCCs."""
        sccs = self.dep_graph.get_sccs()

        for scc in sccs:
            # Extract sub-system for this SCC
            sub_system = self._extract_subsystem(scc)

            # Choose strategy based on SCC properties
            result = self._solve_component(sub_system, scc)
            if result is not None and result.result == CHCResult.UNSAT:
                result.stats = self.stats
                return result

            # Transfer learned interpretations
            if result is not None and result.interpretation is not None:
                for pred_name, formula in result.interpretation.mapping.items():
                    self.lemma_store.update_interpretation(pred_name, formula)
                    self.lemma_store.add_lemma(pred_name, formula, source="modular")

        # Final query check
        interp = self.lemma_store.to_interpretation()

        for clause in self.system.get_queries():
            body = clause.constraint
            for bp in clause.body_preds:
                bp_inst = instantiate_pred(interp, bp)
                body = _and(body, bp_inst)

            self.stats.smt_queries += 1
            result, model = _smt_check(body)
            if result == SMTResult.SAT:
                return CHCOutput(
                    result=CHCResult.UNSAT,
                    derivation=Derivation(clause=clause, children=[], model=model),
                    stats=self.stats
                )

        return CHCOutput(
            result=CHCResult.SAT,
            interpretation=interp,
            stats=self.stats
        )

    def _extract_subsystem(self, scc):
        """Extract a sub-CHC-system for a given SCC."""
        pred_names = set(scc.predicates)
        sub = CHCSystem()

        # Add predicates in SCC
        for pn in pred_names:
            if pn in self.system.predicates:
                sub.add_predicate(pn, self.system.predicates[pn].params)

        # Add clauses relevant to this SCC
        for clause in self.system.clauses:
            if clause.head is not None and clause.head.predicate.name in pred_names:
                # Replace body predicates outside the SCC with their interpretations
                new_body_preds = []
                extra_constraints = clause.constraint
                for bp in clause.body_preds:
                    if bp.predicate.name in pred_names:
                        # Add predicate to sub-system if not already there
                        if bp.predicate.name not in sub.predicates:
                            sub.add_predicate(bp.predicate.name, bp.predicate.params)
                        new_body_preds.append(bp)
                    else:
                        # Replace with interpretation
                        bp_interp = self.lemma_store.get_interpretation(bp.predicate.name)
                        bp_inst = self._instantiate(bp, bp_interp)
                        extra_constraints = _and(extra_constraints, bp_inst)

                sub.add_clause(
                    head=clause.head,
                    body_preds=new_body_preds,
                    constraint=extra_constraints
                )

        # Add query clauses that only reference this SCC's predicates
        for clause in self.system.get_queries():
            all_in_scc = all(bp.predicate.name in pred_names for bp in clause.body_preds)
            if all_in_scc and clause.body_preds:
                # Check at least one pred is in this SCC
                if any(bp.predicate.name in pred_names for bp in clause.body_preds):
                    sub.clauses.append(clause)

        return sub

    def _instantiate(self, pred_app, interp_formula):
        """Instantiate a predicate interpretation with actual arguments."""
        pred = pred_app.predicate
        mapping = {}
        for (param_name, _sort), arg in zip(pred.params, pred_app.args):
            mapping[param_name] = arg
        return _substitute(interp_formula, mapping)

    def _solve_component(self, sub_system, scc):
        """Choose and run the best strategy for a component."""
        if not sub_system.clauses:
            return CHCOutput(result=CHCResult.SAT, interpretation=Interpretation(mapping={}))

        has_nonlinear = any(len(c.body_preds) > 1 for c in sub_system.clauses)

        if has_nonlinear:
            solver = NonlinearCHCSolver(sub_system, max_iterations=self.max_iterations)
        elif scc.is_recursive:
            solver = RecursiveCHCSolver(sub_system,
                                        max_iterations=self.max_iterations,
                                        max_depth=self.max_depth)
        else:
            # Simple: use V109's PDR solver for linear non-recursive
            if sub_system.is_linear and len(sub_system.predicates) <= 2:
                try:
                    solver = PDRCHCSolver(sub_system)
                    result = solver.solve()
                    self.stats.smt_queries += solver.stats.smt_queries
                    return result
                except Exception:
                    pass
            # Fallback to recursive solver (handles non-recursive as base case)
            solver = RecursiveCHCSolver(sub_system,
                                        max_iterations=self.max_iterations,
                                        max_depth=self.max_depth)

        result = solver.solve()
        self.stats.smt_queries += solver.stats.smt_queries
        self.stats.iterations += solver.stats.iterations
        return result


# --- Convenience APIs ---

def solve_recursive_chc(system, max_iterations=30, max_depth=15):
    """
    Solve a CHC system that may contain recursive predicates.
    Returns CHCOutput with result, interpretation, and stats.
    """
    solver = RecursiveCHCSolver(system, max_iterations=max_iterations,
                                 max_depth=max_depth)
    return solver.solve()


def solve_nonlinear_chc(system, max_iterations=30):
    """
    Solve a CHC system with nonlinear clauses.
    Linearizes via product construction, then solves recursively.
    """
    solver = NonlinearCHCSolver(system, max_iterations=max_iterations)
    return solver.solve()


def solve_modular_chc(system, max_iterations=30, max_depth=15):
    """
    Solve a CHC system modularly by decomposing into SCCs.
    Best general-purpose strategy for multi-predicate systems.
    """
    solver = ModularCHCSolver(system, max_iterations=max_iterations,
                               max_depth=max_depth)
    return solver.solve()


def analyze_dependencies(system):
    """Analyze the predicate dependency structure of a CHC system."""
    graph = DependencyGraph(system)
    sccs = graph.get_sccs()
    topo = graph.get_topological_order()
    recursive = graph.get_recursive_predicates()
    nonlinear = graph.get_nonlinear_clauses()

    return {
        'predicates': list(system.predicates.keys()),
        'clauses': len(system.clauses),
        'sccs': [(scc.predicates, scc.is_recursive) for scc in sccs],
        'topological_order': topo,
        'recursive_predicates': recursive,
        'nonlinear_clauses': len(nonlinear),
        'is_linear': system.is_linear,
    }


def compare_strategies(system, max_iterations=30, max_depth=15):
    """
    Compare recursive, modular, and V109 strategies on the same system.
    Returns a dict with results and stats from each approach.
    """
    results = {}

    # Recursive
    try:
        solver = RecursiveCHCSolver(system, max_iterations=max_iterations,
                                     max_depth=max_depth)
        output = solver.solve()
        results['recursive'] = {
            'result': output.result.value,
            'smt_queries': output.stats.smt_queries,
            'iterations': output.stats.iterations,
        }
    except Exception as e:
        results['recursive'] = {'result': 'error', 'error': str(e)}

    # Modular
    try:
        solver = ModularCHCSolver(system, max_iterations=max_iterations,
                                   max_depth=max_depth)
        output = solver.solve()
        results['modular'] = {
            'result': output.result.value,
            'smt_queries': output.stats.smt_queries,
            'iterations': output.stats.iterations,
        }
    except Exception as e:
        results['modular'] = {'result': 'error', 'error': str(e)}

    # V109 BMC
    try:
        solver = BMCSolver(system, max_depth=max_depth)
        output = solver.solve()
        results['bmc'] = {
            'result': output.result.value,
            'smt_queries': output.stats.smt_queries,
        }
    except Exception as e:
        results['bmc'] = {'result': 'error', 'error': str(e)}

    return results


def chc_from_recursive_loop(init_constraint, loop_body_constraint, property_constraint,
                             var_params):
    """
    Convenience: create a CHC system from a recursive loop.

    init_constraint: phi_init(x) -- initial condition
    loop_body_constraint: phi_body(x, x') -- loop body transition
    property_constraint: phi_prop(x) -- property to verify (negated for query)

    var_params: [(name, sort), ...] for the invariant predicate
    """
    system = CHCSystem()

    # Invariant predicate Inv(x)
    inv = system.add_predicate("Inv", var_params)
    var_names = [p[0] for p in var_params]
    inv_vars = [Var(n, s) for n, s in var_params]

    # Primed variables for post-state
    primed_params = [(f"{n}_prime", s) for n, s in var_params]
    inv_primed_vars = [Var(f"{n}_prime", s) for n, s in var_params]

    # Fact: init => Inv(x)
    system.add_fact(apply_pred(inv, inv_vars), init_constraint)

    # Rule: Inv(x) AND body(x, x') => Inv(x')
    system.add_clause(
        head=apply_pred(inv, inv_primed_vars),
        body_preds=[apply_pred(inv, inv_vars)],
        constraint=loop_body_constraint
    )

    # Query: Inv(x) AND NOT(property) => false
    system.add_query(
        body_preds=[apply_pred(inv, inv_vars)],
        constraint=_not(property_constraint)
    )

    return system


def chc_from_multi_phase(phases, transitions, property_constraint, var_params):
    """
    Create a CHC system for a multi-phase program.

    phases: list of (name, init_constraint) -- each phase has an init region
    transitions: list of (from_phase, to_phase, constraint) -- phase transitions
    property_constraint: checked at all phases
    var_params: [(name, sort), ...]
    """
    system = CHCSystem()
    preds = {}

    var_names = [p[0] for p in var_params]
    vars_list = [Var(n, s) for n, s in var_params]
    primed = [Var(f"{n}_prime", s) for n, s in var_params]

    # Create a predicate for each phase
    for phase_name, init in phases:
        pred = system.add_predicate(phase_name, var_params)
        preds[phase_name] = pred
        # Fact: init => Phase(x)
        system.add_fact(apply_pred(pred, vars_list), init)

    # Add transitions
    for from_name, to_name, constraint in transitions:
        system.add_clause(
            head=apply_pred(preds[to_name], primed),
            body_preds=[apply_pred(preds[from_name], vars_list)],
            constraint=constraint
        )

    # Query: any phase AND NOT(property) => false
    for phase_name in preds:
        system.add_query(
            body_preds=[apply_pred(preds[phase_name], vars_list)],
            constraint=_not(property_constraint)
        )

    return system


def recursive_chc_summary(system):
    """Get a summary of a CHC system's structure and solving results."""
    deps = analyze_dependencies(system)
    result = solve_modular_chc(system)

    return {
        'structure': deps,
        'result': result.result.value,
        'interpretation': str(result.interpretation) if result.interpretation else None,
        'stats': {
            'smt_queries': result.stats.smt_queries,
            'iterations': result.stats.iterations,
            'strategy': result.stats.strategy,
        }
    }
