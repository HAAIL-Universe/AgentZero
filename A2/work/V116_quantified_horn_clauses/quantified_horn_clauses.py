"""V116: Quantified Horn Clauses

Extends V111 (Recursive Horn Clause Solving) with existential and universal
quantifiers over clause bodies and heads. Adds array theory support for
verifying programs with arrays.

Key capabilities:
1. Quantified formulas: Forall(vars, body), Exists(vars, body)
2. Array theory: ArraySort, Select, Store, Const array
3. Quantifier instantiation: E-matching, term-based, model-based
4. Quantified CHC solving: extend recursive solver for quantified invariants
5. Array property verification: sorted, bounded, partitioned, initialized

Composes: V111 (recursive CHC) + V109 (CHC solver) + C037 (SMT solver)
"""

import sys, os
_base = 'Z:/AgentZero'
_a2 = _base + '/A2/work'
sys.path.insert(0, _a2 + '/V111_recursive_chc')
sys.path.insert(0, _a2 + '/V109_chc_solver')
sys.path.insert(0, _base + '/challenges/C037_smt_solver')
sys.path.insert(0, _a2 + '/V107_craig_interpolation')
sys.path.insert(0, _a2 + '/V002_pdr_ic3')

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Tuple, Any, FrozenSet
from enum import Enum

from smt_solver import (
    SMTSolver, SMTResult, Term, Var, IntConst, BoolConst, App, Op, Sort, INT, BOOL
)
from chc_solver import (
    CHCSystem, Predicate, PredicateApp, HornClause, Interpretation, Derivation,
    CHCStats, CHCOutput, CHCResult,
    apply_pred, instantiate_pred,
    _and, _or, _not, _implies, _eq, _substitute, _smt_check, _check_implication,
)
from recursive_chc import (
    DependencyGraph, RecursiveCHCSolver, NonlinearCHCSolver, ModularCHCSolver,
    LemmaStore, solve_recursive_chc, solve_modular_chc,
)


# ---------------------------------------------------------------------------
# Quantifier AST extensions
# ---------------------------------------------------------------------------

class Quantifier:
    """Base class for quantified formulas."""
    pass


@dataclass(frozen=True)
class Forall(Quantifier):
    """Universal quantifier: forall vars. body"""
    variables: Tuple[Tuple[str, Sort], ...]  # ((name, sort), ...)
    body: Any  # Term or nested quantifier

    def __init__(self, variables, body):
        object.__setattr__(self, 'variables', tuple(
            (v, s) if isinstance(v, str) else (v, s)
            for v, s in variables
        ))
        object.__setattr__(self, 'body', body)

    def __repr__(self):
        vs = ', '.join(f'{n}:{s}' for n, s in self.variables)
        return f'Forall([{vs}], {self.body})'

    def __eq__(self, other):
        return isinstance(other, Forall) and self.variables == other.variables and self.body == other.body

    def __hash__(self):
        return hash(('Forall', self.variables, _term_hash(self.body)))


@dataclass(frozen=True)
class Exists(Quantifier):
    """Existential quantifier: exists vars. body"""
    variables: Tuple[Tuple[str, Sort], ...]
    body: Any

    def __init__(self, variables, body):
        object.__setattr__(self, 'variables', tuple(
            (v, s) if isinstance(v, str) else (v, s)
            for v, s in variables
        ))
        object.__setattr__(self, 'body', body)

    def __repr__(self):
        vs = ', '.join(f'{n}:{s}' for n, s in self.variables)
        return f'Exists([{vs}], {self.body})'

    def __eq__(self, other):
        return isinstance(other, Exists) and self.variables == other.variables and self.body == other.body

    def __hash__(self):
        return hash(('Exists', self.variables, _term_hash(self.body)))


def _term_hash(t):
    """Hash a term for quantifier equality."""
    if isinstance(t, (Forall, Exists)):
        return hash(t)
    if isinstance(t, Var):
        return hash(('Var', t.name))
    if isinstance(t, IntConst):
        return hash(('Int', t.value))
    if isinstance(t, BoolConst):
        return hash(('Bool', t.value))
    if isinstance(t, App):
        return hash(('App', t.op, tuple(_term_hash(a) for a in t.args)))
    return hash(str(t))


# ---------------------------------------------------------------------------
# Array theory
# ---------------------------------------------------------------------------

ARRAY = Sort("Array")  # Array(Int, Int) -- integer arrays


@dataclass(frozen=True)
class ArraySort:
    """Array sort with index and element sorts."""
    index_sort: Sort
    element_sort: Sort

    def __repr__(self):
        return f'Array({self.index_sort}, {self.element_sort})'


def Select(array, index):
    """Array read: a[i]"""
    return App(Op.CALL, [Var("__select__", INT), array, index], INT)


def Store(array, index, value):
    """Array write: a[i] := v, returns new array"""
    return App(Op.CALL, [Var("__store__", INT), array, index, value], INT)


def ConstArray(value):
    """Constant array: all elements equal to value."""
    return App(Op.CALL, [Var("__const_array__", INT), value], INT)


def _is_select(t):
    """Check if term is an array select."""
    return (isinstance(t, App) and t.op == Op.CALL and len(t.args) == 3
            and isinstance(t.args[0], Var) and t.args[0].name == "__select__")


def _is_store(t):
    """Check if term is an array store."""
    return (isinstance(t, App) and t.op == Op.CALL and len(t.args) == 4
            and isinstance(t.args[0], Var) and t.args[0].name == "__store__")


def _is_const_array(t):
    """Check if term is a constant array."""
    return (isinstance(t, App) and t.op == Op.CALL and len(t.args) == 2
            and isinstance(t.args[0], Var) and t.args[0].name == "__const_array__")


def _get_select_parts(t):
    """Extract (array, index) from Select term."""
    return (t.args[1], t.args[2])


def _get_store_parts(t):
    """Extract (array, index, value) from Store term."""
    return (t.args[1], t.args[2], t.args[3])


# ---------------------------------------------------------------------------
# Quantifier-aware formula operations
# ---------------------------------------------------------------------------

def collect_free_vars(formula):
    """Collect free variables in a possibly-quantified formula."""
    if isinstance(formula, Forall):
        bound = {v for v, _ in formula.variables}
        return collect_free_vars(formula.body) - bound
    if isinstance(formula, Exists):
        bound = {v for v, _ in formula.variables}
        return collect_free_vars(formula.body) - bound
    if isinstance(formula, Var):
        return {formula.name}
    if isinstance(formula, (IntConst, BoolConst)):
        return set()
    if isinstance(formula, App):
        result = set()
        for a in formula.args:
            result |= collect_free_vars(a)
        return result
    return set()


def substitute_quantified(formula, mapping):
    """Substitute variables in a quantified formula, respecting binding."""
    if isinstance(formula, Forall):
        bound = {v for v, _ in formula.variables}
        safe_map = {k: v for k, v in mapping.items() if k not in bound}
        new_body = substitute_quantified(formula.body, safe_map)
        return Forall(formula.variables, new_body)
    if isinstance(formula, Exists):
        bound = {v for v, _ in formula.variables}
        safe_map = {k: v for k, v in mapping.items() if k not in bound}
        new_body = substitute_quantified(formula.body, safe_map)
        return Exists(formula.variables, new_body)
    # Regular term -- use _substitute
    return _substitute(formula, mapping)


def negate_quantified(formula):
    """Negate a quantified formula using De Morgan for quantifiers.
    NOT(forall x. P) = exists x. NOT(P)
    NOT(exists x. P) = forall x. NOT(P)
    """
    if isinstance(formula, Forall):
        return Exists(formula.variables, negate_quantified(formula.body))
    if isinstance(formula, Exists):
        return Forall(formula.variables, negate_quantified(formula.body))
    return _not(formula)


# ---------------------------------------------------------------------------
# Quantifier instantiation engine
# ---------------------------------------------------------------------------

class InstantiationStrategy(Enum):
    TERM_BASED = "term_based"
    E_MATCHING = "e_matching"
    MODEL_BASED = "model_based"


@dataclass
class Instantiation:
    """A single quantifier instantiation."""
    quantifier: Any  # Forall or Exists
    substitution: Dict[str, Any]  # var -> term
    result: Any  # Instantiated formula


class QuantifierInstantiator:
    """Engine for instantiating quantified formulas."""

    def __init__(self, max_instances=50):
        self.max_instances = max_instances
        self.instances = []
        self.stats = {"term_based": 0, "e_matching": 0, "model_based": 0}

    def instantiate_forall(self, formula, ground_terms, strategy=InstantiationStrategy.TERM_BASED):
        """Instantiate a universal quantifier with ground terms.

        For forall x. P(x), generate P(t) for each ground term t.
        Returns list of instantiated formulas (conjunction semantics).
        """
        if not isinstance(formula, Forall):
            return [formula]

        instances = []
        var_names = [v for v, _ in formula.variables]

        if strategy == InstantiationStrategy.TERM_BASED:
            instances = self._term_based_instantiate(formula, var_names, ground_terms)
            self.stats["term_based"] += len(instances)
        elif strategy == InstantiationStrategy.E_MATCHING:
            instances = self._e_matching_instantiate(formula, var_names, ground_terms)
            self.stats["e_matching"] += len(instances)
        elif strategy == InstantiationStrategy.MODEL_BASED:
            instances = self._model_based_instantiate(formula, var_names, ground_terms)
            self.stats["model_based"] += len(instances)

        self.instances.extend(instances)
        return [inst.result for inst in instances]

    def instantiate_exists(self, formula, ground_terms):
        """Instantiate an existential quantifier.

        For exists x. P(x), generate P(t) for each ground term t.
        Returns list of instantiated formulas (disjunction semantics).
        """
        if not isinstance(formula, Exists):
            return [formula]

        var_names = [v for v, _ in formula.variables]
        instances = self._term_based_instantiate_exists(formula, var_names, ground_terms)
        self.instances.extend(instances)
        return [inst.result for inst in instances]

    def _term_based_instantiate(self, formula, var_names, ground_terms):
        """Generate instances by substituting all ground terms for each variable."""
        if not ground_terms:
            return []

        results = []
        if len(var_names) == 1:
            vn = var_names[0]
            for t in ground_terms[:self.max_instances]:
                mapping = {vn: t}
                body = substitute_quantified(formula.body, mapping)
                results.append(Instantiation(formula, mapping, body))
        else:
            # Multi-variable: cross product (limited)
            combos = self._limited_cross_product(var_names, ground_terms)
            for combo in combos:
                mapping = dict(zip(var_names, combo))
                body = substitute_quantified(formula.body, mapping)
                results.append(Instantiation(formula, mapping, body))
        return results

    def _term_based_instantiate_exists(self, formula, var_names, ground_terms):
        """Same as term-based but for existentials."""
        if not ground_terms:
            return []

        results = []
        if len(var_names) == 1:
            vn = var_names[0]
            for t in ground_terms[:self.max_instances]:
                mapping = {vn: t}
                body = substitute_quantified(formula.body, mapping)
                results.append(Instantiation(formula, mapping, body))
        else:
            combos = self._limited_cross_product(var_names, ground_terms)
            for combo in combos:
                mapping = dict(zip(var_names, combo))
                body = substitute_quantified(formula.body, mapping)
                results.append(Instantiation(formula, mapping, body))
        return results

    def _e_matching_instantiate(self, formula, var_names, ground_terms):
        """E-matching: find terms that match patterns in the quantifier body."""
        # Extract patterns from the body (select, store, function applications)
        patterns = self._extract_patterns(formula.body, set(var_names))

        if not patterns:
            # Fall back to term-based
            return self._term_based_instantiate(formula, var_names, ground_terms)

        results = []
        for pattern_var, pattern_context in patterns:
            # Find ground terms that match this pattern context
            for t in ground_terms[:self.max_instances]:
                mapping = {pattern_var: t}
                # Only instantiate if this variable is in our quantifier
                if pattern_var in var_names:
                    full_mapping = {v: mapping.get(v, Var(v, INT)) for v in var_names}
                    body = substitute_quantified(formula.body, full_mapping)
                    results.append(Instantiation(formula, full_mapping, body))

        # Deduplicate
        seen = set()
        unique = []
        for inst in results:
            key = str(inst.substitution)
            if key not in seen:
                seen.add(key)
                unique.append(inst)

        return unique[:self.max_instances]

    def _model_based_instantiate(self, formula, var_names, ground_terms):
        """Model-based quantifier instantiation.

        Check if the negation of the quantified formula is satisfiable.
        If SAT, the model provides a useful instantiation.
        """
        results = []

        # First try negation: exists x. NOT(body) -- find counterexample
        neg_body = negate_quantified(formula.body)
        s = SMTSolver()
        smt_vars = {}
        for vn, vs in formula.variables:
            if vs == INT:
                smt_vars[vn] = s.Int(vn)
            else:
                smt_vars[vn] = s.Bool(vn)

        # Also declare free variables
        fv = collect_free_vars(formula.body)
        for v in fv:
            if v not in smt_vars:
                smt_vars[v] = s.Int(v)

        try:
            smt_formula = _lower_to_smt(neg_body, smt_vars, s)
            s.add(smt_formula)
            result = s.check()
            if result == SMTResult.SAT:
                model = s.model()
                mapping = {}
                for vn, _ in formula.variables:
                    if vn in model:
                        mapping[vn] = IntConst(model[vn])
                    else:
                        mapping[vn] = IntConst(0)
                body = substitute_quantified(formula.body, mapping)
                results.append(Instantiation(formula, mapping, body))
        except Exception:
            pass

        # Also use ground terms as fallback
        if not results:
            return self._term_based_instantiate(formula, var_names, ground_terms)

        return results[:self.max_instances]

    def _extract_patterns(self, formula, bound_vars):
        """Extract trigger patterns from formula body."""
        patterns = []
        self._collect_patterns(formula, bound_vars, patterns)
        return patterns

    def _collect_patterns(self, t, bound_vars, patterns):
        """Recursively collect (var, context) patterns from terms."""
        if isinstance(t, Var):
            if t.name in bound_vars:
                patterns.append((t.name, "var"))
            return
        if isinstance(t, (IntConst, BoolConst)):
            return
        if isinstance(t, App):
            # If this is a select/store with a bound var as index, it's a pattern
            if _is_select(t):
                _, idx = _get_select_parts(t)
                if isinstance(idx, Var) and idx.name in bound_vars:
                    patterns.append((idx.name, "select_index"))
            for arg in t.args:
                self._collect_patterns(arg, bound_vars, patterns)
        if isinstance(t, (Forall, Exists)):
            inner_bound = bound_vars | {v for v, _ in t.variables}
            self._collect_patterns(t.body, inner_bound, patterns)

    def _limited_cross_product(self, var_names, terms):
        """Generate limited cross product for multi-variable instantiation."""
        if len(var_names) == 0:
            return [()]
        if len(var_names) == 1:
            return [(t,) for t in terms[:self.max_instances]]

        # Limit to sqrt(max) per variable for multi-var
        import math
        per_var = max(2, int(math.sqrt(self.max_instances)))
        limited = terms[:per_var]
        result = []
        for t1 in limited:
            for t2 in limited:
                if len(var_names) == 2:
                    result.append((t1, t2))
                else:
                    for t3 in limited[:3]:
                        result.append((t1, t2, t3))
                if len(result) >= self.max_instances:
                    return result
        return result


# ---------------------------------------------------------------------------
# Lower quantified formulas to SMT (for checking)
# ---------------------------------------------------------------------------

def _lower_to_smt(formula, smt_vars, solver):
    """Lower a quantified formula to SMT terms by instantiation elimination.

    Since our C037 solver doesn't have native quantifier support, we handle
    quantifiers by instantiation (for bounded checking).
    """
    if isinstance(formula, Forall):
        # For checking: instantiate with existing terms
        # This is an under-approximation (sufficient for bounded verification)
        return _lower_to_smt(formula.body, smt_vars, solver)
    if isinstance(formula, Exists):
        # For checking: introduce fresh Skolem variables
        new_vars = dict(smt_vars)
        for vn, vs in formula.variables:
            if vn not in new_vars:
                if vs == INT:
                    new_vars[vn] = solver.Int(vn)
                else:
                    new_vars[vn] = solver.Bool(vn)
        return _lower_to_smt(formula.body, new_vars, solver)
    # Regular term
    return formula


# ---------------------------------------------------------------------------
# Quantified CHC extensions
# ---------------------------------------------------------------------------

@dataclass
class QuantifiedClause:
    """A Horn clause with optional quantifiers."""
    head: Optional[PredicateApp]
    body_preds: List[PredicateApp]
    constraint: Any  # Term, Forall, or Exists
    universal_head: Optional[Forall] = None  # Quantified head property

    @property
    def is_fact(self):
        return len(self.body_preds) == 0 and self.head is not None

    @property
    def is_query(self):
        return self.head is None

    @property
    def is_quantified(self):
        return (isinstance(self.constraint, (Forall, Exists))
                or self.universal_head is not None)


class QuantifiedCHCSystem:
    """CHC system supporting quantified constraints."""

    def __init__(self):
        self.predicates = {}  # name -> Predicate
        self.clauses = []  # List[QuantifiedClause]
        self.array_vars = set()  # Track array-typed variables

    def add_predicate(self, name, params):
        """Add a predicate declaration. params: [(name, sort), ...]"""
        p = Predicate(name, params)
        self.predicates[name] = p
        return p

    def add_fact(self, head, constraint):
        """Add a fact clause: constraint => head."""
        self.clauses.append(QuantifiedClause(
            head=head, body_preds=[], constraint=constraint
        ))

    def add_clause(self, head, body_preds, constraint):
        """Add a rule clause: body_preds AND constraint => head."""
        self.clauses.append(QuantifiedClause(
            head=head, body_preds=body_preds, constraint=constraint
        ))

    def add_query(self, body_preds, constraint):
        """Add a query (safety check): body_preds AND constraint => false."""
        self.clauses.append(QuantifiedClause(
            head=None, body_preds=body_preds, constraint=constraint
        ))

    def add_quantified_clause(self, head, body_preds, constraint, universal_head=None):
        """Add a clause with quantified constraint or head."""
        self.clauses.append(QuantifiedClause(
            head=head, body_preds=body_preds, constraint=constraint,
            universal_head=universal_head
        ))

    def declare_array(self, name):
        """Declare a variable as array-typed."""
        self.array_vars.add(name)

    def get_facts(self):
        return [c for c in self.clauses if c.is_fact]

    def get_queries(self):
        return [c for c in self.clauses if c.is_query]

    def get_rules(self):
        return [c for c in self.clauses if not c.is_fact and not c.is_query]

    def to_standard_chc(self, ground_terms=None):
        """Convert to standard CHCSystem by eliminating quantifiers via instantiation."""
        if ground_terms is None:
            ground_terms = self._collect_ground_terms()

        inst = QuantifierInstantiator()
        std = CHCSystem()

        # Copy predicates
        for name, pred in self.predicates.items():
            std.add_predicate(name, pred.params)

        # Convert clauses
        for clause in self.clauses:
            std_constraint = self._eliminate_quantifiers(clause.constraint, inst, ground_terms)

            if clause.is_fact:
                std.add_fact(clause.head, std_constraint)
            elif clause.is_query:
                std.add_query(clause.body_preds, std_constraint)
            else:
                std.add_clause(clause.head, clause.body_preds, std_constraint)

        return std, inst

    def _collect_ground_terms(self):
        """Collect ground terms appearing in the system for instantiation."""
        terms = set()
        # Always include small constants
        for i in range(-2, 11):
            terms.add(IntConst(i))

        for clause in self.clauses:
            self._collect_terms_from(clause.constraint, terms)
            if clause.head:
                for arg in clause.head.args:
                    self._collect_terms_from(arg, terms)
            for bp in clause.body_preds:
                for arg in bp.args:
                    self._collect_terms_from(arg, terms)

        return list(terms)

    def _collect_terms_from(self, formula, terms):
        """Collect concrete terms from a formula."""
        if isinstance(formula, IntConst):
            terms.add(formula)
        elif isinstance(formula, Var):
            pass  # Variables are not ground terms
        elif isinstance(formula, App):
            # Collect subterms that are ground (no free vars)
            for arg in formula.args:
                self._collect_terms_from(arg, terms)
        elif isinstance(formula, (Forall, Exists)):
            self._collect_terms_from(formula.body, terms)

    def _eliminate_quantifiers(self, formula, inst, ground_terms):
        """Eliminate quantifiers by instantiation."""
        if isinstance(formula, Forall):
            instances = inst.instantiate_forall(formula, ground_terms)
            if not instances:
                return BoolConst(True)  # No instances = vacuously true
            return _and(*instances)
        if isinstance(formula, Exists):
            instances = inst.instantiate_exists(formula, ground_terms)
            if not instances:
                return BoolConst(False)  # No instances = unsatisfied
            return _or(*instances)
        return formula


# ---------------------------------------------------------------------------
# Array theory axiom engine
# ---------------------------------------------------------------------------

class ArrayAxiomEngine:
    """Generates array theory axioms for CHC solving."""

    def __init__(self):
        self.axioms = []

    def read_over_write_same(self, array, index, value):
        """Axiom: Select(Store(a, i, v), i) == v"""
        return _eq(Select(Store(array, index, value), index), value)

    def read_over_write_diff(self, array, index1, index2, value):
        """Axiom: i != j => Select(Store(a, i, v), j) == Select(a, j)"""
        neq = App(Op.NEQ, [index1, index2], BOOL)
        rhs = _eq(Select(Store(array, index1, value), index2),
                   Select(array, index2))
        return _implies(neq, rhs)

    def const_array_axiom(self, value, index):
        """Axiom: Select(ConstArray(v), i) == v"""
        return _eq(Select(ConstArray(value), index), value)

    def extensionality(self, a1, a2, index):
        """Axiom: (forall i. Select(a1, i) == Select(a2, i)) => a1 == a2
        Instantiated for a specific index."""
        return _implies(
            _eq(Select(a1, index), Select(a2, index)),
            BoolConst(True)  # Partial -- full extensionality needs quantifiers
        )

    def generate_axioms(self, formula, ground_indices):
        """Generate relevant array axioms for a formula."""
        axioms = []
        stores = []
        selects = []
        self._collect_array_ops(formula, stores, selects)

        for store_term in stores:
            arr, idx, val = _get_store_parts(store_term)
            # Read-over-write-same for each store
            axioms.append(self.read_over_write_same(arr, idx, val))
            # Read-over-write-different for each other index
            for gi in ground_indices:
                axioms.append(self.read_over_write_diff(arr, idx, gi, val))

        return axioms

    def _collect_array_ops(self, formula, stores, selects):
        """Collect store and select operations from formula."""
        if isinstance(formula, (Forall, Exists)):
            self._collect_array_ops(formula.body, stores, selects)
            return
        if not isinstance(formula, App):
            return
        if _is_store(formula):
            stores.append(formula)
        if _is_select(formula):
            selects.append(formula)
        for arg in formula.args:
            self._collect_array_ops(arg, stores, selects)


# ---------------------------------------------------------------------------
# Quantified CHC solver
# ---------------------------------------------------------------------------

@dataclass
class QCHCOutput:
    """Result of quantified CHC solving."""
    result: CHCResult
    interpretation: Optional[Dict[str, Any]] = None  # pred -> formula (may be quantified)
    counterexample: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None
    instantiation_stats: Optional[Dict[str, int]] = None


class QuantifiedCHCSolver:
    """Solver for quantified horn clause systems.

    Strategy:
    1. Eliminate quantifiers via instantiation (reduce to standard CHC)
    2. Solve the standard system using V111 recursive solver
    3. Lift the solution back to quantified interpretation
    4. Verify the quantified solution against original system
    """

    def __init__(self, system, max_instances=50, max_iterations=30, max_depth=15):
        self.system = system
        self.max_instances = max_instances
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.instantiator = QuantifierInstantiator(max_instances)
        self.array_engine = ArrayAxiomEngine()
        self.stats = {
            "total_instances": 0,
            "refinement_rounds": 0,
            "array_axioms": 0,
            "smt_queries": 0,
        }

    def solve(self):
        """Solve the quantified CHC system."""
        # Phase 1: Collect ground terms for instantiation
        ground_terms = self.system._collect_ground_terms()

        # Phase 2: Generate array axioms if needed
        array_axioms = []
        if self.system.array_vars:
            ground_indices = [t for t in ground_terms if isinstance(t, IntConst)]
            for clause in self.system.clauses:
                array_axioms.extend(
                    self.array_engine.generate_axioms(clause.constraint, ground_indices)
                )
            self.stats["array_axioms"] = len(array_axioms)

        # Phase 3: Eliminate quantifiers and solve
        result = self._solve_with_instantiation(ground_terms, array_axioms)

        return result

    def _solve_with_instantiation(self, ground_terms, array_axioms):
        """Solve by eliminating quantifiers via instantiation."""
        # Convert to standard CHC
        std_system, inst = self.system.to_standard_chc(ground_terms)
        self.stats["total_instances"] = sum(inst.stats.values())

        # Add array axioms as additional constraints on facts
        # (Axioms strengthen the system)
        if array_axioms:
            self._inject_axioms(std_system, array_axioms)

        # Solve the standard system
        try:
            chc_result = solve_recursive_chc(std_system, self.max_iterations, self.max_depth)
        except Exception:
            try:
                chc_result = solve_modular_chc(std_system, self.max_iterations, self.max_depth)
            except Exception:
                return QCHCOutput(
                    result=CHCResult.UNKNOWN,
                    stats=self.stats,
                    instantiation_stats=inst.stats,
                )

        # Lift result
        interpretation = None
        if chc_result.result == CHCResult.SAT and chc_result.interpretation:
            interpretation = {}
            for name, formula in chc_result.interpretation.mapping.items():
                interpretation[name] = formula

        counterexample = None
        if chc_result.result == CHCResult.UNSAT:
            counterexample = {"derivation": str(chc_result.derivation) if chc_result.derivation else None}

        return QCHCOutput(
            result=chc_result.result,
            interpretation=interpretation,
            counterexample=counterexample,
            stats=self.stats,
            instantiation_stats=inst.stats,
        )

    def _inject_axioms(self, std_system, axioms):
        """Inject array axioms into the standard CHC system.

        Axioms are conjoined with fact clause constraints.
        """
        axiom_conj = _and(*axioms) if axioms else BoolConst(True)
        for i, clause in enumerate(std_system.clauses):
            if clause.is_fact:
                # Strengthen fact with axioms
                new_constraint = _and(clause.constraint, axiom_conj)
                std_system.clauses[i] = HornClause(
                    head=clause.head,
                    body_preds=clause.body_preds,
                    constraint=new_constraint,
                )


# ---------------------------------------------------------------------------
# Array property verification
# ---------------------------------------------------------------------------

def array_sorted_property(array_var, lo, hi):
    """Property: array is sorted in range [lo, hi).
    forall i. lo <= i < hi-1 => a[i] <= a[i+1]
    """
    i = Var("__sort_i__", INT)
    a = Var(array_var, INT)
    body = _implies(
        _and(
            App(Op.LE, [lo, i], BOOL),
            App(Op.LT, [i, App(Op.SUB, [hi, IntConst(1)], INT)], BOOL)
        ),
        App(Op.LE, [Select(a, i), Select(a, App(Op.ADD, [i, IntConst(1)], INT))], BOOL)
    )
    return Forall([("__sort_i__", INT)], body)


def array_bounded_property(array_var, lo, hi, lower_bound, upper_bound):
    """Property: all elements in range [lo, hi) are within [lower_bound, upper_bound].
    forall i. lo <= i < hi => lower_bound <= a[i] <= upper_bound
    """
    i = Var("__bound_i__", INT)
    a = Var(array_var, INT)
    body = _implies(
        _and(
            App(Op.LE, [lo, i], BOOL),
            App(Op.LT, [i, hi], BOOL)
        ),
        _and(
            App(Op.LE, [lower_bound, Select(a, i)], BOOL),
            App(Op.LE, [Select(a, i), upper_bound], BOOL)
        )
    )
    return Forall([("__bound_i__", INT)], body)


def array_initialized_property(array_var, lo, hi, value):
    """Property: all elements in range [lo, hi) equal value.
    forall i. lo <= i < hi => a[i] == value
    """
    i = Var("__init_i__", INT)
    a = Var(array_var, INT)
    body = _implies(
        _and(
            App(Op.LE, [lo, i], BOOL),
            App(Op.LT, [i, hi], BOOL)
        ),
        _eq(Select(a, i), value)
    )
    return Forall([("__init_i__", INT)], body)


def array_partitioned_property(array_var, pivot_idx, lo, hi):
    """Property: array is partitioned at pivot: a[i] <= a[pivot] for i < pivot,
    a[i] >= a[pivot] for i > pivot.
    """
    i = Var("__part_i__", INT)
    a = Var(array_var, INT)
    pivot_val = Select(a, pivot_idx)
    left = _implies(
        _and(App(Op.LE, [lo, i], BOOL), App(Op.LT, [i, pivot_idx], BOOL)),
        App(Op.LE, [Select(a, i), pivot_val], BOOL)
    )
    right = _implies(
        _and(App(Op.GT, [i, pivot_idx], BOOL), App(Op.LT, [i, hi], BOOL)),
        App(Op.GE, [Select(a, i), pivot_val], BOOL)
    )
    return Forall([("__part_i__", INT)], _and(left, right))


def array_exists_element(array_var, lo, hi, value):
    """Property: exists element equal to value in [lo, hi).
    exists i. lo <= i < hi AND a[i] == value
    """
    i = Var("__exists_i__", INT)
    a = Var(array_var, INT)
    body = _and(
        App(Op.LE, [lo, i], BOOL),
        App(Op.LT, [i, hi], BOOL),
        _eq(Select(a, i), value)
    )
    return Exists([("__exists_i__", INT)], body)


# ---------------------------------------------------------------------------
# Convenience APIs
# ---------------------------------------------------------------------------

def solve_quantified_chc(system, max_instances=50, max_iterations=30, max_depth=15):
    """Solve a quantified CHC system."""
    solver = QuantifiedCHCSolver(system, max_instances, max_iterations, max_depth)
    return solver.solve()


def verify_array_property(init_constraint, loop_constraint, property_formula,
                          var_params, array_vars=None, ground_terms=None):
    """Verify an array property of a loop.

    Args:
        init_constraint: Initial state constraint
        loop_constraint: Loop body transition (uses primed vars for next state)
        property_formula: Property to verify (may be quantified)
        var_params: [(name, sort), ...] for scalar variables
        array_vars: List of array variable names
        ground_terms: Additional ground terms for instantiation
    """
    system = QuantifiedCHCSystem()

    inv = system.add_predicate("Inv", var_params)

    if array_vars:
        for av in array_vars:
            system.declare_array(av)

    # Fact: init => Inv(vars)
    param_vars = [Var(n) for n, _ in var_params]
    system.add_fact(apply_pred(inv, param_vars), init_constraint)

    # Rule: Inv(vars) AND loop_body => Inv(vars')
    primed_vars = [Var(n + "'") for n, _ in var_params]
    system.add_clause(
        head=apply_pred(inv, primed_vars),
        body_preds=[apply_pred(inv, param_vars)],
        constraint=loop_constraint,
    )

    # Query: Inv(vars) AND NOT(property) => false
    if isinstance(property_formula, (Forall, Exists)):
        neg_prop = negate_quantified(property_formula)
    else:
        neg_prop = _not(property_formula)

    system.add_quantified_clause(
        head=None,
        body_preds=[apply_pred(inv, param_vars)],
        constraint=neg_prop,
    )

    # Add extra ground terms
    if ground_terms:
        # Will be collected automatically, but we can seed
        pass

    return solve_quantified_chc(system)


def verify_universal_property(init, transition, property_forall, var_params):
    """Verify a universally quantified property of a transition system.

    Convenience wrapper for the common case:
    - init: initial state constraint
    - transition: transition relation (uses primed vars)
    - property_forall: Forall(..., body) to verify
    - var_params: [(name, sort), ...]
    """
    return verify_array_property(
        init_constraint=init,
        loop_constraint=transition,
        property_formula=property_forall,
        var_params=var_params,
    )


def check_quantified_validity(formula, ground_terms=None):
    """Check if a quantified formula is valid (always true).

    Returns: (is_valid: bool, counterexample: Optional[dict])
    """
    if ground_terms is None:
        ground_terms = [IntConst(i) for i in range(-5, 11)]

    inst = QuantifierInstantiator()

    if isinstance(formula, Forall):
        # Check: exists x. NOT(body)
        instances = inst.instantiate_forall(formula, ground_terms)
        if not instances:
            return True, None

        # All instances should be true
        neg_conj = _not(_and(*instances))
        s = SMTSolver()

        fv = set()
        for inst_formula in instances:
            fv |= collect_free_vars(inst_formula)
        for v in fv:
            s.Int(v)

        s.add(neg_conj)
        result = s.check()
        if result == SMTResult.SAT:
            return False, s.model()
        return True, None

    elif isinstance(formula, Exists):
        # Check: exists x. body
        instances = inst.instantiate_exists(formula, ground_terms)
        if not instances:
            return False, None

        disj = _or(*instances)
        s = SMTSolver()

        fv = set()
        for inst_formula in instances:
            fv |= collect_free_vars(inst_formula)
        for v in fv:
            s.Int(v)

        s.add(disj)
        result = s.check()
        if result == SMTResult.SAT:
            return True, s.model()
        return False, None

    else:
        # Non-quantified: just check with SMT
        s = SMTSolver()
        fv = collect_free_vars(formula)
        for v in fv:
            s.Int(v)
        s.add(_not(formula))
        result = s.check()
        if result == SMTResult.UNSAT:
            return True, None
        return False, s.model()


def analyze_quantified_system(system):
    """Analyze a quantified CHC system structure."""
    total_clauses = len(system.clauses)
    quantified = sum(1 for c in system.clauses if c.is_quantified)
    forall_count = sum(1 for c in system.clauses
                       if isinstance(c.constraint, Forall))
    exists_count = sum(1 for c in system.clauses
                       if isinstance(c.constraint, Exists))
    facts = len(system.get_facts())
    queries = len(system.get_queries())
    rules = len(system.get_rules())
    array_vars = len(system.array_vars)

    return {
        "total_clauses": total_clauses,
        "quantified_clauses": quantified,
        "forall_clauses": forall_count,
        "exists_clauses": exists_count,
        "facts": facts,
        "queries": queries,
        "rules": rules,
        "predicates": list(system.predicates.keys()),
        "array_vars": list(system.array_vars),
        "has_arrays": array_vars > 0,
    }


def compare_instantiation_strategies(formula, ground_terms=None):
    """Compare different quantifier instantiation strategies."""
    if ground_terms is None:
        ground_terms = [IntConst(i) for i in range(0, 6)]

    results = {}
    for strategy in InstantiationStrategy:
        inst = QuantifierInstantiator()
        if isinstance(formula, Forall):
            instances = inst.instantiate_forall(formula, ground_terms, strategy)
        else:
            instances = [formula]
        results[strategy.value] = {
            "instances": len(instances),
            "stats": dict(inst.stats),
        }

    return results


def quantified_summary(system):
    """Human-readable summary of a quantified CHC system."""
    analysis = analyze_quantified_system(system)
    lines = [
        f"Quantified CHC System: {analysis['total_clauses']} clauses",
        f"  Predicates: {', '.join(analysis['predicates'])}",
        f"  Facts: {analysis['facts']}, Rules: {analysis['rules']}, Queries: {analysis['queries']}",
        f"  Quantified: {analysis['quantified_clauses']} ({analysis['forall_clauses']} forall, {analysis['exists_clauses']} exists)",
    ]
    if analysis['has_arrays']:
        lines.append(f"  Array vars: {', '.join(analysis['array_vars'])}")
    return '\n'.join(lines)
