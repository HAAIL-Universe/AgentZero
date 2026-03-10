"""
V010: Predicate Abstraction + CEGAR
====================================
Composes: C037 (SMT solver) + V002 (PDR/IC3) + C010 (parser)

Abstracts a concrete (integer/boolean) transition system into a boolean
system over predicates, model checks it with PDR, and refines predicates
when spurious counterexamples are found (CEGAR loop).

Pipeline:
  1. Start with initial predicates (user-supplied or auto-generated)
  2. Build abstract boolean transition system via Cartesian abstraction
  3. Model check abstract system with PDR
  4. If SAFE -> extract concrete invariant from abstract predicates
  5. If UNSAFE -> check counterexample feasibility in concrete system
  6. If spurious -> refine predicates (WP-based discovery)
  7. Repeat until convergence or iteration limit
"""

import sys, os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from smt_solver import SMTSolver, SMTResult, Var, IntConst, BoolConst, App, Op, Sort, SortKind, Term
from pdr import TransitionSystem, check_ts, PDRResult, PDROutput

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Predicate:
    """A named predicate over concrete program variables."""
    name: str
    formula: Term  # SMT formula over concrete state vars

    def __repr__(self):
        return f"Pred({self.name})"


class CEGARVerdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class CEGARStats:
    iterations: int = 0
    predicates_initial: int = 0
    predicates_final: int = 0
    spurious_traces: int = 0
    pdr_calls: int = 0
    smt_queries: int = 0
    abstract_states_checked: int = 0


@dataclass
class CEGARResult:
    verdict: CEGARVerdict
    invariant: Optional[list] = None       # Concrete invariant formulas (if SAFE)
    counterexample: Optional[list] = None  # Concrete trace (if UNSAFE)
    predicates: list = field(default_factory=list)  # Final predicate set
    stats: CEGARStats = field(default_factory=CEGARStats)


@dataclass
class ConcreteTS:
    """Concrete transition system with named variables and formulas."""
    int_vars: list = field(default_factory=list)    # list of var names
    bool_vars: list = field(default_factory=list)   # list of var names
    init_formula: Optional[Term] = None
    trans_formula: Optional[Term] = None
    prop_formula: Optional[Term] = None
    # Caches
    _var_objects: dict = field(default_factory=dict)
    _prime_objects: dict = field(default_factory=dict)

    def var(self, name):
        """Get or create a Var for a state variable."""
        if name not in self._var_objects:
            sort = BOOL if name in self.bool_vars else INT
            self._var_objects[name] = Var(name, sort)
        return self._var_objects[name]

    def prime(self, name):
        """Get or create a primed (next-state) Var."""
        pname = name + "'"
        if pname not in self._prime_objects:
            sort = BOOL if name in self.bool_vars else INT
            self._prime_objects[pname] = Var(pname, sort)
        return self._prime_objects[pname]

    def all_vars(self):
        return self.int_vars + self.bool_vars


# ---------------------------------------------------------------------------
# SMT helpers
# ---------------------------------------------------------------------------

def _and(*terms):
    """Conjunction, flattening and short-circuiting."""
    flat = []
    for t in terms:
        if isinstance(t, BoolConst):
            if not t.value:
                return BoolConst(False)
            continue
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
    """Disjunction, flattening and short-circuiting."""
    flat = []
    for t in terms:
        if isinstance(t, BoolConst):
            if t.value:
                return BoolConst(True)
            continue
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
    """Negate using complement operators."""
    if isinstance(t, BoolConst):
        return BoolConst(not t.value)
    if isinstance(t, App):
        comp = {Op.EQ: Op.NEQ, Op.NEQ: Op.EQ,
                Op.LT: Op.GE, Op.GE: Op.LT,
                Op.LE: Op.GT, Op.GT: Op.LE,
                Op.NOT: None}
        if t.op == Op.NOT:
            return t.args[0]
        if t.op in comp and comp[t.op] is not None:
            return App(comp[t.op], t.args, BOOL)
        if t.op == Op.AND:
            return _or(*[_not(a) for a in t.args])
        if t.op == Op.OR:
            return _and(*[_not(a) for a in t.args])
    return App(Op.NOT, [t], BOOL)


def _implies(a, b):
    return _or(_not(a), b)


def _ite(cond, then_val, else_val):
    """Build an ITE (if-then-else) term."""
    sort = then_val.sort if hasattr(then_val, 'sort') else INT
    return App(Op.ITE, [cond, then_val, else_val], sort)


def _eq(a, b):
    return App(Op.EQ, [a, b], BOOL)


def _substitute(term, mapping):
    """Substitute variables in term using mapping {name: replacement_term}."""
    if isinstance(term, Var):
        return mapping.get(term.name, term)
    if isinstance(term, (IntConst, BoolConst)):
        return term
    if isinstance(term, App):
        new_args = [_substitute(a, mapping) for a in term.args]
        return App(term.op, new_args, term.sort)
    return term


def _smt_check(formula):
    """Check satisfiability of a formula. Returns (SAT/UNSAT/UNKNOWN, model_or_None)."""
    s = SMTSolver()
    # Register all variables in the formula
    _register_vars(s, formula)
    s.add(formula)
    result = s.check()
    model = s.model() if result == SMTResult.SAT else None
    return result, model


def _register_vars(solver, term, registered=None):
    """Register all variables in a term with the solver."""
    if registered is None:
        registered = set()
    if isinstance(term, Var):
        if term.name not in registered:
            registered.add(term.name)
            if term.sort.kind == SortKind.BOOL:
                solver.Bool(term.name)
            else:
                solver.Int(term.name)
    elif isinstance(term, App):
        for a in term.args:
            _register_vars(solver, a, registered)


# ---------------------------------------------------------------------------
# Predicate abstraction
# ---------------------------------------------------------------------------

def _check_pred_in_state(pred_formula, state_formula):
    """Check if pred must be true/false/unknown given state_formula.

    Returns:
      True  - pred is always true in states satisfying state_formula
      False - pred is always false in states satisfying state_formula
      None  - pred can be either (unknown)
    """
    # Check: state_formula AND NOT(pred) is UNSAT => pred always true
    neg_check = _and(state_formula, _not(pred_formula))
    r1, _ = _smt_check(neg_check)
    if r1 == SMTResult.UNSAT:
        return True

    # Check: state_formula AND pred is UNSAT => pred always false
    pos_check = _and(state_formula, pred_formula)
    r2, _ = _smt_check(pos_check)
    if r2 == SMTResult.UNSAT:
        return False

    return None  # Unknown


def cartesian_abstraction(concrete_ts, predicates):
    """Build an abstract boolean transition system via Cartesian abstraction.

    For each predicate, independently determine:
    - Must it be true/false/unknown in the initial states?
    - For each (current_pred_value, next_pred_value) pair, is the transition feasible?

    Returns a V002 TransitionSystem over boolean variables b_0, b_1, ..., b_{k-1}.
    """
    k = len(predicates)
    abs_ts = TransitionSystem()

    # Create abstract integer variables (0/1 encoding for predicates).
    # We use INT vars instead of BOOL vars because V002's SMT solver
    # matches INT vars by name (in the LIA theory) but BOOL vars by
    # object identity (in the SAT solver). Using INT avoids identity issues.
    abs_vars = []
    domain_clauses = []
    for i, pred in enumerate(predicates):
        bv = abs_ts.add_int_var(f"b_{i}")
        abs_vars.append(bv)
        # Domain constraint: b_i in {0, 1}
        domain_clauses.append(App(Op.GE, [bv, IntConst(0)], BOOL))
        domain_clauses.append(App(Op.LE, [bv, IntConst(1)], BOOL))

    def _is_true(v):
        return _eq(v, IntConst(1))

    def _is_false(v):
        return _eq(v, IntConst(0))

    # --- Abstract init ---
    init_clauses = list(domain_clauses)
    for i, pred in enumerate(predicates):
        val = _check_pred_in_state(pred.formula, concrete_ts.init_formula)
        if val is True:
            init_clauses.append(_is_true(abs_vars[i]))
        elif val is False:
            init_clauses.append(_is_false(abs_vars[i]))
        # None -> no constraint (over-approximation)

    abs_init = _and(*init_clauses) if init_clauses else BoolConst(True)
    abs_ts.set_init(abs_init)

    # --- Abstract transition ---
    # For Cartesian abstraction, we check each predicate independently:
    # For predicate i, can b_i transition from val to val'?
    # We check: exists concrete s, s'. init_region(s) not needed here.
    #   pred_i(s) = val AND trans(s, s') AND pred_i(s') = val'
    # If feasible, allow this transition in the abstract system.
    #
    # For efficiency with PDR, we build the transition as a conjunction
    # of per-predicate constraints.

    trans_clauses = []
    for i, pred in enumerate(predicates):
        bi = abs_vars[i]
        bi_prime = abs_ts.prime(f"b_{i}")

        # Compute the primed version of the predicate formula
        prime_map = {v: concrete_ts.prime(v) for v in concrete_ts.all_vars()}
        pred_primed = _substitute(pred.formula, {v.name: prime_map[v.name] for v in _collect_vars(pred.formula) if v.name in [n for n in concrete_ts.all_vars()]})

        # Check which transitions are feasible
        feasible = _compute_pred_transitions(
            pred.formula, pred_primed,
            concrete_ts.trans_formula, concrete_ts
        )

        # Build abstract transition for this predicate
        # feasible is a set of (current_val, next_val) pairs
        pred_trans = _build_pred_transition_int(bi, bi_prime, feasible)
        if pred_trans is not None:
            trans_clauses.append(pred_trans)

    # Add domain constraints for primed vars
    for i in range(k):
        bip = abs_ts.prime(f"b_{i}")
        trans_clauses.append(App(Op.GE, [bip, IntConst(0)], BOOL))
        trans_clauses.append(App(Op.LE, [bip, IntConst(1)], BOOL))

    abs_trans = _and(*trans_clauses) if trans_clauses else BoolConst(True)
    abs_ts.set_trans(abs_trans)

    # --- Abstract property ---
    # The property must be expressible in terms of the predicates.
    # We find which predicates imply the property and build the abstract version.
    abs_prop = _abstract_property_int(concrete_ts, predicates, abs_vars)
    abs_ts.set_property(abs_prop)

    return abs_ts


def _collect_vars(term):
    """Collect all Var objects in a term."""
    result = set()
    if isinstance(term, Var):
        result.add(term)
    elif isinstance(term, App):
        for a in term.args:
            result.update(_collect_vars(a))
    return result


def _compute_pred_transitions(pred_curr, pred_next, trans_formula, cts):
    """Check which (curr_val, next_val) transitions are feasible for a predicate.

    Returns set of (bool, bool) pairs that are feasible.
    """
    feasible = set()
    for curr_val in [True, False]:
        for next_val in [True, False]:
            # Build: pred_curr_constraint AND trans AND pred_next_constraint
            curr_constraint = pred_curr if curr_val else _not(pred_curr)
            next_constraint = pred_next if next_val else _not(pred_next)
            check_formula = _and(curr_constraint, trans_formula, next_constraint)

            r, _ = _smt_check(check_formula)
            if r == SMTResult.SAT:
                feasible.add((curr_val, next_val))

    return feasible


def _build_pred_transition(bi, bi_prime, feasible):
    """Build a transition formula for one abstract predicate (BOOL vars)."""
    if len(feasible) == 4:
        return None
    if len(feasible) == 0:
        return BoolConst(False)
    disj = []
    for (cv, nv) in feasible:
        ci = bi if cv else _not(bi)
        ni = bi_prime if nv else _not(bi_prime)
        disj.append(_and(ci, ni))
    return _or(*disj)


def _build_pred_transition_int(bi, bi_prime, feasible):
    """Build a transition formula for one abstract predicate (INT 0/1 vars).

    Given feasible (curr, next) pairs, encode as a formula over bi and bi'
    where bi=1 means True, bi=0 means False.
    """
    if len(feasible) == 4:
        return None  # All transitions feasible -> no constraint

    if len(feasible) == 0:
        return BoolConst(False)  # No transitions feasible -> deadlock

    # Build disjunction of feasible transitions
    disj = []
    for (cv, nv) in feasible:
        ci = _eq(bi, IntConst(1)) if cv else _eq(bi, IntConst(0))
        ni = _eq(bi_prime, IntConst(1)) if nv else _eq(bi_prime, IntConst(0))
        disj.append(_and(ci, ni))

    return _or(*disj)


def _abstract_property(concrete_ts, predicates, abs_vars):
    """Abstract the concrete property into predicate space.

    For Cartesian over-approximation, the abstract property should be the
    WEAKEST expressible condition. We use:
    1. First, check if any single predicate is equivalent to the property
    2. Otherwise, find predicates REQUIRED by the property (property => pred)
       and use their conjunction. This is sound: any concrete state satisfying
       the property will satisfy the abstract property.
    3. Fallback: find the single best predicate that implies the property.
    """
    prop = concrete_ts.prop_formula

    # 1. Check for exact match: pred <=> property
    for i, pred in enumerate(predicates):
        check1 = _and(pred.formula, _not(prop))
        r1, _ = _smt_check(check1)
        if r1 != SMTResult.UNSAT:
            continue
        check2 = _and(prop, _not(pred.formula))
        r2, _ = _smt_check(check2)
        if r2 == SMTResult.UNSAT:
            return abs_vars[i]

    # 2. Find predicates required by the property: property => pred
    required = []
    for i, pred in enumerate(predicates):
        check = _and(prop, _not(pred.formula))
        r, _ = _smt_check(check)
        if r == SMTResult.UNSAT:
            required.append(i)

    if required:
        return _and(*[abs_vars[i] for i in required])

    # 3. Fallback: find a predicate that implies the property (strongest available)
    for i, pred in enumerate(predicates):
        check = _and(pred.formula, _not(prop))
        r, _ = _smt_check(check)
        if r == SMTResult.UNSAT:
            return abs_vars[i]

    # Last resort: property is True in abstract space (over-approximation)
    # This means we can't express the property precisely -- CEGAR will refine
    return BoolConst(True)


def _abstract_property_int(concrete_ts, predicates, abs_vars):
    """Abstract property for INT 0/1 encoding.

    Same logic as _abstract_property but uses b_i == 1 for True.
    """
    prop = concrete_ts.prop_formula

    # 1. Check for exact match: pred <=> property
    for i, pred in enumerate(predicates):
        check1 = _and(pred.formula, _not(prop))
        r1, _ = _smt_check(check1)
        if r1 != SMTResult.UNSAT:
            continue
        check2 = _and(prop, _not(pred.formula))
        r2, _ = _smt_check(check2)
        if r2 == SMTResult.UNSAT:
            return _eq(abs_vars[i], IntConst(1))

    # 2. Find predicates required by the property: property => pred
    required = []
    for i, pred in enumerate(predicates):
        check = _and(prop, _not(pred.formula))
        r, _ = _smt_check(check)
        if r == SMTResult.UNSAT:
            required.append(i)

    if required:
        return _and(*[_eq(abs_vars[i], IntConst(1)) for i in required])

    # 3. Fallback: find a predicate that implies the property
    for i, pred in enumerate(predicates):
        check = _and(pred.formula, _not(prop))
        r, _ = _smt_check(check)
        if r == SMTResult.UNSAT:
            return _eq(abs_vars[i], IntConst(1))

    return BoolConst(True)


# ---------------------------------------------------------------------------
# Counterexample analysis
# ---------------------------------------------------------------------------

def check_counterexample_feasibility(abstract_trace, concrete_ts, predicates):
    """Check if an abstract counterexample trace is feasible in the concrete system.

    abstract_trace: list of dicts {pred_name: bool_value} from PDR counterexample
    Returns: (is_feasible, infeasible_step_index, concrete_trace_or_None)
    """
    n = len(abstract_trace)
    if n == 0:
        return True, None, []

    all_vars = concrete_ts.all_vars()

    # Try to find concrete states matching the abstract trace
    s = SMTSolver()

    # Create concrete variables for each step
    step_vars = []
    for step in range(n):
        sv = {}
        for v in concrete_ts.int_vars:
            sv[v] = s.Int(f"{v}_{step}")
        for v in concrete_ts.bool_vars:
            sv[v] = s.Bool(f"{v}_{step}")
        step_vars.append(sv)

    # Add predicate constraints from abstract trace
    for step, abs_state in enumerate(abstract_trace):
        for i, pred in enumerate(predicates):
            pred_name = f"b_{i}"
            if pred_name in abs_state:
                val = abs_state[pred_name]
                # INT encoding: 1 = True, 0 = False
                is_true = val == 1 if isinstance(val, int) else bool(val)
                # Substitute step-specific variables
                var_map = {v: step_vars[step][v] for v in all_vars if v in step_vars[step]}
                pred_at_step = _substitute(pred.formula, var_map)
                if is_true:
                    s.add(pred_at_step)
                else:
                    s.add(_not(pred_at_step))

    # Add init constraint for step 0
    init_at_0 = _substitute(concrete_ts.init_formula,
                            {v: step_vars[0][v] for v in all_vars if v in step_vars[0]})
    s.add(init_at_0)

    # Add transition constraints between consecutive steps
    for step in range(n - 1):
        # Build mapping: current vars -> step_vars[step], primed vars -> step_vars[step+1]
        trans_map = {}
        for v in all_vars:
            if v in step_vars[step]:
                trans_map[v] = step_vars[step][v]
            pv = v + "'"
            if v in step_vars[step + 1]:
                trans_map[pv] = step_vars[step + 1][v]
        trans_at_step = _substitute(concrete_ts.trans_formula, trans_map)
        s.add(trans_at_step)

    # Add property violation at last step
    prop_at_last = _substitute(concrete_ts.prop_formula,
                               {v: step_vars[-1][v] for v in all_vars if v in step_vars[-1]})
    s.add(_not(prop_at_last))

    result = s.check()
    if result == SMTResult.SAT:
        # Feasible -- extract concrete trace
        model = s.model()
        concrete_trace = []
        for step in range(n):
            state = {}
            for v in all_vars:
                key = f"{v}_{step}"
                if key in model:
                    state[v] = model[key]
            concrete_trace.append(state)
        return True, None, concrete_trace

    # Infeasible -- find the earliest failing step via incremental checking
    infeasible_step = _find_infeasible_step(abstract_trace, concrete_ts, predicates)
    return False, infeasible_step, None


def _find_infeasible_step(abstract_trace, concrete_ts, predicates):
    """Find the earliest step where the abstract trace becomes infeasible.

    Uses incremental unrolling: check step 0 alone, then 0-1, then 0-1-2, etc.
    """
    n = len(abstract_trace)
    all_vars = concrete_ts.all_vars()

    for end_step in range(n):
        s = SMTSolver()
        step_vars = []
        for step in range(end_step + 1):
            sv = {}
            for v in concrete_ts.int_vars:
                sv[v] = s.Int(f"{v}_{step}")
            for v in concrete_ts.bool_vars:
                sv[v] = s.Bool(f"{v}_{step}")
            step_vars.append(sv)

        # Init at step 0
        init_at_0 = _substitute(concrete_ts.init_formula,
                                {v: step_vars[0][v] for v in all_vars if v in step_vars[0]})
        s.add(init_at_0)

        # Predicate constraints
        for step in range(end_step + 1):
            abs_state = abstract_trace[step]
            for i, pred in enumerate(predicates):
                pred_name = f"b_{i}"
                if pred_name in abs_state:
                    val = abs_state[pred_name]
                    is_true = val == 1 if isinstance(val, int) else bool(val)
                    var_map = {v: step_vars[step][v] for v in all_vars if v in step_vars[step]}
                    pred_at_step = _substitute(pred.formula, var_map)
                    if is_true:
                        s.add(pred_at_step)
                    else:
                        s.add(_not(pred_at_step))

        # Transitions
        for step in range(end_step):
            trans_map = {}
            for v in all_vars:
                if v in step_vars[step]:
                    trans_map[v] = step_vars[step][v]
                pv = v + "'"
                if v in step_vars[step + 1]:
                    trans_map[pv] = step_vars[step + 1][v]
            trans_at_step = _substitute(concrete_ts.trans_formula, trans_map)
            s.add(trans_at_step)

        result = s.check()
        if result == SMTResult.UNSAT:
            return end_step

    return n - 1  # Last step (shouldn't happen if overall is UNSAT)


# ---------------------------------------------------------------------------
# Predicate refinement
# ---------------------------------------------------------------------------

def refine_predicates(predicates, concrete_ts, infeasible_step, abstract_trace):
    """Discover new predicates to eliminate a spurious counterexample.

    Strategy: weakest-precondition-based refinement.
    At the infeasible step, compute what must hold for the transition to be
    feasible, and add predicates for conditions that distinguish feasible
    from infeasible paths.
    """
    new_preds = []
    all_vars = concrete_ts.all_vars()

    if infeasible_step == 0:
        # Init state is infeasible -- need predicates that distinguish init
        # The abstract trace says certain predicates hold at step 0, but no
        # concrete init state satisfies them all. We need to split the init region.
        abs_state = abstract_trace[0]
        for i, pred in enumerate(predicates):
            pred_name = f"b_{i}"
            if pred_name in abs_state:
                val = abs_state[pred_name]
                actual = _check_pred_in_state(pred.formula, concrete_ts.init_formula)
                if actual is not None and actual != val:
                    # This predicate has a definite value in init that contradicts the trace
                    # No new pred needed -- the abstraction should already capture this
                    pass

        # Try strengthening: find what the init formula implies that we don't track
        new_preds.extend(_discover_init_predicates(concrete_ts, predicates))
    else:
        # Transition step is infeasible
        # Pre-image refinement: compute what must hold before the transition
        # for the post-state predicates to hold
        new_preds.extend(_discover_transition_predicates(
            concrete_ts, predicates, abstract_trace, infeasible_step
        ))

    # Deduplicate: don't add predicates equivalent to existing ones
    existing_names = {p.name for p in predicates}
    unique_new = []
    for p in new_preds:
        if p.name not in existing_names:
            existing_names.add(p.name)
            unique_new.append(p)

    return unique_new


def _discover_init_predicates(concrete_ts, existing_preds):
    """Discover predicates that better characterize the initial states."""
    new_preds = []
    # For each pair of existing predicates, try their conjunction/disjunction
    for i, p1 in enumerate(existing_preds):
        for j, p2 in enumerate(existing_preds):
            if i >= j:
                continue
            conj = _and(p1.formula, p2.formula)
            name = f"{p1.name}_and_{p2.name}"
            val = _check_pred_in_state(conj, concrete_ts.init_formula)
            if val is not None:
                new_preds.append(Predicate(name, conj))
                break
        if new_preds:
            break
    return new_preds


def _discover_transition_predicates(concrete_ts, predicates, abstract_trace, infeasible_step):
    """Discover predicates from an infeasible transition.

    At step k, the abstract trace says:
      - predicates have values abs_state[k] in current state
      - predicates have values abs_state[k+1] in next state (if k+1 exists)
      - transition connects them

    But no concrete transition exists. We compute the weakest precondition
    of the next-state predicates through the transition relation, and add
    predicates that distinguish feasible from infeasible pre-states.
    """
    new_preds = []
    all_vars = concrete_ts.all_vars()

    # Get the post-state predicate constraints
    if infeasible_step < len(abstract_trace):
        post_state = abstract_trace[infeasible_step]
    else:
        post_state = abstract_trace[-1]

    # For each predicate in the post-state, compute its WP through the transition
    for i, pred in enumerate(predicates):
        pred_name = f"b_{i}"
        if pred_name not in post_state:
            continue

        val = post_state[pred_name]

        # WP of pred through transition: substitute primed vars with their
        # transition definitions
        # For functional transitions (x' = expr), WP(pred(x'), trans) = pred(expr)
        trans_map = _extract_functional_map(concrete_ts.trans_formula, all_vars)
        if trans_map:
            # Substitute primed vars in predicate formula
            wp_formula = pred.formula  # pred uses unprimed vars
            # We need to compute: given trans, what must hold in current state
            # for pred to hold in next state?
            # If trans says x' = f(x), then WP(pred(x'), trans) = pred(f(x)/x')
            # But pred is over unprimed vars. So WP = pred[x := f(x)] essentially.
            # Actually: WP of "pred holds after transition" = substitute the
            # transition's effect into the predicate.

            # The predicate is P(x). After transition x->f(x), we need P(f(x)).
            # So the WP is P(f(x)), which is pred.formula with x replaced by f(x).
            wp = _substitute(pred.formula, trans_map)
            wp_name = f"wp_{pred.name}_step{infeasible_step}"

            # Check this WP is actually new (not equivalent to existing preds)
            is_new = True
            for ep in predicates:
                # Quick structural check
                if str(wp) == str(ep.formula):
                    is_new = False
                    break
            if is_new:
                new_preds.append(Predicate(wp_name, wp))

    # If no WP-based preds found, try mid-point predicates
    if not new_preds:
        new_preds.extend(_discover_midpoint_predicates(concrete_ts, predicates, infeasible_step))

    return new_preds


def _extract_functional_map(trans_formula, all_vars):
    """Try to extract a functional transition map {var_name: expr} from trans_formula.

    Looks for conjuncts of the form x' == expr where expr is over unprimed vars.
    """
    conjuncts = _collect_conjuncts(trans_formula)
    fmap = {}
    for c in conjuncts:
        if isinstance(c, App) and c.op == Op.EQ and len(c.args) == 2:
            lhs, rhs = c.args
            if isinstance(lhs, Var) and lhs.name.endswith("'"):
                base = lhs.name[:-1]
                if base in all_vars:
                    fmap[base] = rhs
            elif isinstance(rhs, Var) and rhs.name.endswith("'"):
                base = rhs.name[:-1]
                if base in all_vars:
                    fmap[base] = lhs
    return fmap


def _collect_conjuncts(formula):
    """Break a formula into top-level conjuncts."""
    if isinstance(formula, App) and formula.op == Op.AND:
        result = []
        for a in formula.args:
            result.extend(_collect_conjuncts(a))
        return result
    return [formula]


def _discover_midpoint_predicates(concrete_ts, predicates, step):
    """Generate midpoint predicates as a fallback refinement strategy.

    Creates predicates from:
    - Arithmetic combinations of existing predicate terms
    - Boundary conditions from transitions
    """
    new_preds = []
    all_vars = concrete_ts.all_vars()

    # Strategy: for each variable in the transition, create boundary predicates
    conjuncts = _collect_conjuncts(concrete_ts.trans_formula)
    for c in conjuncts:
        if isinstance(c, App) and c.op == Op.EQ:
            lhs, rhs = c.args
            if isinstance(lhs, Var) and lhs.name.endswith("'"):
                base = lhs.name[:-1]
                # Create predicate: base >= 0, base <= 0, base == rhs_simplified
                for op, name_suffix in [(Op.GE, "ge0"), (Op.LE, "le0")]:
                    p_formula = App(op, [Var(base, INT), IntConst(0)], BOOL)
                    p_name = f"{base}_{name_suffix}"
                    # Don't duplicate
                    if not any(ep.name == p_name for ep in predicates):
                        new_preds.append(Predicate(p_name, p_formula))
                if new_preds:
                    break

    return new_preds


# ---------------------------------------------------------------------------
# CEGAR main loop
# ---------------------------------------------------------------------------

def cegar_check(concrete_ts, initial_predicates, max_iterations=10, max_pdr_frames=50):
    """Run CEGAR loop: abstract, model check, refine.

    Args:
        concrete_ts: ConcreteTS with int_vars, bool_vars, init/trans/prop formulas
        initial_predicates: list of Predicate objects to start with
        max_iterations: max CEGAR refinement iterations
        max_pdr_frames: max frames for each PDR call

    Returns:
        CEGARResult with verdict, invariant, counterexample, final predicates, stats
    """
    predicates = list(initial_predicates)
    stats = CEGARStats(predicates_initial=len(predicates))

    for iteration in range(max_iterations):
        stats.iterations = iteration + 1

        # 1. Build abstract system
        abs_ts = cartesian_abstraction(concrete_ts, predicates)

        # 2. Model check with PDR
        stats.pdr_calls += 1
        pdr_result = check_ts(abs_ts, max_frames=max_pdr_frames)

        if pdr_result.result == PDRResult.SAFE:
            # Property holds in abstraction => holds in concrete (over-approximation)
            # Extract concrete invariant from abstract one
            invariant = _concretize_invariant(pdr_result.invariant, predicates)
            stats.predicates_final = len(predicates)
            return CEGARResult(
                verdict=CEGARVerdict.SAFE,
                invariant=invariant,
                predicates=predicates,
                stats=stats
            )

        if pdr_result.result == PDRResult.UNKNOWN:
            stats.predicates_final = len(predicates)
            return CEGARResult(
                verdict=CEGARVerdict.UNKNOWN,
                predicates=predicates,
                stats=stats
            )

        # 3. UNSAFE -- check counterexample feasibility
        assert pdr_result.result == PDRResult.UNSAFE
        abstract_trace = pdr_result.counterexample.trace

        is_feasible, infeasible_step, concrete_trace = check_counterexample_feasibility(
            abstract_trace, concrete_ts, predicates
        )

        if is_feasible:
            # Real counterexample
            stats.predicates_final = len(predicates)
            return CEGARResult(
                verdict=CEGARVerdict.UNSAFE,
                counterexample=concrete_trace,
                predicates=predicates,
                stats=stats
            )

        # 4. Spurious -- refine predicates
        stats.spurious_traces += 1
        new_preds = refine_predicates(predicates, concrete_ts, infeasible_step, abstract_trace)

        if not new_preds:
            # Can't refine further
            stats.predicates_final = len(predicates)
            return CEGARResult(
                verdict=CEGARVerdict.UNKNOWN,
                predicates=predicates,
                stats=stats
            )

        predicates.extend(new_preds)

    # Max iterations reached
    stats.predicates_final = len(predicates)
    return CEGARResult(
        verdict=CEGARVerdict.UNKNOWN,
        predicates=predicates,
        stats=stats
    )


def _concretize_invariant(abstract_invariant, predicates):
    """Convert abstract invariant (over b_i vars) to concrete formulas.

    The abstract invariant is a list of clauses over boolean variables b_0, b_1, ...
    We substitute each b_i with its corresponding predicate formula.
    """
    if abstract_invariant is None:
        return None

    concrete_clauses = []
    for clause in abstract_invariant:
        pred_map = {f"b_{i}": pred.formula for i, pred in enumerate(predicates)}
        concrete = _substitute_preds(clause, pred_map)
        concrete_clauses.append(concrete)

    return concrete_clauses


def _substitute_preds(term, pred_map):
    """Substitute abstract predicate variables with predicate formulas.

    Handles INT 0/1 encoding: b_i == 1 -> pred_formula, b_i == 0 -> NOT(pred_formula)
    Also handles bare Var references for backward compatibility.
    """
    if isinstance(term, Var) and term.name in pred_map:
        return pred_map[term.name]
    if isinstance(term, (IntConst, BoolConst)):
        return term
    if isinstance(term, Var):
        return term
    if isinstance(term, App):
        # Check for b_i == 1 or b_i == 0 pattern
        if term.op == Op.EQ and len(term.args) == 2:
            lhs, rhs = term.args
            if isinstance(lhs, Var) and lhs.name in pred_map and isinstance(rhs, IntConst):
                if rhs.value == 1:
                    return pred_map[lhs.name]
                elif rhs.value == 0:
                    return _not(pred_map[lhs.name])
            if isinstance(rhs, Var) and rhs.name in pred_map and isinstance(lhs, IntConst):
                if lhs.value == 1:
                    return pred_map[rhs.name]
                elif lhs.value == 0:
                    return _not(pred_map[rhs.name])
        # Check for b_i != 0 (equiv to b_i == 1)
        if term.op == Op.NEQ and len(term.args) == 2:
            lhs, rhs = term.args
            if isinstance(lhs, Var) and lhs.name in pred_map and isinstance(rhs, IntConst):
                if rhs.value == 0:
                    return pred_map[lhs.name]
                elif rhs.value == 1:
                    return _not(pred_map[lhs.name])
        # Check for b_i >= 1 (equiv to b_i == 1 with domain)
        if term.op == Op.GE and len(term.args) == 2:
            lhs, rhs = term.args
            if isinstance(lhs, Var) and lhs.name in pred_map and isinstance(rhs, IntConst) and rhs.value == 1:
                return pred_map[lhs.name]
        # Check for b_i <= 0 (equiv to b_i == 0 with domain)
        if term.op == Op.LE and len(term.args) == 2:
            lhs, rhs = term.args
            if isinstance(lhs, Var) and lhs.name in pred_map and isinstance(rhs, IntConst) and rhs.value == 0:
                return _not(pred_map[lhs.name])

        new_args = [_substitute_preds(a, pred_map) for a in term.args]
        return App(term.op, new_args, term.sort)
    return term


# ---------------------------------------------------------------------------
# High-level API: source-level CEGAR
# ---------------------------------------------------------------------------

def auto_predicates_from_ts(concrete_ts):
    """Auto-generate initial predicates from a transition system.

    Extracts predicates from:
    - Init formula conjuncts (equalities, inequalities)
    - Property formula
    - Transition relation boundary conditions
    """
    predicates = []
    seen_names = set()
    seen_formulas = set()

    def add_pred(name, formula):
        fstr = str(formula)
        if name not in seen_names and fstr not in seen_formulas:
            seen_names.add(name)
            seen_formulas.add(fstr)
            predicates.append(Predicate(name, formula))

    # Property itself is always a predicate
    if concrete_ts.prop_formula is not None:
        add_pred("property", concrete_ts.prop_formula)

    # Extract predicates from init formula
    init_preds = _extract_atomic_predicates(concrete_ts.init_formula)
    for i, (name, formula) in enumerate(init_preds):
        add_pred(f"init_{name}" if name else f"init_{i}", formula)

    # Extract predicates from property
    prop_preds = _extract_atomic_predicates(concrete_ts.prop_formula)
    for i, (name, formula) in enumerate(prop_preds):
        add_pred(f"prop_{name}" if name else f"prop_{i}", formula)

    # Variable-level predicates: x >= 0 for each integer variable
    for v in concrete_ts.int_vars:
        add_pred(f"{v}_ge0", App(Op.GE, [Var(v, INT), IntConst(0)], BOOL))

    return predicates


def _extract_atomic_predicates(formula):
    """Extract atomic comparison predicates from a formula."""
    if formula is None:
        return []

    result = []
    if isinstance(formula, App):
        if formula.op in (Op.EQ, Op.NEQ, Op.LT, Op.LE, Op.GT, Op.GE):
            # This is an atomic predicate
            name = _pred_name_from_formula(formula)
            result.append((name, formula))
        elif formula.op in (Op.AND, Op.OR):
            for a in formula.args:
                result.extend(_extract_atomic_predicates(a))
        elif formula.op == Op.NOT:
            result.extend(_extract_atomic_predicates(formula.args[0]))
    return result


def _pred_name_from_formula(formula):
    """Generate a readable name for an atomic predicate."""
    if isinstance(formula, App) and len(formula.args) == 2:
        lhs, rhs = formula.args
        op_names = {Op.EQ: "eq", Op.NEQ: "neq", Op.LT: "lt",
                    Op.LE: "le", Op.GT: "gt", Op.GE: "ge"}
        op_str = op_names.get(formula.op, "?")
        lhs_str = lhs.name if isinstance(lhs, Var) else str(lhs)
        rhs_str = rhs.name if isinstance(rhs, Var) else str(rhs)
        return f"{lhs_str}_{op_str}_{rhs_str}"
    return None


def verify_with_cegar(concrete_ts, predicates=None, max_iterations=10, max_pdr_frames=50):
    """Convenience API: verify a transition system with CEGAR.

    If no predicates provided, auto-generates them from the system.
    """
    if predicates is None:
        predicates = auto_predicates_from_ts(concrete_ts)
    return cegar_check(concrete_ts, predicates, max_iterations, max_pdr_frames)


# ---------------------------------------------------------------------------
# Source-level verification (composes with C010 parser)
# ---------------------------------------------------------------------------

def _parse_source(source):
    """Parse C10 source code to AST."""
    from stack_vm import lex, Parser
    tokens = lex(source)
    return Parser(tokens).parse()


def extract_loop_ts(source):
    """Extract a ConcreteTS from a C10 while-loop program.

    Expects a program of the form:
      let x = init_x;
      let y = init_y;
      while (cond) {
        x = update_x;
        y = update_y;
      }

    Returns a ConcreteTS suitable for CEGAR verification.
    """
    from stack_vm import lex, Parser

    tokens = lex(source)
    program = Parser(tokens).parse()

    cts = ConcreteTS()
    inits = {}
    trans_parts = []
    loop_cond = None

    for stmt in program.stmts:
        stype = type(stmt).__name__
        if stype == 'LetDecl':
            vname = stmt.name
            cts.int_vars.append(vname)
            init_val = _ast_to_smt(stmt.value, cts)
            inits[vname] = init_val
        elif stype == 'WhileStmt':
            loop_cond = _ast_to_smt(stmt.cond, cts)
            body_assigns = _extract_assignments(stmt.body, cts)
            # Build guarded transition:
            # (cond AND body_trans) OR (!cond AND frame)
            body_parts = []
            frame_parts = []
            for vname in cts.int_vars:
                v = cts.var(vname)
                vp = cts.prime(vname)
                if vname in body_assigns:
                    body_parts.append(_eq(vp, body_assigns[vname]))
                else:
                    body_parts.append(_eq(vp, v))
                frame_parts.append(_eq(vp, v))

            guarded = _or(
                _and(loop_cond, *body_parts),
                _and(_not(loop_cond), *frame_parts)
            )
            trans_parts.append(guarded)

    # Init formula
    init_clauses = [_eq(cts.var(v), inits[v]) for v in inits]
    cts.init_formula = _and(*init_clauses) if init_clauses else BoolConst(True)

    # Transition formula
    cts.trans_formula = _and(*trans_parts) if trans_parts else BoolConst(True)

    return cts


def _ast_to_smt(node, cts):
    """Convert a C10 AST expression to an SMT term."""
    ntype = type(node).__name__
    if ntype == 'IntLit':
        return IntConst(node.value)
    if ntype in ('ASTVar', 'Var'):
        return cts.var(node.name)
    if ntype == 'BinOp':
        left = _ast_to_smt(node.left, cts)
        right = _ast_to_smt(node.right, cts)
        ops = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
               '<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE,
               '==': Op.EQ, '!=': Op.NEQ}
        op = ops.get(node.op)
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        sort = BOOL if op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NEQ) else INT
        return App(op, [left, right], sort)
    if ntype == 'UnaryOp':
        operand = _ast_to_smt(node.operand, cts)
        if node.op == '-':
            return App(Op.NEG, [operand], INT)
    raise ValueError(f"Unsupported AST node: {ntype}")


def _extract_assignments(body, cts):
    """Extract simple assignments from a loop body."""
    assigns = {}
    stmts = body.stmts if hasattr(body, 'stmts') else body
    for stmt in stmts:
        stype = type(stmt).__name__
        if stype == 'Assign':
            assigns[stmt.name] = _ast_to_smt(stmt.value, cts)
        elif stype == 'IfStmt':
            # Conditional assignment -> ITE
            cond = _ast_to_smt(stmt.cond, cts)
            then_assigns = _extract_assignments(stmt.then_body, cts)
            else_assigns = _extract_assignments(stmt.else_body, cts) if stmt.else_body else {}
            for vname in set(list(then_assigns.keys()) + list(else_assigns.keys())):
                then_val = then_assigns.get(vname, cts.var(vname))
                else_val = else_assigns.get(vname, cts.var(vname))
                assigns[vname] = _ite(cond, then_val, else_val)
    return assigns


def verify_loop_with_cegar(source, property_source, predicates=None, **kwargs):
    """Verify a property about a loop using CEGAR.

    Args:
        source: C10 source with a while loop
        property_source: A string like "x >= 0" (over loop variables)
        predicates: Optional initial predicates
        **kwargs: passed to cegar_check

    Returns: CEGARResult
    """
    cts = extract_loop_ts(source)

    # Parse property
    prop = _parse_property(property_source, cts)
    cts.prop_formula = prop

    return verify_with_cegar(cts, predicates, **kwargs)


def _parse_property(prop_str, cts):
    """Parse a simple property string into an SMT formula."""
    # Support: "x >= 0", "x + y == 10", etc.
    from stack_vm import lex, Parser

    # Wrap as an expression statement
    tokens = lex(prop_str + ";")
    program = Parser(tokens).parse()
    if program.stmts:
        return _ast_to_smt(program.stmts[0], cts)
    raise ValueError(f"Cannot parse property: {prop_str}")
