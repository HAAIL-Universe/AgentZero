"""
V002: Property-Directed Reachability (PDR/IC3)
Unbounded model checking using SMT (C037).

Unlike C036's bounded model checking which can only verify properties up to a
fixed depth, PDR can prove properties hold for ALL reachable states by
computing inductive invariants incrementally.

This is an infinite-state PDR implementation using Linear Integer Arithmetic,
adapted from the classic finite-state IC3 algorithm. Key adaptations:
  - Uses formula-based cubes (not just exact model points)
  - Property-directed blocking: uses NOT(property) as the initial bad region
  - Generalization via relative inductiveness checks with SMT

References: Bradley (2011) "SAT-Based Model Checking without Unrolling"
            Cimatti & Griggio (2012) "Software Model Checking via IC3"
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# Import C037 SMT solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import (SMTSolver, SMTResult, Var, IntConst, BoolConst,
                         App, Op, BOOL, INT, Sort, Term)


# --- Data Structures ---

class PDRResult(Enum):
    SAFE = "safe"          # Property holds for all reachable states
    UNSAFE = "unsafe"      # Counterexample found
    UNKNOWN = "unknown"    # Resource limit reached


@dataclass
class Counterexample:
    """A counterexample trace: sequence of states from init to property violation."""
    trace: list  # List of dicts mapping var names to values
    length: int  # Number of steps

    def __repr__(self):
        lines = [f"Counterexample (length {self.length}):"]
        for i, state in enumerate(self.trace):
            lines.append(f"  Step {i}: {state}")
        return "\n".join(lines)


@dataclass
class PDRStats:
    """Statistics about the PDR run."""
    frames_created: int = 0
    clauses_learned: int = 0
    smt_queries: int = 0
    blocked_cubes: int = 0
    propagated_clauses: int = 0


@dataclass
class PDROutput:
    """Result of a PDR check."""
    result: PDRResult
    invariant: Optional[list] = None       # Inductive invariant (list of clauses) if SAFE
    counterexample: Optional[Counterexample] = None  # If UNSAFE
    stats: PDRStats = field(default_factory=PDRStats)
    num_frames: int = 0


class TransitionSystem:
    """
    Defines a symbolic transition system for PDR.

    A transition system consists of:
      - state_vars: list of (name, sort) pairs defining state variables
      - init: formula over state vars describing initial states
      - trans: formula over state vars and primed state vars describing transitions
      - prop: formula over state vars that should hold in all reachable states

    Primed variables (next-state) are named with a "'" suffix automatically.
    """

    def __init__(self):
        self.state_vars = []     # [(name, sort), ...]
        self.init_formula = None # Term (formula over current-state vars)
        self.trans_formula = None # Term (formula over current + primed vars)
        self.prop_formula = None  # Term (formula over current-state vars)
        self._var_cache = {}     # name -> Var term

    def add_var(self, name, sort=INT):
        """Add a state variable."""
        self.state_vars.append((name, sort))
        return self._make_var(name, sort)

    def add_int_var(self, name):
        return self.add_var(name, INT)

    def add_bool_var(self, name):
        return self.add_var(name, BOOL)

    def _make_var(self, name, sort):
        if name not in self._var_cache:
            self._var_cache[name] = Var(name, sort)
        return self._var_cache[name]

    def var(self, name):
        """Get a reference to a state variable by name."""
        if name in self._var_cache:
            return self._var_cache[name]
        for vname, vsort in self.state_vars:
            if vname == name:
                return self._make_var(name, vsort)
        raise KeyError(f"Unknown variable: {name}")

    def prime(self, name):
        """Get the primed (next-state) version of a variable."""
        base_sort = None
        for vname, vsort in self.state_vars:
            if vname == name:
                base_sort = vsort
                break
        if base_sort is None:
            raise KeyError(f"Unknown variable: {name}")
        pname = name + "'"
        return self._make_var(pname, base_sort)

    def set_init(self, formula):
        """Set the initial state formula."""
        self.init_formula = formula

    def set_trans(self, formula):
        """Set the transition relation formula."""
        self.trans_formula = formula

    def set_property(self, formula):
        """Set the property to verify."""
        self.prop_formula = formula


def _substitute(term, mapping):
    """
    Substitute variables in a term according to mapping {name: new_term}.
    Returns a new term with substitutions applied.
    """
    if isinstance(term, Var):
        if term.name in mapping:
            return mapping[term.name]
        return term
    elif isinstance(term, (IntConst, BoolConst)):
        return term
    elif isinstance(term, App):
        new_args = [_substitute(a, mapping) for a in term.args]
        return App(term.op, new_args, term.sort)
    else:
        return term


def _negate(term):
    """
    Negate a boolean term. Uses complement operators instead of wrapping
    in NOT, because the SMT solver has a known asymmetry with NOT(EQ).
    """
    if isinstance(term, BoolConst):
        return BoolConst(not term.value)
    if isinstance(term, App):
        if term.op == Op.NOT:
            return term.args[0]
        # Use complement operators for comparisons
        complement = {
            Op.EQ: Op.NEQ, Op.NEQ: Op.EQ,
            Op.LT: Op.GE,  Op.GE: Op.LT,
            Op.LE: Op.GT,  Op.GT: Op.LE,
        }
        if term.op in complement:
            return App(complement[term.op], term.args, BOOL)
        # For AND/OR, use De Morgan's laws
        if term.op == Op.AND:
            return _or(*[_negate(a) for a in term.args])
        if term.op == Op.OR:
            return _and(*[_negate(a) for a in term.args])
        if term.op == Op.IMPLIES:
            # NOT(a => b) = a AND NOT b
            return _and(term.args[0], _negate(term.args[1]))
        if term.op == Op.IFF:
            # NOT(a <=> b): negate as (a AND NOT b) OR (NOT a AND b)
            # But simpler: just wrap in NOT for IFF
            return App(Op.NOT, [term], BOOL)
    return App(Op.NOT, [term], BOOL)


def _and(*terms):
    """Conjunction of terms."""
    terms = [t for t in terms if t is not None]
    # Flatten nested ANDs
    flat = []
    for t in terms:
        if isinstance(t, BoolConst):
            if not t.value:
                return BoolConst(False)
            continue  # Skip True
        if isinstance(t, App) and t.op == Op.AND:
            flat.extend(t.args)
        else:
            flat.append(t)
    if len(flat) == 0:
        return BoolConst(True)
    if len(flat) == 1:
        return flat[0]
    return App(Op.AND, flat, BOOL)


def _or(*terms):
    """Disjunction of terms."""
    terms = [t for t in terms if t is not None]
    flat = []
    for t in terms:
        if isinstance(t, BoolConst):
            if t.value:
                return BoolConst(True)
            continue  # Skip False
        if isinstance(t, App) and t.op == Op.OR:
            flat.extend(t.args)
        else:
            flat.append(t)
    if len(flat) == 0:
        return BoolConst(False)
    if len(flat) == 1:
        return flat[0]
    return App(Op.OR, flat, BOOL)


def _implies(a, b):
    """Implication."""
    return App(Op.IMPLIES, [a, b], BOOL)


def _eq(a, b):
    """Equality."""
    if a.sort == BOOL:
        return App(Op.IFF, [a, b], BOOL)
    return App(Op.EQ, [a, b], BOOL)


def _collect_conjuncts(formula):
    """Break a formula into its top-level conjuncts."""
    if isinstance(formula, App) and formula.op == Op.AND:
        result = []
        for arg in formula.args:
            result.extend(_collect_conjuncts(arg))
        return result
    return [formula]


def _extract_transition_map(trans_formula):
    """
    Extract a mapping {primed_var_name: expression} from a functional
    transition relation. Handles conjunctions of equalities like:
      x' == f(x) AND y' == g(x, y)

    Returns dict mapping primed var names to their defining expressions,
    or None if the transition isn't purely functional.
    """
    conjuncts = _collect_conjuncts(trans_formula)
    mapping = {}
    for conj in conjuncts:
        if isinstance(conj, App) and conj.op == Op.EQ:
            lhs, rhs = conj.args
            if isinstance(lhs, Var) and lhs.name.endswith("'"):
                mapping[lhs.name] = rhs
            elif isinstance(rhs, Var) and rhs.name.endswith("'"):
                mapping[rhs.name] = lhs
    return mapping if mapping else None


def _compute_pre_image(bad_formula, trans_map):
    """
    Compute the pre-image of bad_formula through a functional transition.
    Substitutes primed vars in bad_formula with their transition expressions.

    bad_formula is over current-state vars (but represents next-state constraint
    when primed). We need to:
    1. Prime bad_formula (replace x with x')
    2. Substitute x' with trans_map[x'] = f(x)
    This gives us a formula over current-state vars.
    """
    # First, collect all current-state var names mentioned in bad_formula
    # and build a priming substitution
    def _get_vars(term):
        if isinstance(term, Var):
            return {term.name}
        elif isinstance(term, App):
            result = set()
            for a in term.args:
                result.update(_get_vars(a))
            return result
        return set()

    var_names = _get_vars(bad_formula)

    # Build substitution: for each var x in bad_formula, replace with trans_map[x']
    # (which is the expression for x' in terms of current-state vars)
    substitution = {}
    for name in var_names:
        pname = name + "'"
        if pname in trans_map:
            substitution[name] = trans_map[pname]

    if not substitution:
        return None

    return _substitute(bad_formula, substitution)


# --- PDR Engine ---

class PDREngine:
    """
    Property-Directed Reachability engine for infinite-state systems.

    Implements an adaptation of IC3/PDR for Linear Integer Arithmetic
    using the C037 SMT solver. Key differences from finite-state IC3:
      - Cubes are formulas (not just conjunctions of propositional literals)
      - Uses NOT(property) as the initial bad region
      - Generalization checks relative inductiveness with SMT
    """

    def __init__(self, ts: TransitionSystem, max_frames=100):
        self.ts = ts
        self.max_frames = max_frames
        self.stats = PDRStats()
        self.frames = []

        # Variable mappings for current <-> primed
        self._curr_to_prime = {}
        self._prime_to_curr = {}
        for name, sort in ts.state_vars:
            pname = name + "'"
            curr_var = ts.var(name)
            prime_var = ts.prime(name)
            self._curr_to_prime[name] = prime_var
            self._prime_to_curr[pname] = curr_var

        # Try to extract functional transition map for pre-image computation
        self._trans_map = _extract_transition_map(ts.trans_formula) if ts.trans_formula else None

    def _compute_predecessor(self, bad_formula):
        """
        Compute the pre-image of bad_formula through the transition relation.
        Returns a formula over current-state vars, or None if can't compute.
        """
        if self._trans_map:
            pre = _compute_pre_image(bad_formula, self._trans_map)
            if pre is not None:
                return pre
        return None

    def _to_prime(self, formula):
        """Replace current-state vars with primed vars in formula."""
        return _substitute(formula, self._curr_to_prime)

    def _to_curr(self, formula):
        """Replace primed vars with current-state vars in formula."""
        return _substitute(formula, self._prime_to_curr)

    def _new_solver(self):
        """Create a fresh SMT solver with all state vars registered."""
        s = SMTSolver()
        for name, sort in self.ts.state_vars:
            if sort == INT:
                s.Int(name)
                s.Int(name + "'")
            elif sort == BOOL:
                s.Bool(name)
                s.Bool(name + "'")
        return s

    def _check_sat(self, *assertions):
        """Check satisfiability of a conjunction of assertions."""
        self.stats.smt_queries += 1
        s = self._new_solver()
        for a in assertions:
            if a is not None:
                s.add(a)
        result = s.check()
        if result == SMTResult.SAT:
            return True, s.model()
        return False, None

    def _frame_formula(self, i):
        """Get the formula for frame i (conjunction of all its clauses)."""
        if i == 0:
            return self.ts.init_formula
        if not self.frames[i]:
            return BoolConst(True)
        return _and(*self.frames[i])

    def _extract_model_state(self, model):
        """Extract state variable values from a model as a dict."""
        state = {}
        for name, _ in self.ts.state_vars:
            if name in model:
                state[name] = model[name]
        return state

    def _extract_cube_from_model(self, model):
        """
        Extract a cube (conjunction of literals) from a model.
        For integer vars: exact equality. For bool vars: polarity.
        Used for counterexample trace reconstruction and predecessor computation.
        """
        literals = []
        for name, sort in self.ts.state_vars:
            if name in model:
                val = model[name]
                v = self.ts.var(name)
                if sort == INT:
                    literals.append(_eq(v, IntConst(val)))
                elif sort == BOOL:
                    if val:
                        literals.append(v)
                    else:
                        literals.append(_negate(v))
        return literals

    def _is_blocked(self, formula, frame_idx):
        """Check if a formula is blocked by frame frame_idx (Fi AND formula is UNSAT)."""
        sat, _ = self._check_sat(self._frame_formula(frame_idx), formula)
        return not sat

    def _try_block_formula(self, bad_formula, frame_idx):
        """
        Try to find a clause that blocks bad_formula at frame frame_idx.

        Strategy:
        1. Check if bad_formula has a predecessor in F_{i-1}
        2. If not, learn a blocking clause (generalized from the UNSAT proof)
        3. If yes, return the predecessor for recursive blocking

        Returns:
          ('blocked', clause) if successfully blocked
          ('predecessor', model, pred_state) if predecessor found
          ('initial', model) if bad formula is reachable from init in one step
        """
        if frame_idx == 0:
            # Check if bad_formula intersects initial states
            sat, model = self._check_sat(self.ts.init_formula, bad_formula)
            if sat:
                return ('initial', model)
            else:
                return ('blocked', self.ts.prop_formula)

        # Check: F_{i-1} AND Trans AND bad_formula' is SAT?
        fi_prev = self._frame_formula(frame_idx - 1)
        primed_bad = self._to_prime(bad_formula)
        sat, model = self._check_sat(fi_prev, self.ts.trans_formula, primed_bad)

        if sat:
            # Predecessor exists
            return ('predecessor', model, self._extract_model_state(model))
        else:
            # No predecessor: learn blocking clause
            # The blocking clause is: anything that
            #   (a) holds in init
            #   (b) is relatively inductive wrt F_{i-1}
            #   (c) excludes bad_formula
            # Try the property itself first (most general)
            clause = self._find_blocking_clause(bad_formula, frame_idx - 1)
            return ('blocked', clause)

    def _find_blocking_clause(self, bad_formula, rel_frame_idx):
        """
        Find a clause that blocks bad_formula and is relatively inductive
        wrt frame rel_frame_idx.

        A valid blocking clause c must satisfy:
          1. Init => c  (holds in initial states)
          2. Fi AND c AND Trans => c'  (relatively inductive wrt Fi)
          3. c AND bad_formula is UNSAT  (actually excludes the bad states)

        Strategy (in order of preference):
        1. Property itself (most general, if it excludes bad)
        2. Negation of bad_formula (exact blocking)
        3. Conjuncts of property that exclude bad
        """
        prop = self.ts.prop_formula
        fi = self._frame_formula(rel_frame_idx)

        candidates = []

        # Strategy 1: Property itself (most general)
        candidates.append(prop)

        # Strategy 2: Pre-image of property through transition
        # Generalizes blocking to formula-level (e.g., x>=0 instead of x!=-1)
        if self._trans_map:
            pre_prop = _compute_pre_image(prop, self._trans_map)
            if pre_prop is not None and str(pre_prop) != str(prop):
                candidates.append(pre_prop)

        # Strategy 3: Conjuncts of property
        conjuncts = _collect_conjuncts(prop)
        if len(conjuncts) > 1:
            candidates.extend(conjuncts)

        # Strategy 4: NOT(bad_formula) -- exact blocking (least general)
        candidates.append(_negate(bad_formula))

        for clause in candidates:
            # Check 3: clause excludes bad_formula
            sat_excl, _ = self._check_sat(clause, bad_formula)
            if sat_excl:
                continue  # Clause doesn't exclude bad states

            # Check 1: Init => clause (Init AND NOT clause is UNSAT)
            sat_init, _ = self._check_sat(self.ts.init_formula, _negate(clause))
            if sat_init:
                continue  # Clause doesn't hold in init

            # Check 2: relatively inductive wrt Fi
            primed_neg = _negate(self._to_prime(clause))
            sat_ind, _ = self._check_sat(fi, clause, self.ts.trans_formula, primed_neg)
            if sat_ind:
                continue  # Not relatively inductive

            return clause

        # Fallback: NOT(bad_formula) without inductiveness requirement
        # This is always a valid blocking clause for the current frame
        return _negate(bad_formula)

    def check(self):
        """
        Run PDR algorithm.
        Returns PDROutput with result, optional invariant, optional counterexample.
        """
        ts = self.ts

        # Validate
        if ts.init_formula is None:
            raise ValueError("No initial state formula set")
        if ts.trans_formula is None:
            raise ValueError("No transition relation set")
        if ts.prop_formula is None:
            raise ValueError("No property formula set")

        # Step 0: Check if Init => Property
        sat, model = self._check_sat(ts.init_formula, _negate(ts.prop_formula))
        if sat:
            state = self._extract_model_state(model)
            return PDROutput(
                result=PDRResult.UNSAFE,
                counterexample=Counterexample(trace=[state], length=0),
                stats=self.stats,
                num_frames=0
            )

        # Initialize frames
        self.frames = [[], []]
        self.stats.frames_created = 2

        for iteration in range(self.max_frames):
            k = len(self.frames) - 1

            # --- Blocking phase ---
            # Check if Fk AND NOT(Property) is SAT
            fk = self._frame_formula(k)
            sat, model = self._check_sat(fk, _negate(ts.prop_formula))

            if sat:
                # Bad state found at frontier. Try to block NOT(property).
                result = self._block_bad_states(k)
                if result is not None:
                    return PDROutput(
                        result=PDRResult.UNSAFE,
                        counterexample=result,
                        stats=self.stats,
                        num_frames=len(self.frames)
                    )
                continue

            # --- Propagation phase ---
            self.frames.append([])
            self.stats.frames_created += 1

            fixpoint = self._propagate()
            if fixpoint is not None:
                invariant = list(self.frames[fixpoint])
                return PDROutput(
                    result=PDRResult.SAFE,
                    invariant=invariant,
                    stats=self.stats,
                    num_frames=len(self.frames)
                )

        return PDROutput(
            result=PDRResult.UNKNOWN,
            stats=self.stats,
            num_frames=len(self.frames)
        )

    def _block_bad_states(self, frontier):
        """
        Block all bad states (NOT property) from the frontier frame.

        Uses an obligation queue. Each obligation is:
          (bad_formula, frame_idx, trace_models)

        For infinite-state PDR, the initial bad formula is NOT(property)
        rather than an exact model point.

        Returns Counterexample if real cex found, None if all blocked.
        """
        neg_prop = _negate(self.ts.prop_formula)

        # Obligations: (formula_to_block, frame_index, trace_of_models)
        obligations = [(neg_prop, frontier, [])]
        max_obligations = 1000  # Safety bound

        processed = 0
        while obligations and processed < max_obligations:
            processed += 1

            # Prioritize higher frames
            obligations.sort(key=lambda o: -o[1])
            bad_formula, fi, trace_models = obligations.pop()

            if fi == 0:
                # Check if bad formula intersects init
                sat, model = self._check_sat(self.ts.init_formula, bad_formula)
                if sat:
                    # Real counterexample
                    init_state = self._extract_model_state(model)
                    full_trace = [init_state] + trace_models
                    return Counterexample(trace=full_trace, length=len(full_trace) - 1)
                else:
                    # Bad formula doesn't intersect init, blocked at frame 0
                    self.stats.blocked_cubes += 1
                    continue

            # Check if already blocked at this frame
            if self._is_blocked(bad_formula, fi):
                self.stats.blocked_cubes += 1
                continue

            # Check for predecessor
            fi_prev = self._frame_formula(fi - 1)
            primed_bad = self._to_prime(bad_formula)
            sat, model = self._check_sat(fi_prev, self.ts.trans_formula, primed_bad)

            if sat:
                # Predecessor found. Need to block it first.
                # Extract concrete states for the counterexample trace
                pred_state = self._extract_model_state(model)
                bad_state = {}
                for name, _ in self.ts.state_vars:
                    pname = name + "'"
                    if pname in model:
                        bad_state[name] = model[pname]

                # Use exact model cube for predecessor (for trace construction).
                # Pre-image is too aggressive here -- it can create deeply
                # nested formulas with ITE transitions.
                pred_cube = self._extract_cube_from_model(model)
                pred_formula = _and(*pred_cube) if pred_cube else BoolConst(True)

                # Re-enqueue current obligation
                obligations.append((bad_formula, fi, trace_models))
                # Enqueue predecessor
                new_trace = [bad_state] + trace_models if bad_state else trace_models
                obligations.append((pred_formula, fi - 1, new_trace))
            else:
                # No predecessor: learn blocking clause
                clause = self._find_blocking_clause(bad_formula, fi - 1)
                # Add to frames 1..fi
                clause_str = str(clause)
                for j in range(1, fi + 1):
                    if not any(str(c) == clause_str for c in self.frames[j]):
                        self.frames[j].append(clause)
                self.stats.clauses_learned += 1
                self.stats.blocked_cubes += 1

        return None

    def _propagate(self):
        """
        Push clauses from frame i to frame i+1 where possible.
        Returns frame index where fixpoint detected, or None.
        """
        for i in range(1, len(self.frames) - 1):
            for clause in list(self.frames[i]):
                fi = self._frame_formula(i)
                primed_neg = _negate(self._to_prime(clause))
                sat, _ = self._check_sat(fi, self.ts.trans_formula, primed_neg)
                if not sat:
                    clause_str = str(clause)
                    if not any(str(c) == clause_str for c in self.frames[i + 1]):
                        self.frames[i + 1].append(clause)
                        self.stats.propagated_clauses += 1

            # Check fixpoint: Fi subset of Fi+1 (or both empty)
            fi_strs = set(str(c) for c in self.frames[i])
            fi1_strs = set(str(c) for c in self.frames[i + 1])
            if fi_strs == fi1_strs:
                return i

        return None


# --- Convenience API ---

def check_ts(ts: TransitionSystem, max_frames=100):
    """Check a transition system using PDR. Returns PDROutput."""
    engine = PDREngine(ts, max_frames=max_frames)
    return engine.check()


def make_counter_system(bits=None, max_val=None):
    """
    Create a simple counter transition system for testing.
    Counter starts at 0, increments by 1 each step.
    """
    ts = TransitionSystem()
    c = ts.add_int_var("c")
    cp = ts.prime("c")

    ts.set_init(_eq(c, IntConst(0)))
    ts.set_trans(_eq(cp, App(Op.ADD, [c, IntConst(1)], INT)))

    if max_val is not None:
        ts.set_property(App(Op.LT, [c, IntConst(max_val)], BOOL))
    else:
        ts.set_property(App(Op.GE, [c, IntConst(0)], BOOL))

    return ts


def make_two_counter_system():
    """
    Two counters: x starts at 0 and increments, y starts at 10 and decrements.
    Property: x + y == 10 (should be invariant).
    """
    ts = TransitionSystem()
    x = ts.add_int_var("x")
    y = ts.add_int_var("y")
    xp = ts.prime("x")
    yp = ts.prime("y")

    ts.set_init(_and(
        _eq(x, IntConst(0)),
        _eq(y, IntConst(10))
    ))

    ts.set_trans(_and(
        _eq(xp, App(Op.ADD, [x, IntConst(1)], INT)),
        _eq(yp, App(Op.SUB, [y, IntConst(1)], INT))
    ))

    ts.set_property(
        _eq(App(Op.ADD, [x, y], INT), IntConst(10))
    )

    return ts
