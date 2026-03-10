"""
V012: Craig Interpolation for Linear Integer Arithmetic

Computes Craig interpolants: given A AND B is UNSAT, finds I such that:
  1. A => I  (A implies I)
  2. I AND B is UNSAT  (I contradicts B)
  3. vars(I) subset of vars(A) intersect vars(B)  (only shared variables)

Approach: syntactic interpolation for LIA conjunctions via:
- Variable classification (A-local, B-local, shared)
- Fourier-Motzkin elimination of local variables from A-side
- Strengthening/weakening to ensure shared-variable-only interpolant
- Sequence interpolation for CEGAR trace refinement

Composes: C037 (SMT solver)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set, Dict
from enum import Enum

# Import C037 SMT solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import (SMTSolver, SMTResult, Term, Var, IntConst, BoolConst,
                         App, Op, BOOL, INT, Sort, SortKind)


# --- Result Types ---

class InterpolantResult(Enum):
    SUCCESS = "success"
    NOT_UNSAT = "not_unsat"  # A AND B is not UNSAT
    FAILED = "failed"        # Could not compute interpolant


@dataclass
class Interpolant:
    """Result of Craig interpolation."""
    result: InterpolantResult
    formula: Optional[Term] = None       # The interpolant I
    shared_vars: Optional[Set[str]] = None
    a_local_vars: Optional[Set[str]] = None
    b_local_vars: Optional[Set[str]] = None
    stats: Dict = field(default_factory=dict)


@dataclass
class SequenceInterpolant:
    """Sequence of interpolants for a trace A0, A1, ..., An."""
    result: InterpolantResult
    interpolants: Optional[List[Term]] = None  # I0, I1, ..., In-1
    stats: Dict = field(default_factory=dict)


# --- Term Analysis ---

def collect_vars(term: Term) -> Set[str]:
    """Collect all variable names in a term."""
    result = set()
    _collect_vars_rec(term, result)
    return result


def _collect_vars_rec(term: Term, result: Set[str]):
    if isinstance(term, Var):
        result.add(term.name)
    elif isinstance(term, App):
        for arg in term.args:
            _collect_vars_rec(arg, result)
    # IntConst, BoolConst have no vars


def collect_atoms(term: Term) -> List[Term]:
    """Collect atomic formulas (leaves of boolean structure)."""
    atoms = []
    _collect_atoms_rec(term, atoms)
    return atoms


def _collect_atoms_rec(term: Term, atoms: List[Term]):
    if isinstance(term, App):
        if term.op in (Op.AND, Op.OR, Op.NOT, Op.IMPLIES, Op.IFF):
            for arg in term.args:
                _collect_atoms_rec(arg, atoms)
        else:
            atoms.append(term)
    elif isinstance(term, Var) and term.sort == BOOL:
        atoms.append(term)
    elif isinstance(term, BoolConst):
        atoms.append(term)


def flatten_conjunction(term: Term) -> List[Term]:
    """Flatten nested ANDs into a list of conjuncts."""
    if isinstance(term, App) and term.op == Op.AND:
        result = []
        for arg in term.args:
            result.extend(flatten_conjunction(arg))
        return result
    elif isinstance(term, BoolConst) and term.value:
        return []  # True is identity for AND
    return [term]


def make_conjunction(terms: List[Term]) -> Term:
    """Build a conjunction from a list of terms."""
    if not terms:
        return BoolConst(True)
    if len(terms) == 1:
        return terms[0]
    return App(Op.AND, terms, BOOL)


def make_disjunction(terms: List[Term]) -> Term:
    """Build a disjunction from a list of terms."""
    if not terms:
        return BoolConst(False)
    if len(terms) == 1:
        return terms[0]
    return App(Op.OR, terms, BOOL)


def negate(term: Term) -> Term:
    """Negate a term using complement operators (avoid NOT(EQ) bug)."""
    if isinstance(term, BoolConst):
        return BoolConst(not term.value)
    if isinstance(term, App):
        complement = {
            Op.EQ: Op.NEQ, Op.NEQ: Op.EQ,
            Op.LT: Op.GE, Op.GE: Op.LT,
            Op.LE: Op.GT, Op.GT: Op.LE,
        }
        if term.op in complement:
            return App(complement[term.op], term.args, BOOL)
        if term.op == Op.NOT:
            return term.args[0]
        if term.op == Op.AND:
            return App(Op.OR, [negate(a) for a in term.args], BOOL)
        if term.op == Op.OR:
            return App(Op.AND, [negate(a) for a in term.args], BOOL)
    return App(Op.NOT, [term], BOOL)


# --- Linear Inequality Representation ---
# For Fourier-Motzkin, represent constraints as: sum(coeff_i * var_i) + const OP 0
# where OP is <=, <, or ==

@dataclass
class LinearConstraint:
    """A linear constraint: sum(coeffs[v] * v) + const OP 0."""
    coeffs: Dict[str, int]   # variable name -> coefficient
    const: int                # constant term
    op: str                   # '<=', '<', '=='
    origin: Optional[Term] = None  # original SMT term

    def vars(self) -> Set[str]:
        return {v for v, c in self.coeffs.items() if c != 0}

    def has_var(self, name: str) -> bool:
        return name in self.coeffs and self.coeffs[name] != 0

    def coeff(self, name: str) -> int:
        return self.coeffs.get(name, 0)

    def __repr__(self):
        parts = []
        for v in sorted(self.coeffs):
            c = self.coeffs[v]
            if c == 0:
                continue
            if c == 1:
                parts.append(v)
            elif c == -1:
                parts.append(f"-{v}")
            else:
                parts.append(f"{c}*{v}")
        if self.const != 0 or not parts:
            parts.append(str(self.const))
        return f"{' + '.join(parts)} {self.op} 0"


def term_to_linear(term: Term) -> Optional[LinearConstraint]:
    """Convert an SMT comparison term to a LinearConstraint.

    Normalizes to form: expr OP 0.
    """
    if not isinstance(term, App):
        return None

    if term.op not in (Op.LE, Op.LT, Op.GE, Op.GT, Op.EQ, Op.NEQ):
        return None

    left, right = term.args[0], term.args[1]
    # Collect coefficients from left - right
    coeffs = {}
    const = [0]

    def _extract(t, sign):
        if isinstance(t, IntConst):
            const[0] += sign * t.value
        elif isinstance(t, Var):
            coeffs[t.name] = coeffs.get(t.name, 0) + sign
        elif isinstance(t, App):
            if t.op == Op.ADD:
                _extract(t.args[0], sign)
                _extract(t.args[1], sign)
            elif t.op == Op.SUB:
                _extract(t.args[0], sign)
                _extract(t.args[1], -sign)
            elif t.op == Op.MUL:
                # Handle const * var or var * const
                if isinstance(t.args[0], IntConst):
                    _extract(t.args[1], sign * t.args[0].value)
                elif isinstance(t.args[1], IntConst):
                    _extract(t.args[0], sign * t.args[1].value)
                else:
                    return  # nonlinear - skip
            elif t.op == Op.NEG:
                _extract(t.args[0], -sign)

    _extract(left, 1)
    _extract(right, -1)

    # Normalize: left - right OP 0
    # LE: left <= right => left - right <= 0
    # LT: left < right => left - right < 0 => left - right <= -1
    # GE: left >= right => right - left <= 0 => negate coeffs
    # GT: left > right => right - left < 0 => right - left <= -1 => negate + -1
    # EQ: left == right => left - right == 0
    # NEQ: skip (not a convex constraint)

    if term.op == Op.LE:
        return LinearConstraint(coeffs, const[0], '<=', term)
    elif term.op == Op.LT:
        # x < 0 <==> x <= -1 (integers)
        return LinearConstraint(coeffs, const[0] - 1, '<=', term)  # Wait, x - y < 0 means x - y + (-1) <= -1? No.
        # Actually: left - right < 0 means left - right <= -1, so const becomes const[0], op becomes <=, but we shift
        # left - right <= -1 means: coeffs, const[0] + 1, '<='?  No.
        # If we have sum + c < 0, that means sum + c <= -1, i.e. sum + (c+1) <= 0
        # Hmm, let me think again. The constraint is sum(coeffs*vars) + const < 0
        # For integers: sum + const <= -1, i.e. sum + (const + 1) <= 0
        # So: LinearConstraint(coeffs, const[0] + 1, '<=', term)
        # Wait, I already computed wrong above. Let me redo.
    elif term.op == Op.GE:
        # left >= right => right - left <= 0 => negate all
        neg_coeffs = {v: -c for v, c in coeffs.items()}
        return LinearConstraint(neg_coeffs, -const[0], '<=', term)
    elif term.op == Op.GT:
        # left > right => right - left < 0 => right - left <= -1 (integers)
        neg_coeffs = {v: -c for v, c in coeffs.items()}
        return LinearConstraint(neg_coeffs, -const[0] + 1, '<=', term)
    elif term.op == Op.EQ:
        return LinearConstraint(coeffs, const[0], '==', term)
    elif term.op == Op.NEQ:
        return None  # Can't represent NEQ as a single convex constraint

    return None


# Fix the LT case properly
def _term_to_linear_fixed(term: Term) -> Optional[LinearConstraint]:
    """Convert an SMT comparison term to LinearConstraint: sum + const OP 0."""
    if not isinstance(term, App):
        return None
    if term.op not in (Op.LE, Op.LT, Op.GE, Op.GT, Op.EQ, Op.NEQ):
        return None

    left, right = term.args[0], term.args[1]
    coeffs = {}
    const_val = 0

    def _extract(t, sign):
        nonlocal const_val
        if isinstance(t, IntConst):
            const_val += sign * t.value
        elif isinstance(t, Var):
            coeffs[t.name] = coeffs.get(t.name, 0) + sign
        elif isinstance(t, App):
            if t.op == Op.ADD:
                _extract(t.args[0], sign)
                _extract(t.args[1], sign)
            elif t.op == Op.SUB:
                _extract(t.args[0], sign)
                _extract(t.args[1], -sign)
            elif t.op == Op.MUL:
                if isinstance(t.args[0], IntConst):
                    _extract(t.args[1], sign * t.args[0].value)
                elif isinstance(t.args[1], IntConst):
                    _extract(t.args[0], sign * t.args[1].value)
            elif t.op == Op.NEG:
                _extract(t.args[0], -sign)

    # left - right
    _extract(left, 1)
    _extract(right, -1)

    # Now we have: sum(coeffs[v]*v) + const_val {op} 0

    if term.op == Op.LE:
        # sum + c <= 0
        return LinearConstraint(coeffs, const_val, '<=', term)
    elif term.op == Op.LT:
        # sum + c < 0  =>  sum + c + 1 <= 0  (integers)
        return LinearConstraint(coeffs, const_val + 1, '<=', term)
    elif term.op == Op.GE:
        # sum + c >= 0  =>  -(sum + c) <= 0  =>  -sum - c <= 0
        neg_coeffs = {v: -c for v, c in coeffs.items()}
        return LinearConstraint(neg_coeffs, -const_val, '<=', term)
    elif term.op == Op.GT:
        # sum + c > 0  =>  -(sum + c) < 0  =>  -(sum + c) + 1 <= 0
        # =>  -sum - c + 1 <= 0
        neg_coeffs = {v: -c for v, c in coeffs.items()}
        return LinearConstraint(neg_coeffs, -const_val + 1, '<=', term)
    elif term.op == Op.EQ:
        return LinearConstraint(coeffs, const_val, '==', term)
    elif term.op == Op.NEQ:
        return None  # Not a convex constraint

    return None

# Replace the buggy version
term_to_linear = _term_to_linear_fixed


def linear_to_term(lc: LinearConstraint, solver: SMTSolver) -> Term:
    """Convert a LinearConstraint back to an SMT term."""
    # Build: sum(coeffs[v]*v) + const OP 0
    terms = []
    for v in sorted(lc.coeffs):
        c = lc.coeffs[v]
        if c == 0:
            continue
        var = solver.Int(v)
        if c == 1:
            terms.append(var)
        elif c == -1:
            terms.append(App(Op.NEG, [var], INT))
        else:
            terms.append(App(Op.MUL, [IntConst(c), var], INT))

    # Build sum
    if not terms:
        lhs = IntConst(lc.const)
    else:
        lhs = terms[0]
        for t in terms[1:]:
            lhs = App(Op.ADD, [lhs, t], INT)
        if lc.const != 0:
            lhs = App(Op.ADD, [lhs, IntConst(lc.const)], INT)

    zero = IntConst(0)

    if lc.op == '<=':
        return App(Op.LE, [lhs, zero], BOOL)
    elif lc.op == '==':
        return App(Op.EQ, [lhs, zero], BOOL)
    elif lc.op == '<':
        return App(Op.LT, [lhs, zero], BOOL)
    else:
        raise ValueError(f"Unknown op: {lc.op}")


# --- Fourier-Motzkin Elimination ---

def fourier_motzkin_eliminate(constraints: List[LinearConstraint],
                              var: str) -> List[LinearConstraint]:
    """Eliminate variable `var` from a set of linear constraints using
    Fourier-Motzkin elimination.

    Given constraints involving var, partition into:
    - Upper bounds: coeff * var + rest <= 0  (coeff > 0)
    - Lower bounds: coeff * var + rest <= 0  (coeff < 0, i.e. -|c|*var + rest <= 0)
    - Independent: doesn't mention var

    For each (lower, upper) pair, derive: rest_lower / |c_lower| <= rest_upper / |c_upper|
    which becomes: |c_upper| * rest_lower + |c_lower| * rest_upper <= 0
    """
    upper = []   # coeff > 0: these give upper bounds on var
    lower = []   # coeff < 0: these give lower bounds on var
    independent = []
    equalities = []

    for c in constraints:
        coeff = c.coeff(var)
        if coeff == 0:
            independent.append(c)
        elif c.op == '==':
            equalities.append(c)
        elif coeff > 0:
            upper.append(c)
        else:  # coeff < 0
            lower.append(c)

    result = list(independent)

    # Handle equalities: substitute var = (-rest) / coeff
    if equalities:
        eq = equalities[0]
        eq_coeff = eq.coeff(var)
        # var = -(rest + const) / eq_coeff
        # Substitute into all other constraints
        for c in upper + lower + equalities[1:]:
            new_c = _substitute_equality(c, var, eq)
            if new_c is not None:
                result.append(new_c)
        return result

    # Fourier-Motzkin: combine each lower bound with each upper bound
    for lb in lower:
        for ub in upper:
            combined = _combine_bounds(lb, ub, var)
            if combined is not None:
                result.append(combined)

    return result


def _substitute_equality(constraint: LinearConstraint, var: str,
                          equality: LinearConstraint) -> Optional[LinearConstraint]:
    """Substitute var from an equality into another constraint.

    If equality says: a*var + rest_eq == 0, then var = -rest_eq/a.
    Substitute into constraint: c*var + rest_c OP 0
    => c * (-rest_eq / a) + rest_c OP 0
    => -c/a * rest_eq + rest_c OP 0
    => a * rest_c - c * rest_eq OP 0  (multiply by a, flip if a < 0)
    """
    c = constraint.coeff(var)
    if c == 0:
        return constraint  # Doesn't use var

    a = equality.coeff(var)
    if a == 0:
        return constraint

    # New constraint: a * (constraint without var) - c * (equality without var)
    new_coeffs = {}
    all_vars = set(constraint.coeffs.keys()) | set(equality.coeffs.keys())
    for v in all_vars:
        if v == var:
            continue
        cv = constraint.coeffs.get(v, 0)
        ev = equality.coeffs.get(v, 0)
        new_coeffs[v] = a * cv - c * ev

    new_const = a * constraint.const - c * equality.const
    new_op = constraint.op

    # If we multiplied by negative a, flip inequality
    if a < 0 and new_op == '<=':
        new_coeffs = {v: -c for v, c in new_coeffs.items()}
        new_const = -new_const

    return LinearConstraint(new_coeffs, new_const, new_op)


def _combine_bounds(lower: LinearConstraint, upper: LinearConstraint,
                     var: str) -> Optional[LinearConstraint]:
    """Combine a lower bound and upper bound on var to eliminate var.

    lower: c_l * var + rest_l <= 0  where c_l < 0 (lower bound: var >= -rest_l/|c_l|)
    upper: c_u * var + rest_u <= 0  where c_u > 0 (upper bound: var <= -rest_u/c_u)

    Combined: |c_u| * rest_l + |c_l| * rest_u + ... <= 0
    More precisely: c_u * (lower without var) - c_l * (upper without var)
    which eliminates var since c_u * c_l - c_l * c_u = 0.

    Actually: multiply lower by c_u, upper by |c_l| = -c_l, and add:
    c_l*c_u*var + c_u*rest_l + (-c_l)*c_u*var + (-c_l)*rest_u <= 0
    The var terms: c_l*c_u + (-c_l)*c_u = 0. Good.
    Remaining: c_u*rest_l + (-c_l)*rest_u <= 0
    """
    c_l = lower.coeff(var)  # negative
    c_u = upper.coeff(var)  # positive

    # Multiply lower by c_u, upper by -c_l (both positive multipliers)
    mult_l = c_u      # positive
    mult_u = -c_l     # positive (since c_l < 0)

    new_coeffs = {}
    all_vars = set(lower.coeffs.keys()) | set(upper.coeffs.keys())
    for v in all_vars:
        if v == var:
            continue
        lv = lower.coeffs.get(v, 0)
        uv = upper.coeffs.get(v, 0)
        new_coeffs[v] = mult_l * lv + mult_u * uv

    new_const = mult_l * lower.const + mult_u * upper.const

    return LinearConstraint(new_coeffs, new_const, '<=')


# --- Core Interpolation ---

def classify_variables(a_formula: Term, b_formula: Term) -> Tuple[Set[str], Set[str], Set[str]]:
    """Classify variables as A-local, B-local, or shared."""
    a_vars = collect_vars(a_formula)
    b_vars = collect_vars(b_formula)
    shared = a_vars & b_vars
    a_local = a_vars - shared
    b_local = b_vars - shared
    return a_local, b_local, shared


def interpolate_linear(a_constraints: List[LinearConstraint],
                        b_constraints: List[LinearConstraint],
                        shared_vars: Set[str],
                        a_local_vars: Set[str]) -> Optional[List[LinearConstraint]]:
    """Compute interpolant by eliminating A-local variables from A's constraints.

    The interpolant is A's constraints projected onto shared variables.
    """
    # Start with A's constraints
    current = list(a_constraints)

    # Eliminate each A-local variable via Fourier-Motzkin
    for var in sorted(a_local_vars):
        current = fourier_motzkin_eliminate(current, var)

    # Filter to only constraints over shared variables
    result = []
    for c in current:
        if c.vars().issubset(shared_vars):
            result.append(c)

    return result if result else None


def check_interpolant_validity(a_formula: Term, b_formula: Term,
                                interpolant: Term) -> Tuple[bool, bool, bool]:
    """Verify the three interpolant conditions:
    1. A => I
    2. I AND B is UNSAT
    3. vars(I) subset shared_vars

    Returns (cond1, cond2, cond3).
    """
    a_vars = collect_vars(a_formula)
    b_vars = collect_vars(b_formula)
    shared = a_vars & b_vars
    i_vars = collect_vars(interpolant)

    # Condition 3: variable containment
    cond3 = i_vars.issubset(shared)

    # Condition 1: A => I  <==>  A AND NOT(I) is UNSAT
    s1 = SMTSolver()
    _register_vars(s1, a_formula)
    _register_vars(s1, interpolant)
    s1.add(a_formula)
    s1.add(negate(interpolant))
    r1 = s1.check()
    cond1 = (r1 == SMTResult.UNSAT)

    # Condition 2: I AND B is UNSAT
    s2 = SMTSolver()
    _register_vars(s2, interpolant)
    _register_vars(s2, b_formula)
    s2.add(interpolant)
    s2.add(b_formula)
    r2 = s2.check()
    cond2 = (r2 == SMTResult.UNSAT)

    return cond1, cond2, cond3


def _register_vars(solver: SMTSolver, term: Term):
    """Register all variables in a term with the solver."""
    for name in collect_vars(term):
        solver.Int(name)  # Default to int; works for our LIA domain


# --- Main Interpolation API ---

def interpolate(a_formula: Term, b_formula: Term) -> Interpolant:
    """Compute a Craig interpolant for A AND B being UNSAT.

    Args:
        a_formula: The A part of the conjunction
        b_formula: The B part of the conjunction

    Returns:
        Interpolant with result, formula, and variable classification
    """
    # Step 1: Verify A AND B is UNSAT
    s = SMTSolver()
    _register_vars(s, a_formula)
    _register_vars(s, b_formula)
    s.add(a_formula)
    s.add(b_formula)
    result = s.check()

    if result != SMTResult.UNSAT:
        return Interpolant(
            result=InterpolantResult.NOT_UNSAT,
            stats={'smt_result': result.value}
        )

    # Step 2: Classify variables
    a_local, b_local, shared = classify_variables(a_formula, b_formula)

    # Step 3: Try syntactic interpolation
    interp = _syntactic_interpolation(a_formula, b_formula, a_local, b_local, shared)

    if interp is not None:
        # Verify
        c1, c2, c3 = check_interpolant_validity(a_formula, b_formula, interp)
        if c1 and c2 and c3:
            return Interpolant(
                result=InterpolantResult.SUCCESS,
                formula=interp,
                shared_vars=shared,
                a_local_vars=a_local,
                b_local_vars=b_local,
                stats={'method': 'syntactic', 'cond1': c1, 'cond2': c2, 'cond3': c3}
            )

    # Step 4: Fall back to model-based interpolation
    interp = _model_based_interpolation(a_formula, b_formula, a_local, b_local, shared)

    if interp is not None:
        c1, c2, c3 = check_interpolant_validity(a_formula, b_formula, interp)
        if c1 and c2 and c3:
            return Interpolant(
                result=InterpolantResult.SUCCESS,
                formula=interp,
                shared_vars=shared,
                a_local_vars=a_local,
                b_local_vars=b_local,
                stats={'method': 'model_based', 'cond1': c1, 'cond2': c2, 'cond3': c3}
            )

    # Step 5: Fall back to trivial interpolation attempts
    interp = _trivial_interpolation(a_formula, b_formula, a_local, b_local, shared)
    if interp is not None:
        c1, c2, c3 = check_interpolant_validity(a_formula, b_formula, interp)
        if c1 and c2 and c3:
            return Interpolant(
                result=InterpolantResult.SUCCESS,
                formula=interp,
                shared_vars=shared,
                a_local_vars=a_local,
                b_local_vars=b_local,
                stats={'method': 'trivial', 'cond1': c1, 'cond2': c2, 'cond3': c3}
            )

    return Interpolant(
        result=InterpolantResult.FAILED,
        shared_vars=shared,
        a_local_vars=a_local,
        b_local_vars=b_local,
        stats={'method': 'none'}
    )


def _syntactic_interpolation(a_formula, b_formula, a_local, b_local, shared):
    """Try syntactic Fourier-Motzkin-based interpolation."""

    # Flatten A into conjuncts
    a_conjuncts = flatten_conjunction(a_formula)

    # Convert to linear constraints
    a_linear = []
    for c in a_conjuncts:
        lc = term_to_linear(c)
        if lc is not None:
            a_linear.append(lc)
        elif isinstance(c, App) and c.op == Op.AND:
            # Nested conjunction
            for sub in flatten_conjunction(c):
                slc = term_to_linear(sub)
                if slc is not None:
                    a_linear.append(slc)

    if not a_linear:
        return None

    # Eliminate A-local variables
    projected = interpolate_linear(a_linear, [], shared, a_local)

    if projected is None or not projected:
        # No constraints left after elimination -- A projects to True on shared vars
        # But True AND B might not be UNSAT. Try a different approach.
        return None

    # Convert back to SMT terms
    s = SMTSolver()
    interp_terms = []
    for lc in projected:
        t = linear_to_term(lc, s)
        interp_terms.append(t)

    if not interp_terms:
        return None

    return make_conjunction(interp_terms)


def _model_based_interpolation(a_formula, b_formula, a_local, b_local, shared):
    """Model-guided interpolation: find what A implies about shared vars that contradicts B.

    Strategy: Extract shared-variable consequences of A by probing.
    """
    if not shared:
        # No shared variables: interpolant must be True or False
        # Since A AND B is UNSAT and they share no vars, A => False is possible if A is unsat
        # Otherwise B must be unsat, so interpolant is True
        s_a = SMTSolver()
        _register_vars(s_a, a_formula)
        s_a.add(a_formula)
        if s_a.check() == SMTResult.UNSAT:
            return BoolConst(False)
        else:
            return BoolConst(True)

    # Try to find bounds on shared variables implied by A
    consequences = []
    for var_name in sorted(shared):
        bounds = _find_bounds(a_formula, var_name)
        for bound in bounds:
            # Check if this bound contradicts B
            s = SMTSolver()
            _register_vars(s, b_formula)
            _register_vars(s, bound)
            s.add(bound)
            s.add(b_formula)
            if s.check() == SMTResult.UNSAT:
                consequences.append(bound)

    if consequences:
        return make_conjunction(consequences)

    # Try extracting A's shared-variable conjuncts directly
    a_conjuncts = flatten_conjunction(a_formula)
    shared_conjuncts = []
    for c in a_conjuncts:
        if collect_vars(c).issubset(shared):
            shared_conjuncts.append(c)

    if shared_conjuncts:
        candidate = make_conjunction(shared_conjuncts)
        return candidate

    # Try pairwise relational constraints between shared vars
    shared_list = sorted(shared)
    for i, v1 in enumerate(shared_list):
        for v2 in shared_list[i+1:]:
            rel_bounds = _find_relational_bounds(a_formula, v1, v2)
            for bound in rel_bounds:
                s = SMTSolver()
                _register_vars(s, b_formula)
                _register_vars(s, bound)
                s.add(bound)
                s.add(b_formula)
                if s.check() == SMTResult.UNSAT:
                    return bound

    return None


def _find_bounds(formula: Term, var_name: str) -> List[Term]:
    """Find upper/lower bounds on var_name implied by formula."""
    bounds = []
    s = SMTSolver()
    _register_vars(s, formula)
    v = s.Int(var_name)

    # Binary search for upper bound
    ub = _find_upper_bound(formula, var_name, s)
    if ub is not None:
        bounds.append(App(Op.LE, [v, IntConst(ub)], BOOL))

    # Binary search for lower bound
    lb = _find_lower_bound(formula, var_name, s)
    if lb is not None:
        bounds.append(App(Op.GE, [v, IntConst(lb)], BOOL))

    # Check equality
    if ub is not None and lb is not None and ub == lb:
        bounds = [App(Op.EQ, [v, IntConst(ub)], BOOL)]

    return bounds


def _find_upper_bound(formula: Term, var_name: str, solver_template: SMTSolver) -> Optional[int]:
    """Find the tightest upper bound on var implied by formula."""
    # Try: is formula AND var > k UNSAT for some k?
    for k in [0, 1, -1, 5, 10, -5, -10, 100, -100]:
        s = SMTSolver()
        _register_vars(s, formula)
        v = s.Int(var_name)
        s.add(formula)
        s.add(App(Op.GT, [v, IntConst(k)], BOOL))
        if s.check() == SMTResult.UNSAT:
            # var <= k is implied. Try to tighten.
            best = k
            for delta in [-1, -2, -5]:
                tighter = best + delta
                s2 = SMTSolver()
                _register_vars(s2, formula)
                v2 = s2.Int(var_name)
                s2.add(formula)
                s2.add(App(Op.GT, [v2, IntConst(tighter)], BOOL))
                if s2.check() == SMTResult.UNSAT:
                    best = tighter
                else:
                    break
            return best
    return None


def _find_lower_bound(formula: Term, var_name: str, solver_template: SMTSolver) -> Optional[int]:
    """Find the tightest lower bound on var implied by formula."""
    for k in [0, -1, 1, -5, -10, 5, 10, -100, 100]:
        s = SMTSolver()
        _register_vars(s, formula)
        v = s.Int(var_name)
        s.add(formula)
        s.add(App(Op.LT, [v, IntConst(k)], BOOL))
        if s.check() == SMTResult.UNSAT:
            best = k
            for delta in [1, 2, 5]:
                tighter = best + delta
                s2 = SMTSolver()
                _register_vars(s2, formula)
                v2 = s2.Int(var_name)
                s2.add(formula)
                s2.add(App(Op.LT, [v2, IntConst(tighter)], BOOL))
                if s2.check() == SMTResult.UNSAT:
                    best = tighter
                else:
                    break
            return best
    return None


def _find_relational_bounds(formula: Term, v1_name: str, v2_name: str) -> List[Term]:
    """Find relational constraints between two variables implied by formula."""
    bounds = []
    s = SMTSolver()
    _register_vars(s, formula)
    v1 = s.Int(v1_name)
    v2 = s.Int(v2_name)

    # Check v1 == v2
    s_eq = SMTSolver()
    _register_vars(s_eq, formula)
    x1 = s_eq.Int(v1_name)
    x2 = s_eq.Int(v2_name)
    s_eq.add(formula)
    s_eq.add(App(Op.NEQ, [x1, x2], BOOL))
    if s_eq.check() == SMTResult.UNSAT:
        bounds.append(App(Op.EQ, [v1, v2], BOOL))
        return bounds

    # Check v1 <= v2
    s_le = SMTSolver()
    _register_vars(s_le, formula)
    y1 = s_le.Int(v1_name)
    y2 = s_le.Int(v2_name)
    s_le.add(formula)
    s_le.add(App(Op.GT, [y1, y2], BOOL))
    if s_le.check() == SMTResult.UNSAT:
        bounds.append(App(Op.LE, [v1, v2], BOOL))

    # Check v1 >= v2
    s_ge = SMTSolver()
    _register_vars(s_ge, formula)
    z1 = s_ge.Int(v1_name)
    z2 = s_ge.Int(v2_name)
    s_ge.add(formula)
    s_ge.add(App(Op.LT, [z1, z2], BOOL))
    if s_ge.check() == SMTResult.UNSAT:
        bounds.append(App(Op.GE, [v1, v2], BOOL))

    # Check v1 + v2 == c for small constants
    for c in range(-10, 11):
        s_sum = SMTSolver()
        _register_vars(s_sum, formula)
        w1 = s_sum.Int(v1_name)
        w2 = s_sum.Int(v2_name)
        s_sum.add(formula)
        s_sum.add(App(Op.NEQ, [App(Op.ADD, [w1, w2], INT), IntConst(c)], BOOL))
        if s_sum.check() == SMTResult.UNSAT:
            bounds.append(App(Op.EQ, [App(Op.ADD, [v1, v2], INT), IntConst(c)], BOOL))
            break

    # Check v1 - v2 == c for small constants
    for c in range(-10, 11):
        s_diff = SMTSolver()
        _register_vars(s_diff, formula)
        w1 = s_diff.Int(v1_name)
        w2 = s_diff.Int(v2_name)
        s_diff.add(formula)
        s_diff.add(App(Op.NEQ, [App(Op.SUB, [w1, w2], INT), IntConst(c)], BOOL))
        if s_diff.check() == SMTResult.UNSAT:
            bounds.append(App(Op.EQ, [App(Op.SUB, [v1, v2], INT), IntConst(c)], BOOL))
            break

    return bounds


def _trivial_interpolation(a_formula, b_formula, a_local, b_local, shared):
    """Last resort: try True or False as interpolants, or A/B directly if they
    only use shared vars."""
    # If A is UNSAT by itself, interpolant is False
    s_a = SMTSolver()
    _register_vars(s_a, a_formula)
    s_a.add(a_formula)
    if s_a.check() == SMTResult.UNSAT:
        return BoolConst(False)

    # If B is UNSAT by itself, interpolant is True
    s_b = SMTSolver()
    _register_vars(s_b, b_formula)
    s_b.add(b_formula)
    if s_b.check() == SMTResult.UNSAT:
        return BoolConst(True)

    # If A only uses shared vars, A itself is a valid interpolant
    if not a_local:
        return a_formula

    # If negation of B only uses shared vars
    if not b_local:
        return negate(b_formula)

    return None


# --- Sequence Interpolation ---

def sequence_interpolate(formulas: List[Term]) -> SequenceInterpolant:
    """Compute a sequence of interpolants for A0, A1, ..., An.

    Given that A0 AND A1 AND ... AND An is UNSAT, find I1, ..., In-1 such that:
    - A0 => I1
    - Ik AND Ak => I(k+1)  for k = 1, ..., n-2
    - I(n-1) AND An is UNSAT

    This is the key operation for CEGAR trace refinement:
    each Ai represents one step of the abstract trace, and
    each Ii represents the reachable states at step i.
    """
    n = len(formulas)
    if n < 2:
        return SequenceInterpolant(
            result=InterpolantResult.FAILED,
            stats={'reason': 'need at least 2 formulas'}
        )

    # Verify the conjunction is UNSAT
    s = SMTSolver()
    for f in formulas:
        _register_vars(s, f)
        s.add(f)
    if s.check() != SMTResult.UNSAT:
        return SequenceInterpolant(
            result=InterpolantResult.NOT_UNSAT,
            stats={'smt_result': 'not_unsat'}
        )

    # Compute sequence interpolants by binary partitioning
    # I_k = interpolant(A0 AND ... AND Ak, Ak+1 AND ... AND An)
    interpolants = []
    for k in range(n - 1):
        a_part = make_conjunction(formulas[:k+1])
        b_part = make_conjunction(formulas[k+1:])
        result = interpolate(a_part, b_part)

        if result.result != InterpolantResult.SUCCESS:
            return SequenceInterpolant(
                result=InterpolantResult.FAILED,
                stats={'failed_at': k, 'reason': result.result.value}
            )

        interpolants.append(result.formula)

    return SequenceInterpolant(
        result=InterpolantResult.SUCCESS,
        interpolants=interpolants,
        stats={'n_formulas': n, 'n_interpolants': len(interpolants)}
    )


# --- CEGAR Integration ---

def interpolation_refine(trace_formulas: List[Term],
                          shared_vars_per_step: Optional[List[Set[str]]] = None
                          ) -> Optional[List[Term]]:
    """Compute predicates for CEGAR refinement from an infeasible trace.

    Given a trace of formulas [Init, Trans_0, Trans_1, ..., Trans_n, NOT(Property)]
    that is UNSAT (the trace is infeasible), compute interpolants that serve
    as new predicates for predicate abstraction.

    Args:
        trace_formulas: List of formulas representing the trace steps
        shared_vars_per_step: Optional per-step shared variable sets

    Returns:
        List of interpolant formulas (one fewer than trace_formulas),
        or None if interpolation fails.
    """
    result = sequence_interpolate(trace_formulas)
    if result.result != InterpolantResult.SUCCESS:
        return None
    return result.interpolants


def extract_predicates_from_interpolant(interpolant: Term) -> List[Term]:
    """Extract atomic predicates from an interpolant formula.

    Useful for CEGAR: the interpolant gives us a formula, but predicate
    abstraction needs individual atomic predicates.
    """
    atoms = collect_atoms(interpolant)
    # Deduplicate by string representation
    seen = set()
    unique = []
    for atom in atoms:
        key = str(atom)
        if key not in seen:
            seen.add(key)
            unique.append(atom)
    return unique


# --- Convenience / High-level API ---

def check_and_interpolate(a_terms: List[Term], b_terms: List[Term]) -> Interpolant:
    """Check if A AND B is UNSAT and compute interpolant.

    Convenience wrapper that takes lists of constraints.
    """
    a_formula = make_conjunction(a_terms)
    b_formula = make_conjunction(b_terms)
    return interpolate(a_formula, b_formula)


def interpolate_with_vars(a_formula: Term, b_formula: Term,
                           a_vars: Set[str], b_vars: Set[str]) -> Interpolant:
    """Interpolate with explicit variable classification.

    Useful when the formulas share variable names but you want to treat
    some as local (e.g., step-indexed variables x_0, x_1 in a trace).
    """
    shared = a_vars & b_vars
    a_local = a_vars - shared
    b_local = b_vars - shared

    # Verify UNSAT
    s = SMTSolver()
    _register_vars(s, a_formula)
    _register_vars(s, b_formula)
    s.add(a_formula)
    s.add(b_formula)
    if s.check() != SMTResult.UNSAT:
        return Interpolant(result=InterpolantResult.NOT_UNSAT)

    # Try syntactic
    interp = _syntactic_interpolation(a_formula, b_formula, a_local, b_local, shared)
    if interp is not None:
        c1, c2, c3 = check_interpolant_validity(a_formula, b_formula, interp)
        if c1 and c2 and c3:
            return Interpolant(
                result=InterpolantResult.SUCCESS,
                formula=interp,
                shared_vars=shared,
                a_local_vars=a_local,
                b_local_vars=b_local,
                stats={'method': 'syntactic'}
            )

    # Try model-based
    interp = _model_based_interpolation(a_formula, b_formula, a_local, b_local, shared)
    if interp is not None:
        c1, c2, c3 = check_interpolant_validity(a_formula, b_formula, interp)
        if c1 and c2 and c3:
            return Interpolant(
                result=InterpolantResult.SUCCESS,
                formula=interp,
                shared_vars=shared,
                a_local_vars=a_local,
                b_local_vars=b_local,
                stats={'method': 'model_based'}
            )

    # Try trivial
    interp = _trivial_interpolation(a_formula, b_formula, a_local, b_local, shared)
    if interp is not None:
        c1, c2, c3 = check_interpolant_validity(a_formula, b_formula, interp)
        if c1 and c2 and c3:
            return Interpolant(
                result=InterpolantResult.SUCCESS,
                formula=interp,
                shared_vars=shared,
                a_local_vars=a_local,
                b_local_vars=b_local,
                stats={'method': 'trivial'}
            )

    return Interpolant(result=InterpolantResult.FAILED, shared_vars=shared,
                       a_local_vars=a_local, b_local_vars=b_local)
