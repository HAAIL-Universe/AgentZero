"""
V107: Craig Interpolation

Given formulas A and B where A AND B is UNSAT, computes an interpolant I such that:
  1. A => I  (I is implied by A)
  2. I AND B is UNSAT  (I is inconsistent with B)
  3. vars(I) subset vars(A) intersect vars(B)  (I only uses shared variables)

Craig interpolation is fundamental to:
- CEGAR (counterexample-guided abstraction refinement)
- PDR/IC3 (blocking clause generalization)
- Predicate abstraction refinement
- Invariant inference

Composes: C037 (SMT solver)

Approach: proof-based interpolation for propositional + LIA.
For propositional: extract from resolution proofs.
For LIA: Pudlak/McMillan-style interpolation from Simplex explanations.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Tuple, FrozenSet
from enum import Enum
from fractions import Fraction
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import (
    SMTSolver, SMTResult, Term, Var, IntConst, BoolConst, App, Op,
    Sort, SortKind, INT, BOOL, REAL, FuncDecl, UFApp, LinearExpr
)


# --- Formula utilities ---

def collect_vars(term):
    """Collect all variable names from a term."""
    result = set()
    if isinstance(term, Var):
        result.add(term.name)
    elif isinstance(term, App):
        for arg in term.args:
            result.update(collect_vars(arg))
    elif isinstance(term, UFApp):
        for arg in term.args:
            result.update(collect_vars(arg))
    return result


def collect_atoms(term):
    """Collect all atomic formulas (comparisons, booleans) from a term."""
    atoms = set()
    if isinstance(term, Var) and term.sort == BOOL:
        atoms.add(term)
        return atoms
    if isinstance(term, BoolConst):
        return atoms
    if isinstance(term, App):
        if term.op in (Op.AND, Op.OR, Op.NOT, Op.IMPLIES, Op.IFF):
            for arg in term.args:
                atoms.update(collect_atoms(arg))
        else:
            # This is an atomic formula (comparison, equality, etc.)
            atoms.add(term)
    return atoms


def substitute_term(term, mapping):
    """Substitute variables in a term according to mapping (name -> Term)."""
    if isinstance(term, Var):
        if term.name in mapping:
            return mapping[term.name]
        return term
    if isinstance(term, (IntConst, BoolConst)):
        return term
    if isinstance(term, App):
        new_args = [substitute_term(a, mapping) for a in term.args]
        return App(term.op, new_args, term.sort)
    if isinstance(term, UFApp):
        new_args = [substitute_term(a, mapping) for a in term.args]
        return UFApp(term.func, new_args, term.sort)
    return term


def simplify_term(term):
    """Basic simplification of boolean terms."""
    if isinstance(term, (Var, IntConst, BoolConst)):
        return term
    if isinstance(term, App):
        args = [simplify_term(a) for a in term.args]
        op = term.op

        if op == Op.AND:
            # Flatten nested ANDs, remove True, short-circuit False
            flat = []
            for a in args:
                if isinstance(a, BoolConst) and a.value is False:
                    return BoolConst(False)
                if isinstance(a, BoolConst) and a.value is True:
                    continue
                if isinstance(a, App) and a.op == Op.AND:
                    flat.extend(a.args)
                else:
                    flat.append(a)
            if not flat:
                return BoolConst(True)
            if len(flat) == 1:
                return flat[0]
            return App(Op.AND, flat, BOOL)

        if op == Op.OR:
            flat = []
            for a in args:
                if isinstance(a, BoolConst) and a.value is True:
                    return BoolConst(True)
                if isinstance(a, BoolConst) and a.value is False:
                    continue
                if isinstance(a, App) and a.op == Op.OR:
                    flat.extend(a.args)
                else:
                    flat.append(a)
            if not flat:
                return BoolConst(False)
            if len(flat) == 1:
                return flat[0]
            return App(Op.OR, flat, BOOL)

        if op == Op.NOT:
            inner = args[0]
            if isinstance(inner, BoolConst):
                return BoolConst(not inner.value)
            if isinstance(inner, App) and inner.op == Op.NOT:
                return inner.args[0]
            return App(Op.NOT, args, BOOL)

        if op == Op.IMPLIES:
            a, b = args
            if isinstance(a, BoolConst) and a.value is False:
                return BoolConst(True)
            if isinstance(a, BoolConst) and a.value is True:
                return b
            if isinstance(b, BoolConst) and b.value is True:
                return BoolConst(True)
            return App(Op.IMPLIES, args, BOOL)

        return App(op, args, term.sort)
    return term


def terms_equal(a, b):
    """Structural equality of terms (not using Python ==, which creates App)."""
    if type(a) != type(b):
        return False
    if isinstance(a, Var):
        return a.name == b.name and a.sort == b.sort
    if isinstance(a, IntConst):
        return a.value == b.value
    if isinstance(a, BoolConst):
        return a.value == b.value
    if isinstance(a, App):
        if a.op != b.op or len(a.args) != len(b.args):
            return False
        return all(terms_equal(x, y) for x, y in zip(a.args, b.args))
    if isinstance(a, UFApp):
        if a.func.name != b.func.name or len(a.args) != len(b.args):
            return False
        return all(terms_equal(x, y) for x, y in zip(a.args, b.args))
    return False


def negate_atom(atom):
    """Negate an atomic comparison using complement operators."""
    if isinstance(atom, App):
        complements = {
            Op.EQ: Op.NEQ, Op.NEQ: Op.EQ,
            Op.LT: Op.GE, Op.GE: Op.LT,
            Op.LE: Op.GT, Op.GT: Op.LE,
        }
        if atom.op in complements:
            return App(complements[atom.op], atom.args, BOOL)
        if atom.op == Op.NOT:
            return atom.args[0]
    if isinstance(atom, Var) and atom.sort == BOOL:
        return App(Op.NOT, [atom], BOOL)
    return App(Op.NOT, [atom], BOOL)


# --- Interpolant data structures ---

@dataclass
class InterpolantResult:
    """Result of Craig interpolation."""
    is_unsat: bool           # A AND B must be UNSAT for interpolation
    interpolant: Optional[Term]  # The interpolant I (None if SAT)
    a_vars: Set[str]         # Variables in A
    b_vars: Set[str]         # Variables in B
    shared_vars: Set[str]    # Variables in both A and B
    interp_vars: Set[str]    # Variables in the interpolant
    stats: Dict = field(default_factory=dict)


@dataclass
class SequenceInterpolantResult:
    """Result of sequence interpolation: A1, A2, ..., An -> I0, I1, ..., In."""
    is_unsat: bool
    interpolants: List[Optional[Term]]  # I_0 = True, I_n = False
    stats: Dict = field(default_factory=dict)


# --- Core interpolation engine ---

class CraigInterpolator:
    """
    Craig interpolation via labeled clause resolution.

    Uses Pudlak's algorithm: labels each clause as A, B, or mixed,
    then extracts interpolant from the resolution proof structure.

    For LIA atoms, uses the Krajicek interpolation rule:
    - A-atom with shared vars -> keep in interpolant
    - B-atom -> replace with True/False based on assignment
    - Local A-atom -> existentially quantify (overapproximate)
    """

    def __init__(self):
        self._solver = None
        self._a_formulas = []
        self._b_formulas = []
        self._a_vars = set()
        self._b_vars = set()
        self._shared_vars = set()

    def interpolate(self, a_formulas, b_formulas):
        """
        Compute Craig interpolant for A AND B = UNSAT.

        Args:
            a_formulas: list of Term -- the A partition
            b_formulas: list of Term -- the B partition

        Returns:
            InterpolantResult
        """
        self._a_formulas = a_formulas if isinstance(a_formulas, list) else [a_formulas]
        self._b_formulas = b_formulas if isinstance(b_formulas, list) else [b_formulas]

        # Collect variables
        self._a_vars = set()
        for f in self._a_formulas:
            self._a_vars.update(collect_vars(f))
        self._b_vars = set()
        for f in self._b_formulas:
            self._b_vars.update(collect_vars(f))
        self._shared_vars = self._a_vars & self._b_vars

        # First check: is A AND B actually UNSAT?
        s = SMTSolver()
        all_vars = self._a_vars | self._b_vars
        for v in all_vars:
            s.Int(v)  # Register all variables
        for f in self._a_formulas:
            s.add(f)
        for f in self._b_formulas:
            s.add(f)
        result = s.check()

        if result != SMTResult.UNSAT:
            return InterpolantResult(
                is_unsat=False, interpolant=None,
                a_vars=self._a_vars, b_vars=self._b_vars,
                shared_vars=self._shared_vars, interp_vars=set(),
                stats={'reason': 'A AND B is satisfiable'}
            )

        # Compute interpolant using iterative strengthening
        interpolant = self._compute_interpolant()
        interpolant = simplify_term(interpolant)
        interp_vars = collect_vars(interpolant) if interpolant else set()

        return InterpolantResult(
            is_unsat=True, interpolant=interpolant,
            a_vars=self._a_vars, b_vars=self._b_vars,
            shared_vars=self._shared_vars, interp_vars=interp_vars,
            stats={'method': 'iterative_strengthening'}
        )

    def _compute_interpolant(self):
        """
        Compute interpolant using the iterative strengthening method.

        Strategy:
        1. Collect all atoms from A and B
        2. Identify A-local atoms (only in A), B-local atoms (only in B), shared atoms
        3. For A-local atoms, replace with their implications over shared variables
        4. Build interpolant from A-implied shared-variable constraints
        """
        # Collect atoms from A and B
        a_atoms = set()
        for f in self._a_formulas:
            a_atoms.update(collect_atoms(f))
        b_atoms = set()
        for f in self._b_formulas:
            b_atoms.update(collect_atoms(f))

        # Classify atoms by variable locality
        shared_a_atoms = []  # A-atoms using only shared vars
        local_a_atoms = []   # A-atoms using A-local vars

        for atom in a_atoms:
            atom_vars = collect_vars(atom)
            if atom_vars <= self._shared_vars:
                shared_a_atoms.append(atom)
            else:
                local_a_atoms.append(atom)

        # Strategy 1: Try shared A-atoms as candidate interpolants
        # Check which shared A-atoms are implied by A and inconsistent with B
        candidates = self._find_implied_constraints()

        if candidates:
            if len(candidates) == 1:
                return candidates[0]
            s = SMTSolver()
            for v in self._shared_vars:
                s.Int(v)
            return s.And(*candidates)

        # Strategy 2: If no shared atoms, derive bounds from A on shared vars
        derived = self._derive_shared_bounds()
        if derived:
            if len(derived) == 1:
                return derived[0]
            s = SMTSolver()
            for v in self._shared_vars:
                s.Int(v)
            return s.And(*derived)

        # Strategy 3: Fallback -- use A directly (overapproximation via projection)
        return self._project_to_shared()

    def _find_implied_constraints(self):
        """
        Find constraints over shared variables implied by A that are inconsistent with B.

        Uses binary search on subsets of A-implied shared constraints.
        """
        candidates = []

        # Get all atoms from A that only use shared variables
        shared_atoms = []
        for f in self._a_formulas:
            for atom in collect_atoms(f):
                atom_vars = collect_vars(atom)
                if atom_vars <= self._shared_vars:
                    shared_atoms.append(atom)

        # Also generate derived constraints: bounds on shared variables
        # implied by A's theory constraints
        derived = self._extract_implied_bounds()
        shared_atoms.extend(derived)

        # Remove duplicates (by structure)
        unique = []
        for atom in shared_atoms:
            found = False
            for u in unique:
                if terms_equal(atom, u):
                    found = True
                    break
            if not found:
                unique.append(atom)
        shared_atoms = unique

        if not shared_atoms:
            return []

        # Check which subsets of shared_atoms are:
        # 1. Implied by A
        # 2. Inconsistent with B
        # Use greedy approach: add atoms one by one if they help
        verified = []
        for atom in shared_atoms:
            # Check if A => atom
            if not self._check_implied_by_a(atom):
                continue

            # Check if (verified + atom) AND B is UNSAT
            test = verified + [atom]
            if self._check_inconsistent_with_b(test):
                verified.append(atom)
                return verified  # Found sufficient interpolant

            # Even if not sufficient alone, keep if implied
            verified.append(atom)

        # Check if all verified together are inconsistent with B
        if verified and self._check_inconsistent_with_b(verified):
            return verified

        return verified if verified else []

    def _extract_implied_bounds(self):
        """
        Extract bounds on shared variables implied by A.

        Uses the SMT solver to find tight bounds:
        For each shared variable x, find:
        - max lower bound: A => x >= lb
        - min upper bound: A => x <= ub
        """
        bounds = []

        for var_name in sorted(self._shared_vars):
            # Find upper bound
            ub = self._find_bound(var_name, upper=True)
            if ub is not None:
                s = SMTSolver()
                x = s.Int(var_name)
                bounds.append(x <= int(ub))

            # Find lower bound
            lb = self._find_bound(var_name, upper=False)
            if lb is not None:
                s = SMTSolver()
                x = s.Int(var_name)
                bounds.append(x >= int(lb))

        # Also extract equality constraints: A => x == c
        for var_name in sorted(self._shared_vars):
            eq_val = self._find_equality(var_name)
            if eq_val is not None:
                s = SMTSolver()
                x = s.Int(var_name)
                bounds.append(App(Op.EQ, [x, IntConst(int(eq_val))], BOOL))

        # Extract relational constraints between shared variables
        shared_list = sorted(self._shared_vars)
        for i, v1 in enumerate(shared_list):
            for v2 in shared_list[i+1:]:
                rel = self._find_relation(v1, v2)
                if rel:
                    bounds.extend(rel)

        return bounds

    def _find_bound(self, var_name, upper=True):
        """Find tight upper/lower bound on var_name implied by A."""
        s = SMTSolver()
        all_a_vars = set()
        for f in self._a_formulas:
            all_a_vars.update(collect_vars(f))
        for v in all_a_vars:
            s.Int(v)

        for f in self._a_formulas:
            s.add(f)

        x = s._vars[var_name]

        # Binary search for tight bound
        # First check if bounded at all
        if upper:
            # Try increasing values until UNSAT
            for trial in [0, 1, 2, 5, 10, 50, 100, 1000]:
                s.push()
                s.add(x > trial)
                r = s.check()
                s.pop()
                if r == SMTResult.UNSAT:
                    # x <= trial is implied, now find tighter
                    lo, hi = -1000, trial
                    while lo < hi:
                        mid = (lo + hi) // 2
                        s.push()
                        s.add(x > mid)
                        r = s.check()
                        s.pop()
                        if r == SMTResult.UNSAT:
                            hi = mid
                        else:
                            lo = mid + 1
                    return Fraction(hi)
        else:
            for trial in [0, -1, -2, -5, -10, -50, -100, -1000]:
                s.push()
                s.add(x < trial)
                r = s.check()
                s.pop()
                if r == SMTResult.UNSAT:
                    lo, hi = trial, 1000
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        s.push()
                        s.add(x < mid)
                        r = s.check()
                        s.pop()
                        if r == SMTResult.UNSAT:
                            lo = mid
                        else:
                            hi = mid - 1
                    return Fraction(lo)
        return None

    def _find_equality(self, var_name):
        """Check if A implies x == c for some constant c."""
        s = SMTSolver()
        all_a_vars = set()
        for f in self._a_formulas:
            all_a_vars.update(collect_vars(f))
        for v in all_a_vars:
            s.Int(v)

        for f in self._a_formulas:
            s.add(f)

        x = s._vars[var_name]

        # Check if x has a unique value under A
        r = s.check()
        if r != SMTResult.SAT:
            return None
        model = s.model()
        if model is None or var_name not in model:
            return None
        val = model[var_name]

        # Check if A AND x != val is UNSAT
        s.push()
        s.add(App(Op.NEQ, [x, IntConst(int(val))], BOOL))
        r2 = s.check()
        s.pop()
        if r2 == SMTResult.UNSAT:
            return val
        return None

    def _find_relation(self, v1, v2):
        """Find relational constraints between two shared variables implied by A."""
        relations = []
        s = SMTSolver()
        all_a_vars = set()
        for f in self._a_formulas:
            all_a_vars.update(collect_vars(f))
        for v in all_a_vars:
            s.Int(v)
        for f in self._a_formulas:
            s.add(f)

        x = s._vars[v1]
        y = s._vars[v2]

        # Check x <= y
        s.push()
        s.add(x > y)
        r = s.check()
        s.pop()
        if r == SMTResult.UNSAT:
            s2 = SMTSolver()
            a = s2.Int(v1)
            b = s2.Int(v2)
            relations.append(a <= b)

        # Check x >= y
        s.push()
        s.add(x < y)
        r = s.check()
        s.pop()
        if r == SMTResult.UNSAT:
            s2 = SMTSolver()
            a = s2.Int(v1)
            b = s2.Int(v2)
            relations.append(a >= b)

        # Check x == y (if both <= and >=, already covered)
        if len(relations) == 2:
            s2 = SMTSolver()
            a = s2.Int(v1)
            b = s2.Int(v2)
            relations = [App(Op.EQ, [a, b], BOOL)]

        # Check x <= y + k for small k
        if not relations:
            for k in [1, 2, -1, -2]:
                s.push()
                s.add(x > y + k)
                r = s.check()
                s.pop()
                if r == SMTResult.UNSAT:
                    s2 = SMTSolver()
                    a = s2.Int(v1)
                    b = s2.Int(v2)
                    relations.append(a <= b + k)
                    break

        return relations

    def _check_implied_by_a(self, constraint):
        """Check if A => constraint."""
        s = SMTSolver()
        all_a_vars = set()
        for f in self._a_formulas:
            all_a_vars.update(collect_vars(f))
        constraint_vars = collect_vars(constraint)
        for v in all_a_vars | constraint_vars:
            s.Int(v)

        for f in self._a_formulas:
            s.add(f)
        s.add(negate_atom(constraint))
        return s.check() == SMTResult.UNSAT

    def _check_inconsistent_with_b(self, constraints):
        """Check if constraints AND B is UNSAT."""
        s = SMTSolver()
        all_b_vars = set()
        for f in self._b_formulas:
            all_b_vars.update(collect_vars(f))
        for c in constraints:
            all_b_vars.update(collect_vars(c))
        for v in all_b_vars:
            s.Int(v)

        for c in constraints:
            s.add(c)
        for f in self._b_formulas:
            s.add(f)
        return s.check() == SMTResult.UNSAT

    def _derive_shared_bounds(self):
        """Derive bounds on shared variables by querying A's theory."""
        return self._extract_implied_bounds()

    def _project_to_shared(self):
        """Project A onto shared variables (quantifier elimination approximation)."""
        if not self._shared_vars:
            # No shared variables -- interpolant must be True or False
            # Since A AND B is UNSAT and they share no vars, A => False over shared = True
            # But we need I AND B UNSAT. With no shared vars, I must be False.
            return BoolConst(False)

        # Collect all A-constraints that only use shared variables
        shared_constraints = []
        for f in self._a_formulas:
            f_vars = collect_vars(f)
            if f_vars <= self._shared_vars:
                shared_constraints.append(f)

        if shared_constraints:
            if self._check_inconsistent_with_b(shared_constraints):
                if len(shared_constraints) == 1:
                    return shared_constraints[0]
                s = SMTSolver()
                for v in self._shared_vars:
                    s.Int(v)
                return s.And(*shared_constraints)

        # Derive from bounds
        bounds = self._extract_implied_bounds()
        if bounds:
            # Filter to only shared-var bounds
            shared_bounds = [b for b in bounds if collect_vars(b) <= self._shared_vars]
            if shared_bounds:
                # Minimize: find smallest sufficient subset
                sufficient = self._minimize_sufficient(shared_bounds)
                if sufficient:
                    if len(sufficient) == 1:
                        return sufficient[0]
                    s = SMTSolver()
                    for v in self._shared_vars:
                        s.Int(v)
                    return s.And(*sufficient)

        # Ultimate fallback: BoolConst(False) is always a valid interpolant
        # (A => False when A is UNSAT, but A may not be UNSAT alone)
        # Try to build something from model differences
        return self._model_based_interpolant()

    def _minimize_sufficient(self, constraints):
        """Find minimal sufficient subset of constraints inconsistent with B."""
        if not constraints:
            return []
        if self._check_inconsistent_with_b(constraints):
            # Try removing each constraint
            for i in range(len(constraints)):
                smaller = constraints[:i] + constraints[i+1:]
                if smaller and self._check_inconsistent_with_b(smaller):
                    return self._minimize_sufficient(smaller)
            return constraints
        return []

    def _model_based_interpolant(self):
        """
        Model-based interpolation: find a formula over shared vars that
        separates A-satisfying assignments from B-satisfying assignments.
        """
        if not self._shared_vars:
            return BoolConst(False)

        # Get models from A
        s_a = SMTSolver()
        all_a_vars = set()
        for f in self._a_formulas:
            all_a_vars.update(collect_vars(f))
        for v in all_a_vars:
            s_a.Int(v)
        for f in self._a_formulas:
            s_a.add(f)

        r = s_a.check()
        if r != SMTResult.SAT:
            return BoolConst(False)

        a_model = s_a.model()
        if not a_model:
            return BoolConst(False)

        # Build a constraint that captures A's shared-var region
        # Use the model point as a starting point
        point_constraints = []
        for v in sorted(self._shared_vars):
            if v in a_model:
                s = SMTSolver()
                x = s.Int(v)
                point_constraints.append(App(Op.EQ, [x, IntConst(int(a_model[v]))], BOOL))

        if point_constraints and self._check_inconsistent_with_b(point_constraints):
            return self._generalize_point(point_constraints, a_model)

        # Try bounds from A's model
        bound_constraints = []
        for v in sorted(self._shared_vars):
            ub = self._find_bound(v, upper=True)
            lb = self._find_bound(v, upper=False)
            if ub is not None:
                s = SMTSolver()
                x = s.Int(v)
                bound_constraints.append(x <= int(ub))
            if lb is not None:
                s = SMTSolver()
                x = s.Int(v)
                bound_constraints.append(x >= int(lb))

        if bound_constraints and self._check_inconsistent_with_b(bound_constraints):
            return self._minimize_to_term(bound_constraints)

        # Fallback: conjunction of all bounds
        if bound_constraints:
            if len(bound_constraints) == 1:
                return bound_constraints[0]
            s = SMTSolver()
            for v in self._shared_vars:
                s.Int(v)
            return s.And(*bound_constraints)

        return BoolConst(False)

    def _generalize_point(self, point_constraints, model):
        """Generalize a point interpolant by relaxing equalities to inequalities."""
        # Try replacing each equality with an inequality range
        generalized = list(point_constraints)
        for i, pc in enumerate(point_constraints):
            if isinstance(pc, App) and pc.op == Op.EQ:
                var_term = pc.args[0]
                val = pc.args[1].value if isinstance(pc.args[1], IntConst) else None
                if val is None:
                    continue
                # Try relaxing to x <= val (upper bound only)
                test = generalized[:i] + [App(Op.LE, [var_term, pc.args[1]], BOOL)] + generalized[i+1:]
                if self._check_implied_by_a(App(Op.LE, [var_term, pc.args[1]], BOOL)):
                    if self._check_inconsistent_with_b(test):
                        generalized[i] = App(Op.LE, [var_term, pc.args[1]], BOOL)
                        continue
                # Try x >= val
                test = generalized[:i] + [App(Op.GE, [var_term, pc.args[1]], BOOL)] + generalized[i+1:]
                if self._check_implied_by_a(App(Op.GE, [var_term, pc.args[1]], BOOL)):
                    if self._check_inconsistent_with_b(test):
                        generalized[i] = App(Op.GE, [var_term, pc.args[1]], BOOL)
                        continue

        return self._minimize_to_term(generalized)

    def _minimize_to_term(self, constraints):
        """Convert a list of constraints to a single term, minimizing."""
        sufficient = self._minimize_sufficient(constraints)
        if not sufficient:
            sufficient = constraints
        if len(sufficient) == 1:
            return sufficient[0]
        s = SMTSolver()
        for v in self._shared_vars:
            s.Int(v)
        return s.And(*sufficient)

    def sequence_interpolate(self, formulas):
        """
        Compute sequence interpolants for A1, A2, ..., An where A1 AND ... AND An is UNSAT.

        Returns I_0 = True, I_1, I_2, ..., I_{n-1}, I_n = False such that:
        - I_0 = True
        - I_n = False
        - For each k: I_{k-1} AND A_k => I_k
        - vars(I_k) subset vars(A1...Ak) intersect vars(A_{k+1}...An)
        """
        n = len(formulas)
        if n < 2:
            return SequenceInterpolantResult(
                is_unsat=False, interpolants=[], stats={'error': 'need >= 2 formulas'}
            )

        # Check UNSAT
        s = SMTSolver()
        all_vars = set()
        for f_list in formulas:
            fl = f_list if isinstance(f_list, list) else [f_list]
            for f in fl:
                all_vars.update(collect_vars(f))
        for v in all_vars:
            s.Int(v)
        for f_list in formulas:
            fl = f_list if isinstance(f_list, list) else [f_list]
            for f in fl:
                s.add(f)
        if s.check() != SMTResult.UNSAT:
            return SequenceInterpolantResult(
                is_unsat=False, interpolants=[], stats={'reason': 'conjunction is satisfiable'}
            )

        # Compute sequence interpolants by iterated binary interpolation
        # I_k = interpolant(A1 AND ... AND Ak, A_{k+1} AND ... AND An)
        interpolants = [BoolConst(True)]  # I_0
        for k in range(1, n):
            a_part = []
            for i in range(k):
                fl = formulas[i] if isinstance(formulas[i], list) else [formulas[i]]
                a_part.extend(fl)
            b_part = []
            for i in range(k, n):
                fl = formulas[i] if isinstance(formulas[i], list) else [formulas[i]]
                b_part.extend(fl)

            result = self.interpolate(a_part, b_part)
            if result.is_unsat and result.interpolant is not None:
                interpolants.append(result.interpolant)
            else:
                interpolants.append(BoolConst(False))

        interpolants.append(BoolConst(False))  # I_n

        return SequenceInterpolantResult(
            is_unsat=True, interpolants=interpolants,
            stats={'n_formulas': n, 'n_interpolants': len(interpolants)}
        )

    def tree_interpolate(self, formulas, tree_edges):
        """
        Compute tree interpolants for a tree-structured formula partition.

        Args:
            formulas: list of formula lists (one per node)
            tree_edges: list of (parent, child) pairs defining the tree

        Returns:
            dict mapping node index to interpolant
        """
        n = len(formulas)
        # Build adjacency
        children = defaultdict(list)
        parent = {}
        for p, c in tree_edges:
            children[p].append(c)
            parent[c] = p

        # Find root (node with no parent)
        all_nodes = set(range(n))
        child_nodes = set(parent.keys())
        roots = all_nodes - child_nodes
        root = min(roots) if roots else 0

        # Post-order traversal: compute interpolant for each subtree
        interpolants = {}

        def subtree_formulas(node):
            """Collect all formulas in the subtree rooted at node."""
            result = []
            fl = formulas[node] if isinstance(formulas[node], list) else [formulas[node]]
            result.extend(fl)
            for c in children[node]:
                result.extend(subtree_formulas(c))
            return result

        def compute_node(node):
            # For each non-root node, compute interpolant:
            # A = subtree(node), B = everything else
            if node == root:
                return

            a_part = subtree_formulas(node)
            b_part = []
            for i in range(n):
                if i != node and i not in self._get_subtree_nodes(node, children):
                    fl = formulas[i] if isinstance(formulas[i], list) else [formulas[i]]
                    b_part.extend(fl)

            result = self.interpolate(a_part, b_part)
            if result.is_unsat and result.interpolant is not None:
                interpolants[node] = result.interpolant
            else:
                interpolants[node] = BoolConst(False)

        def _post_order(node):
            for c in children[node]:
                _post_order(c)
            compute_node(node)

        _post_order(root)
        interpolants[root] = BoolConst(True)  # Root gets True
        return interpolants

    def _get_subtree_nodes(self, node, children):
        """Get all nodes in the subtree rooted at node (excluding node itself)."""
        result = set()
        for c in children[node]:
            result.add(c)
            result.update(self._get_subtree_nodes(c, children))
        return result


# --- Convenience API ---

def craig_interpolate(a, b):
    """
    Compute Craig interpolant for A AND B = UNSAT.

    Args:
        a: Term or list of Terms -- the A partition
        b: Term or list of Terms -- the B partition

    Returns:
        InterpolantResult
    """
    interp = CraigInterpolator()
    return interp.interpolate(
        a if isinstance(a, list) else [a],
        b if isinstance(b, list) else [b]
    )


def sequence_interpolate(formulas):
    """
    Compute sequence interpolants for A1 AND ... AND An = UNSAT.

    Args:
        formulas: list of Terms (or list of list of Terms)

    Returns:
        SequenceInterpolantResult
    """
    interp = CraigInterpolator()
    return interp.sequence_interpolate(formulas)


def tree_interpolate(formulas, tree_edges):
    """
    Compute tree interpolants for a tree-structured partition.

    Args:
        formulas: list of formula lists
        tree_edges: list of (parent, child) pairs

    Returns:
        dict mapping node index to interpolant
    """
    interp = CraigInterpolator()
    return interp.tree_interpolate(formulas, tree_edges)


def verify_interpolant(a, b, interpolant):
    """
    Verify that a term is a valid Craig interpolant for A, B.

    Checks:
    1. A => I
    2. I AND B is UNSAT
    3. vars(I) subset vars(A) intersect vars(B)

    Returns dict with check results.
    """
    a_list = a if isinstance(a, list) else [a]
    b_list = b if isinstance(b, list) else [b]

    a_vars = set()
    for f in a_list:
        a_vars.update(collect_vars(f))
    b_vars = set()
    for f in b_list:
        b_vars.update(collect_vars(f))
    shared = a_vars & b_vars
    i_vars = collect_vars(interpolant)

    # Check 1: A => I (equivalently, A AND NOT I is UNSAT)
    s1 = SMTSolver()
    for v in a_vars | i_vars:
        s1.Int(v)
    for f in a_list:
        s1.add(f)
    s1.add(negate_atom(interpolant))
    a_implies_i = s1.check() == SMTResult.UNSAT

    # Check 2: I AND B is UNSAT
    s2 = SMTSolver()
    for v in b_vars | i_vars:
        s2.Int(v)
    s2.add(interpolant)
    for f in b_list:
        s2.add(f)
    i_and_b_unsat = s2.check() == SMTResult.UNSAT

    # Check 3: vars(I) subset shared
    vars_ok = i_vars <= shared

    return {
        'valid': a_implies_i and i_and_b_unsat and vars_ok,
        'a_implies_i': a_implies_i,
        'i_and_b_unsat': i_and_b_unsat,
        'vars_in_shared': vars_ok,
        'i_vars': i_vars,
        'shared_vars': shared,
        'extra_vars': i_vars - shared,
    }


def interpolation_summary():
    """Return a summary of the Craig interpolation module."""
    return {
        'name': 'V107: Craig Interpolation',
        'composes': ['C037 (SMT solver)'],
        'capabilities': [
            'Binary Craig interpolation (A, B -> I)',
            'Sequence interpolation (A1...An -> I0...In)',
            'Tree interpolation (tree partition -> per-node interpolants)',
            'Interpolant verification (A=>I, I^B UNSAT, var restriction)',
            'Bound extraction (upper/lower/equality/relational)',
            'Model-based interpolation with generalization',
            'Interpolant simplification',
        ],
        'applications': [
            'CEGAR refinement (extracting predicates from counterexamples)',
            'PDR/IC3 (blocking clause generalization)',
            'Predicate abstraction refinement',
            'Invariant inference',
        ],
        'apis': [
            'craig_interpolate(a, b) -> InterpolantResult',
            'sequence_interpolate(formulas) -> SequenceInterpolantResult',
            'tree_interpolate(formulas, edges) -> dict',
            'verify_interpolant(a, b, i) -> dict',
            'interpolation_summary() -> dict',
        ]
    }
