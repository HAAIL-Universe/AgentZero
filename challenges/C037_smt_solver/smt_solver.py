"""
C037: SMT Solver -- Satisfiability Modulo Theories
Composes C035 (SAT Solver) with a Simplex-based theory solver for
Linear Integer Arithmetic (LIA).

Architecture: DPLL(T)
- Boolean backbone: C035 SAT solver handles propositional structure
- Theory solver: Simplex algorithm checks integer arithmetic feasibility
- Integration: theory lemmas fed back as clauses to SAT solver

Capabilities:
- Linear integer arithmetic: x + 2*y <= 5, x >= 0, x == y + 1
- Boolean combinations of arithmetic constraints
- Uninterpreted functions with congruence closure
- Incremental solving with push/pop
- Model generation (satisfying assignment for both Bool and Int)
- UNSAT core extraction
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum
from collections import defaultdict
from fractions import Fraction
import copy

# Import C035 SAT solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C035_sat_solver'))
from sat_solver import Solver as SATSolver, SolverResult as SATResult, Literal


# --- Core Types ---

class SMTResult(Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


class SortKind(Enum):
    BOOL = "Bool"
    INT = "Int"
    REAL = "Real"


class Op(Enum):
    # Arithmetic
    ADD = "+"
    SUB = "-"
    MUL = "*"
    NEG = "neg"
    # Comparison
    LE = "<="
    LT = "<"
    GE = ">="
    GT = ">"
    EQ = "=="
    NEQ = "!="
    # Boolean
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "=>"
    IFF = "<=>"
    # Misc
    ITE = "ite"


@dataclass(frozen=True)
class Sort:
    kind: SortKind

    def __repr__(self):
        return self.kind.value

BOOL = Sort(SortKind.BOOL)
INT = Sort(SortKind.INT)
REAL = Sort(SortKind.REAL)


class Term:
    """Base class for SMT terms."""
    _counter = 0

    def __init__(self, sort: Sort):
        Term._counter += 1
        self.id = Term._counter
        self.sort = sort

    def __add__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.ADD, [self, other], INT)

    def __radd__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.ADD, [other, self], INT)

    def __sub__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.SUB, [self, other], INT)

    def __rsub__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.SUB, [other, self], INT)

    def __mul__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.MUL, [self, other], INT)

    def __rmul__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.MUL, [other, self], INT)

    def __neg__(self):
        return App(Op.NEG, [self], INT)

    def __le__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.LE, [self, other], BOOL)

    def __lt__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.LT, [self, other], BOOL)

    def __ge__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.GE, [self, other], BOOL)

    def __gt__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return App(Op.GT, [self, other], BOOL)

    def __eq__(self, other):
        if other is None or not isinstance(other, Term):
            return NotImplemented
        return App(Op.EQ, [self, other], BOOL)

    def __ne__(self, other):
        if other is None or not isinstance(other, Term):
            return NotImplemented
        return App(Op.NEQ, [self, other], BOOL)

    def __hash__(self):
        return hash(self.id)


class Var(Term):
    """A named variable."""
    def __init__(self, name: str, sort: Sort):
        super().__init__(sort)
        self.name = name

    def __repr__(self):
        return self.name


class IntConst(Term):
    """An integer constant."""
    def __init__(self, value: int):
        super().__init__(INT)
        self.value = value

    def __repr__(self):
        return str(self.value)


class BoolConst(Term):
    """A boolean constant."""
    def __init__(self, value: bool):
        super().__init__(BOOL)
        self.value = value

    def __repr__(self):
        return str(self.value).lower()


class App(Term):
    """Function application / operation."""
    def __init__(self, op: Op, args: list, sort: Sort):
        super().__init__(sort)
        self.op = op
        self.args = args

    def __repr__(self):
        if self.op == Op.NEG:
            return f"(-{self.args[0]})"
        if len(self.args) == 2:
            return f"({self.args[0]} {self.op.value} {self.args[1]})"
        return f"({self.op.value} {' '.join(str(a) for a in self.args)})"


class FuncDecl:
    """Uninterpreted function declaration."""
    def __init__(self, name: str, domain: list, range_sort: Sort):
        self.name = name
        self.domain = domain
        self.range_sort = range_sort

    def __call__(self, *args):
        return UFApp(self, list(args), self.range_sort)


class UFApp(Term):
    """Uninterpreted function application."""
    def __init__(self, func: FuncDecl, args: list, sort: Sort):
        super().__init__(sort)
        self.func = func
        self.args = args

    def __repr__(self):
        arg_str = ', '.join(str(a) for a in self.args)
        return f"{self.func.name}({arg_str})"


# --- Linear Arithmetic Representation ---

@dataclass
class LinearExpr:
    """Represents c0 + c1*x1 + c2*x2 + ... using Fraction coefficients."""
    coeffs: dict = field(default_factory=dict)  # var_name -> Fraction
    const: Fraction = field(default_factory=lambda: Fraction(0))

    def __add__(self, other):
        if isinstance(other, (int, Fraction)):
            return LinearExpr(dict(self.coeffs), self.const + Fraction(other))
        result = LinearExpr(dict(self.coeffs), self.const + other.const)
        for v, c in other.coeffs.items():
            result.coeffs[v] = result.coeffs.get(v, Fraction(0)) + c
        # Remove zeros
        result.coeffs = {v: c for v, c in result.coeffs.items() if c != 0}
        return result

    def __sub__(self, other):
        if isinstance(other, (int, Fraction)):
            return LinearExpr(dict(self.coeffs), self.const - Fraction(other))
        result = LinearExpr(dict(self.coeffs), self.const - other.const)
        for v, c in other.coeffs.items():
            result.coeffs[v] = result.coeffs.get(v, Fraction(0)) - c
        result.coeffs = {v: c for v, c in result.coeffs.items() if c != 0}
        return result

    def __mul__(self, scalar):
        s = Fraction(scalar)
        result = LinearExpr({v: c * s for v, c in self.coeffs.items()}, self.const * s)
        result.coeffs = {v: c for v, c in result.coeffs.items() if c != 0}
        return result

    def __neg__(self):
        return self * Fraction(-1)

    def __repr__(self):
        parts = []
        for v, c in sorted(self.coeffs.items()):
            if c == 1:
                parts.append(v)
            elif c == -1:
                parts.append(f"-{v}")
            else:
                parts.append(f"{c}*{v}")
        if self.const != 0 or not parts:
            parts.append(str(self.const))
        return " + ".join(parts)

    def is_constant(self):
        return len(self.coeffs) == 0

    def evaluate(self, assignment: dict) -> Fraction:
        result = self.const
        for v, c in self.coeffs.items():
            if v not in assignment:
                raise ValueError(f"Variable {v} not in assignment")
            result += c * Fraction(assignment[v])
        return result


class ConstraintKind(Enum):
    LE = "<="   # expr <= 0
    EQ = "=="   # expr == 0
    LT = "<"    # expr < 0 (for integers: expr <= -1)


@dataclass
class Constraint:
    """A linear constraint: expr <= 0 or expr == 0."""
    expr: LinearExpr
    kind: ConstraintKind
    atom_id: int = 0  # corresponding Boolean variable in SAT solver

    def __repr__(self):
        return f"{self.expr} {self.kind.value} 0"


# --- Simplex Theory Solver ---

class SimplexSolver:
    """
    Simplex-based theory solver for linear arithmetic.
    Uses the General Simplex algorithm (Dutertre & de Moura, 2006).

    Each constraint ax + by + ... <= c is converted to a slack variable:
        s = c - ax - by - ...  (s >= 0)

    The tableau tracks: slack = const - sum(coeff * original_var)
    Bounds on original variables come from the constraints.
    """

    def __init__(self):
        self.vars = set()            # all variable names
        self.basic_vars = set()      # slack/basic variables
        self.nonbasic_vars = set()   # original/nonbasic variables
        self.tableau = {}            # basic_var -> {nonbasic_var: coeff, '__const': val}
        self.lower = {}              # var -> Fraction (lower bound)
        self.upper = {}              # var -> Fraction (upper bound)
        self.assignment = {}         # var -> Fraction (current assignment)
        self._slack_count = 0
        self._explanation = {}       # var_bound -> set of constraint atom_ids
        self._conflict = None        # set of atom_ids explaining conflict

    def reset(self):
        self.__init__()

    def _new_slack(self) -> str:
        self._slack_count += 1
        return f"__s{self._slack_count}"

    def add_variable(self, name: str):
        if name not in self.vars:
            self.vars.add(name)
            self.nonbasic_vars.add(name)
            self.assignment[name] = Fraction(0)
            self.lower[name] = None  # no bound
            self.upper[name] = None

    def add_constraint(self, expr: LinearExpr, kind: ConstraintKind, atom_id: int = 0):
        """Add a linear constraint. Returns True if immediately feasible."""
        # Ensure all variables exist
        for v in expr.coeffs:
            self.add_variable(v)

        if kind == ConstraintKind.EQ:
            # expr == 0 becomes expr <= 0 AND -expr <= 0
            self.add_constraint(expr, ConstraintKind.LE, atom_id)
            self.add_constraint(-expr, ConstraintKind.LE, atom_id)
            return

        if kind == ConstraintKind.LT:
            # For integers: expr < 0 means expr <= -1
            new_expr = LinearExpr(dict(expr.coeffs), expr.const + Fraction(1))
            self.add_constraint(new_expr, ConstraintKind.LE, atom_id)
            return

        # Now handle expr <= 0
        # Introduce slack: s = -expr, with s >= 0
        slack = self._new_slack()
        self.vars.add(slack)
        self.basic_vars.add(slack)
        self.lower[slack] = Fraction(0)
        self.upper[slack] = None

        # s = -expr.const - sum(expr.coeffs[v] * v)
        row = {'__const': -expr.const}
        for v, c in expr.coeffs.items():
            row[v] = -c

        self.tableau[slack] = row

        # Compute initial assignment for slack
        val = row['__const']
        for v, c in row.items():
            if v == '__const':
                continue
            val += c * self.assignment.get(v, Fraction(0))
        self.assignment[slack] = val

        # Track explanation
        self._explanation[(slack, 'lower')] = {atom_id}

    def _value(self, var: str) -> Fraction:
        return self.assignment.get(var, Fraction(0))

    def _update_nonbasic(self, var: str, new_val: Fraction):
        """Update a nonbasic variable and adjust all basic variables."""
        delta = new_val - self.assignment[var]
        self.assignment[var] = new_val
        for basic_var, row in self.tableau.items():
            if var in row:
                self.assignment[basic_var] += row[var] * delta

    def _pivot(self, basic: str, nonbasic: str):
        """Pivot: swap a basic and nonbasic variable."""
        row = self.tableau[basic]
        coeff = row[nonbasic]
        if coeff == 0:
            raise ValueError(f"Cannot pivot: coefficient of {nonbasic} in {basic} is 0")

        # New row for nonbasic (which becomes basic):
        # nonbasic = (1/coeff) * (basic - const - sum(other_coeff * other_var))
        # But in tableau form: basic = const + sum(c_i * nb_i)
        # So: nonbasic = (basic - const - sum(c_j * nb_j for j != nonbasic)) / coeff
        # New row: nonbasic = -const/coeff + (1/coeff)*basic - sum(c_j/coeff * nb_j)

        new_row = {}
        inv_coeff = Fraction(1) / coeff
        new_row['__const'] = -row['__const'] * inv_coeff

        # The old basic variable becomes nonbasic in the new row
        new_row[basic] = inv_coeff

        for v, c in row.items():
            if v == '__const' or v == nonbasic:
                continue
            new_row[v] = -c * inv_coeff

        # Remove old row, update other rows
        del self.tableau[basic]

        for other_basic, other_row in self.tableau.items():
            if nonbasic in other_row:
                c_nb = other_row[nonbasic]
                # Substitute nonbasic = new_row
                del other_row[nonbasic]
                other_row['__const'] = other_row.get('__const', Fraction(0)) + c_nb * new_row['__const']
                for v, c in new_row.items():
                    if v == '__const':
                        continue
                    other_row[v] = other_row.get(v, Fraction(0)) + c_nb * c
                # Clean zeros
                to_del = [v for v, c in other_row.items() if v != '__const' and c == 0]
                for v in to_del:
                    del other_row[v]

        self.tableau[nonbasic] = new_row

        self.basic_vars.discard(basic)
        self.basic_vars.add(nonbasic)
        self.nonbasic_vars.discard(nonbasic)
        self.nonbasic_vars.add(basic)

    def check(self) -> bool:
        """Run Simplex to check feasibility. Returns True if feasible.
        Uses Bland's rule (smallest index) to prevent cycling."""
        self._conflict = None
        max_iterations = 5000

        for _ in range(max_iterations):
            # Find a violated basic variable (smallest name for Bland's rule)
            violated = None
            for bv in sorted(self.basic_vars):
                val = self.assignment[bv]
                lb = self.lower.get(bv)
                ub = self.upper.get(bv)
                if lb is not None and val < lb:
                    violated = (bv, 'lower')
                    break
                if ub is not None and val > ub:
                    violated = (bv, 'upper')
                    break

            if violated is None:
                return True  # All bounds satisfied

            bv, bound_type = violated
            row = self.tableau.get(bv, {})

            if bound_type == 'lower':
                # Need to increase bv. Find nonbasic var to pivot with.
                pivot_var = None
                for nb in sorted(row.keys()):
                    if nb == '__const':
                        continue
                    coeff = row[nb]
                    if coeff > 0 and (self.upper.get(nb) is None or self.assignment[nb] < self.upper[nb]):
                        pivot_var = nb
                        break
                    if coeff < 0 and (self.lower.get(nb) is None or self.assignment[nb] > self.lower[nb]):
                        pivot_var = nb
                        break

                if pivot_var is None:
                    self._collect_conflict(bv, bound_type, row)
                    return False

                # Update nonbasic to fix bv, then pivot
                coeff = row[pivot_var]
                deficit = self.lower[bv] - self.assignment[bv]
                delta = deficit / coeff
                new_val = self.assignment[pivot_var] + delta
                self._update_nonbasic(pivot_var, new_val)
                self._pivot(bv, pivot_var)

            else:  # upper
                pivot_var = None
                for nb in sorted(row.keys()):
                    if nb == '__const':
                        continue
                    coeff = row[nb]
                    if coeff < 0 and (self.upper.get(nb) is None or self.assignment[nb] < self.upper[nb]):
                        pivot_var = nb
                        break
                    if coeff > 0 and (self.lower.get(nb) is None or self.assignment[nb] > self.lower[nb]):
                        pivot_var = nb
                        break

                if pivot_var is None:
                    self._collect_conflict(bv, bound_type, row)
                    return False

                coeff = row[pivot_var]
                excess = self.assignment[bv] - self.upper[bv]
                delta = -excess / coeff
                new_val = self.assignment[pivot_var] + delta
                self._update_nonbasic(pivot_var, new_val)
                self._pivot(bv, pivot_var)

        return False  # Exceeded iterations

    def _collect_conflict(self, bv: str, bound_type: str, row: dict):
        """Collect atom IDs involved in the conflict."""
        conflict = set()
        # The bound on bv
        key = (bv, bound_type)
        if key in self._explanation:
            conflict.update(self._explanation[key])
        # Bounds of nonbasic vars that prevent fixing bv
        for nb in row:
            if nb == '__const':
                continue
            coeff = row[nb]
            if bound_type == 'lower':
                if coeff > 0:
                    k = (nb, 'upper')
                else:
                    k = (nb, 'lower')
            else:
                if coeff < 0:
                    k = (nb, 'upper')
                else:
                    k = (nb, 'lower')
            if k in self._explanation:
                conflict.update(self._explanation[k])
        self._conflict = conflict

    def get_conflict(self) -> Optional[set]:
        return self._conflict

    def get_value(self, var: str) -> Optional[Fraction]:
        if var in self.assignment:
            return self.assignment[var]
        return None

    def save_state(self):
        """Save state for backtracking."""
        return {
            'vars': set(self.vars),
            'basic_vars': set(self.basic_vars),
            'nonbasic_vars': set(self.nonbasic_vars),
            'tableau': {k: dict(v) for k, v in self.tableau.items()},
            'lower': dict(self.lower),
            'upper': dict(self.upper),
            'assignment': dict(self.assignment),
            'slack_count': self._slack_count,
            'explanation': {k: set(v) for k, v in self._explanation.items()},
        }

    def restore_state(self, state):
        """Restore state for backtracking."""
        self.vars = state['vars']
        self.basic_vars = state['basic_vars']
        self.nonbasic_vars = state['nonbasic_vars']
        self.tableau = state['tableau']
        self.lower = state['lower']
        self.upper = state['upper']
        self.assignment = state['assignment']
        self._slack_count = state['slack_count']
        self._explanation = state['explanation']
        self._conflict = None


# --- Congruence Closure (for Uninterpreted Functions) ---

class CongruenceClosure:
    """Union-Find based congruence closure for equality reasoning."""

    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.classes = {}  # representative -> set of term ids
        self.func_apps = {}  # (func_name, arg_rep_tuple) -> term_id
        self.term_info = {}  # term_id -> (func_name, [arg_ids]) or None
        self.merge_history = []
        self._diseq = []  # list of (term_a_id, term_b_id, atom_id)
        self._eq_atoms = []  # list of (term_a_id, term_b_id, atom_id)

    def add_term(self, term_id: int, func_name: str = None, arg_ids: list = None):
        if term_id in self.parent:
            return
        self.parent[term_id] = term_id
        self.rank[term_id] = 0
        self.classes[term_id] = {term_id}
        if func_name is not None:
            self.term_info[term_id] = (func_name, arg_ids or [])
            # Check for congruence
            arg_reps = tuple(self.find(a) for a in arg_ids)
            key = (func_name, arg_reps)
            if key in self.func_apps:
                # Congruence: merge
                other = self.func_apps[key]
                self.merge(term_id, other)
            else:
                self.func_apps[key] = term_id
        else:
            self.term_info[term_id] = None

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.add_term(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def merge(self, a: int, b: int) -> bool:
        """Merge equivalence classes. Returns False if inconsistent (disequality violation)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True

        self.merge_history.append((ra, rb, dict(self.func_apps)))

        # Union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        self.classes[ra] = self.classes.get(ra, {ra}) | self.classes.get(rb, {rb})
        if rb in self.classes:
            del self.classes[rb]

        # Update func_apps for congruence propagation
        # Collect all function applications involving terms in rb's old class
        pending_merges = []
        new_apps = {}
        for key, tid in list(self.func_apps.items()):
            fname, arg_reps = key
            new_reps = tuple(self.find(a) for a in arg_reps)
            new_key = (fname, new_reps)
            if new_key != key:
                del self.func_apps[key]
                if new_key in self.func_apps:
                    pending_merges.append((tid, self.func_apps[new_key]))
                else:
                    new_apps[new_key] = tid
            elif new_key in new_apps:
                pending_merges.append((tid, new_apps[new_key]))

        self.func_apps.update(new_apps)

        for pa, pb in pending_merges:
            if not self.merge(pa, pb):
                return False

        # Check disequalities
        for da, db, _ in self._diseq:
            if self.find(da) == self.find(db):
                return False

        return True

    def are_equal(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def add_disequality(self, a: int, b: int, atom_id: int):
        self._diseq.append((a, b, atom_id))

    def add_equality(self, a: int, b: int, atom_id: int):
        self._eq_atoms.append((a, b, atom_id))

    def check(self) -> bool:
        """Check consistency of all equalities and disequalities."""
        for a, b, _ in self._diseq:
            if self.find(a) == self.find(b):
                return False
        return True

    def save_state(self):
        return {
            'parent': dict(self.parent),
            'rank': dict(self.rank),
            'classes': {k: set(v) for k, v in self.classes.items()},
            'func_apps': dict(self.func_apps),
            'term_info': dict(self.term_info),
            'merge_history_len': len(self.merge_history),
            'diseq': list(self._diseq),
            'eq_atoms': list(self._eq_atoms),
        }

    def restore_state(self, state):
        self.parent = state['parent']
        self.rank = state['rank']
        self.classes = state['classes']
        self.func_apps = state['func_apps']
        self.term_info = state['term_info']
        self.merge_history = self.merge_history[:state['merge_history_len']]
        self._diseq = state['diseq']
        self._eq_atoms = state['eq_atoms']


# --- SMT Solver ---

class SMTSolver:
    """
    DPLL(T) SMT Solver combining SAT solving with theory reasoning.

    Usage:
        s = SMTSolver()
        x = s.Int('x')
        y = s.Int('y')
        s.add(x + y <= 10)
        s.add(x >= 0)
        s.add(y >= 0)
        s.add(x + y >= 5)
        result = s.check()
        if result == SMTResult.SAT:
            print(s.model())
    """

    def __init__(self):
        self.sat = SATSolver()
        self.simplex = SimplexSolver()
        self.cc = CongruenceClosure()

        self._vars = {}              # name -> Var term
        self._atoms = {}             # atom_id (SAT var) -> Constraint or equality info
        self._term_to_atom = {}      # term.id -> atom_id
        self._atom_to_term = {}      # atom_id -> term
        self._assertions = []        # list of Term (top-level assertions)
        self._model = None
        self._result = SMTResult.UNKNOWN
        self._scope_stack = []       # for push/pop
        self._unsat_core = None

        # Tseitin mapping
        self._tseitin_cache = {}     # term.id -> SAT literal int

    # --- Variable creation ---

    def Int(self, name: str) -> Var:
        if name in self._vars:
            return self._vars[name]
        v = Var(name, INT)
        self._vars[name] = v
        self.simplex.add_variable(name)
        return v

    def Bool(self, name: str) -> Var:
        if name in self._vars:
            return self._vars[name]
        v = Var(name, BOOL)
        self._vars[name] = v
        return v

    def Real(self, name: str) -> Var:
        if name in self._vars:
            return self._vars[name]
        v = Var(name, REAL)
        self._vars[name] = v
        self.simplex.add_variable(name)
        return v

    def Function(self, name: str, *sorts) -> FuncDecl:
        """Declare an uninterpreted function. Last sort is range, rest are domain."""
        domain = list(sorts[:-1])
        range_sort = sorts[-1]
        return FuncDecl(name, domain, range_sort)

    # --- Convenience constructors ---

    def IntVal(self, v: int) -> IntConst:
        return IntConst(v)

    def BoolVal(self, v: bool) -> BoolConst:
        return BoolConst(v)

    def And(self, *args) -> Term:
        flat = list(args)
        if len(flat) == 0:
            return BoolConst(True)
        if len(flat) == 1:
            return flat[0]
        return App(Op.AND, flat, BOOL)

    def Or(self, *args) -> Term:
        flat = list(args)
        if len(flat) == 0:
            return BoolConst(False)
        if len(flat) == 1:
            return flat[0]
        return App(Op.OR, flat, BOOL)

    def Not(self, a: Term) -> Term:
        return App(Op.NOT, [a], BOOL)

    def Implies(self, a: Term, b: Term) -> Term:
        return App(Op.IMPLIES, [a, b], BOOL)

    def Iff(self, a: Term, b: Term) -> Term:
        return App(Op.IFF, [a, b], BOOL)

    def If(self, cond: Term, then_val: Term, else_val: Term) -> Term:
        sort = then_val.sort
        return App(Op.ITE, [cond, then_val, else_val], sort)

    def Distinct(self, *args) -> Term:
        """All arguments must be pairwise distinct."""
        constraints = []
        args_list = list(args)
        for i in range(len(args_list)):
            for j in range(i + 1, len(args_list)):
                constraints.append(args_list[i] != args_list[j])
        return self.And(*constraints)

    # --- Assertion ---

    def add(self, constraint: Term):
        """Add a constraint to the solver."""
        self._assertions.append(constraint)

    def push(self):
        """Save solver state."""
        self._scope_stack.append({
            'assertions_len': len(self._assertions),
            'simplex_state': self.simplex.save_state(),
            'cc_state': self.cc.save_state(),
            'tseitin_cache': dict(self._tseitin_cache),
            'atoms': dict(self._atoms),
            'term_to_atom': dict(self._term_to_atom),
            'atom_to_term': dict(self._atom_to_term),
        })

    def pop(self):
        """Restore solver state."""
        if not self._scope_stack:
            raise ValueError("No scope to pop")
        state = self._scope_stack.pop()
        self._assertions = self._assertions[:state['assertions_len']]
        self.simplex.restore_state(state['simplex_state'])
        self.cc.restore_state(state['cc_state'])
        self._tseitin_cache = state['tseitin_cache']
        self._atoms = state['atoms']
        self._term_to_atom = state['term_to_atom']
        self._atom_to_term = state['atom_to_term']
        # Need fresh SAT solver
        self.sat = SATSolver()
        self._result = SMTResult.UNKNOWN
        self._model = None

    # --- Term to Linear Expression ---

    def _to_linear(self, term: Term) -> LinearExpr:
        """Convert an arithmetic term to a LinearExpr."""
        if isinstance(term, IntConst):
            return LinearExpr(const=Fraction(term.value))
        if isinstance(term, Var) and term.sort in (INT, REAL):
            return LinearExpr(coeffs={term.name: Fraction(1)})
        if isinstance(term, App):
            if term.op == Op.ADD:
                return self._to_linear(term.args[0]) + self._to_linear(term.args[1])
            if term.op == Op.SUB:
                return self._to_linear(term.args[0]) - self._to_linear(term.args[1])
            if term.op == Op.MUL:
                left = self._to_linear(term.args[0])
                right = self._to_linear(term.args[1])
                # One side must be constant for linear arithmetic
                if left.is_constant():
                    return right * left.const
                if right.is_constant():
                    return left * right.const
                raise ValueError("Non-linear multiplication not supported in LIA")
            if term.op == Op.NEG:
                return -self._to_linear(term.args[0])
        raise ValueError(f"Cannot convert {term} to linear expression")

    # --- Tseitin Transformation ---

    def _encode(self, term: Term) -> int:
        """Encode a term as a SAT literal using Tseitin transformation.
        Returns a SAT literal (positive int = true, negative = negated)."""

        if term.id in self._tseitin_cache:
            return self._tseitin_cache[term.id]

        if isinstance(term, BoolConst):
            v = self.sat.new_var()
            if term.value:
                self.sat.add_clause([v])
            else:
                self.sat.add_clause([-v])
            self._tseitin_cache[term.id] = v
            return v

        if isinstance(term, Var) and term.sort == BOOL:
            v = self.sat.new_var()
            self._tseitin_cache[term.id] = v
            self._atom_to_term[v] = term
            return v

        if isinstance(term, App):
            if term.sort == BOOL and term.op in (Op.LE, Op.LT, Op.GE, Op.GT, Op.EQ, Op.NEQ):
                return self._encode_comparison(term)

            if term.op == Op.AND:
                v = self.sat.new_var()
                arg_lits = [self._encode(a) for a in term.args]
                # v <=> (a1 AND a2 AND ...)
                # v => a_i  for all i:  (-v OR a_i)
                for al in arg_lits:
                    self.sat.add_clause([-v, al])
                # (a1 AND a2 AND ...) => v:  (-a1 OR -a2 OR ... OR v)
                self.sat.add_clause([-al for al in arg_lits] + [v])
                self._tseitin_cache[term.id] = v
                return v

            if term.op == Op.OR:
                v = self.sat.new_var()
                arg_lits = [self._encode(a) for a in term.args]
                # v <=> (a1 OR a2 OR ...)
                # v => (a1 OR a2 OR ...):  (-v OR a1 OR a2 OR ...)
                self.sat.add_clause([-v] + arg_lits)
                # a_i => v  for all i:  (-a_i OR v)
                for al in arg_lits:
                    self.sat.add_clause([-al, v])
                self._tseitin_cache[term.id] = v
                return v

            if term.op == Op.NOT:
                inner = self._encode(term.args[0])
                self._tseitin_cache[term.id] = -inner
                return -inner

            if term.op == Op.IMPLIES:
                # a => b  is  !a OR b
                a_lit = self._encode(term.args[0])
                b_lit = self._encode(term.args[1])
                v = self.sat.new_var()
                # v <=> (!a OR b)
                self.sat.add_clause([-v, -a_lit, b_lit])
                self.sat.add_clause([a_lit, v])
                self.sat.add_clause([-b_lit, v])
                self._tseitin_cache[term.id] = v
                return v

            if term.op == Op.IFF:
                a_lit = self._encode(term.args[0])
                b_lit = self._encode(term.args[1])
                v = self.sat.new_var()
                # v <=> (a <=> b) which is (a => b) AND (b => a)
                # v => (a => b): -v OR -a OR b
                self.sat.add_clause([-v, -a_lit, b_lit])
                # v => (b => a): -v OR -b OR a
                self.sat.add_clause([-v, -b_lit, a_lit])
                # (a <=> b) => v: need a,b same => v
                # (!a OR b) AND (!b OR a) => v
                # Equivalent: (a AND !b) OR (!a AND b) => !v
                self.sat.add_clause([a_lit, b_lit, v])
                self.sat.add_clause([-a_lit, -b_lit, v])
                self._tseitin_cache[term.id] = v
                return v

            if term.op == Op.ITE:
                cond_lit = self._encode(term.args[0])
                then_lit = self._encode(term.args[1])
                else_lit = self._encode(term.args[2])
                v = self.sat.new_var()
                # v <=> ITE(c, t, e)
                # c => (v <=> t)
                self.sat.add_clause([-cond_lit, -v, then_lit])
                self.sat.add_clause([-cond_lit, v, -then_lit])
                # !c => (v <=> e)
                self.sat.add_clause([cond_lit, -v, else_lit])
                self.sat.add_clause([cond_lit, v, -else_lit])
                self._tseitin_cache[term.id] = v
                return v

        # Equality/disequality between non-bool terms (UF)
        if isinstance(term, App) and term.op in (Op.EQ, Op.NEQ):
            return self._encode_uf_equality(term)

        raise ValueError(f"Cannot encode term: {term}")

    def _contains_ite(self, term: Term) -> bool:
        """Check if a term contains an ITE sub-expression."""
        if isinstance(term, App):
            if term.op == Op.ITE:
                return True
            return any(self._contains_ite(a) for a in term.args)
        return False

    def _encode_ite_comparison(self, term: App) -> int:
        """Encode a comparison containing ITE by case-splitting."""
        # Find the ITE and split: ITE(c, t, e) op rhs =>
        # (c AND (t op rhs)) OR (!c AND (e op rhs))
        # This handles ITE in either operand
        ite_term = self._find_ite(term)
        if ite_term is None:
            raise ValueError(f"No ITE found in {term}")

        cond = ite_term.args[0]
        then_val = ite_term.args[1]
        else_val = ite_term.args[2]

        # Create two versions of the comparison: one for each ITE branch
        then_term = self._substitute_ite(term, ite_term.id, then_val)
        else_term = self._substitute_ite(term, ite_term.id, else_val)

        cond_lit = self._encode(cond)
        then_lit = self._encode(then_term)
        else_lit = self._encode(else_term)

        v = self.sat.new_var()
        # v <=> ITE(cond, then_comparison, else_comparison)
        self.sat.add_clause([-cond_lit, -v, then_lit])
        self.sat.add_clause([-cond_lit, v, -then_lit])
        self.sat.add_clause([cond_lit, -v, else_lit])
        self.sat.add_clause([cond_lit, v, -else_lit])
        self._tseitin_cache[term.id] = v
        return v

    def _find_ite(self, term: Term) -> Optional[App]:
        """Find first ITE sub-expression in a term."""
        if isinstance(term, App):
            if term.op == Op.ITE:
                return term
            for arg in term.args:
                found = self._find_ite(arg)
                if found:
                    return found
        return None

    def _substitute_ite(self, term: Term, ite_id: int, replacement: Term) -> Term:
        """Replace an ITE sub-expression with its branch value."""
        if term.id == ite_id:
            return replacement
        if isinstance(term, App):
            new_args = [self._substitute_ite(a, ite_id, replacement) for a in term.args]
            if any(new_args[i] is not term.args[i] for i in range(len(new_args))):
                return App(term.op, new_args, term.sort)
        return term

    def _encode_comparison(self, term: App) -> int:
        """Encode an arithmetic comparison as a theory atom."""
        left = term.args[0]
        right = term.args[1]

        # Handle ITE in arithmetic comparisons
        if self._contains_ite(left) or self._contains_ite(right):
            return self._encode_ite_comparison(term)

        try:
            left_lin = self._to_linear(left)
            right_lin = self._to_linear(right)
        except ValueError:
            # Non-linear or UF -- handle as UF equality
            if term.op in (Op.EQ, Op.NEQ):
                return self._encode_uf_equality(term)
            raise

        # Create SAT variable for this atom
        atom_id = self.sat.new_var()

        if term.op == Op.LE:
            # left <= right  =>  left - right <= 0
            expr = left_lin - right_lin
            constraint = Constraint(expr, ConstraintKind.LE, atom_id)
        elif term.op == Op.LT:
            # left < right  =>  left - right < 0  =>  left - right <= -1 (integers)
            expr = left_lin - right_lin
            constraint = Constraint(expr, ConstraintKind.LT, atom_id)
        elif term.op == Op.GE:
            # left >= right  =>  right - left <= 0
            expr = right_lin - left_lin
            constraint = Constraint(expr, ConstraintKind.LE, atom_id)
        elif term.op == Op.GT:
            # left > right  =>  right - left < 0
            expr = right_lin - left_lin
            constraint = Constraint(expr, ConstraintKind.LT, atom_id)
        elif term.op == Op.EQ:
            # left == right  =>  left - right == 0
            expr = left_lin - right_lin
            constraint = Constraint(expr, ConstraintKind.EQ, atom_id)
        elif term.op == Op.NEQ:
            # left != right => (left < right) OR (left > right)
            # For integers: (left - right <= -1) OR (right - left <= -1)
            lt_term = App(Op.LT, [left, right], BOOL)
            gt_term = App(Op.GT, [left, right], BOOL)
            or_term = App(Op.OR, [lt_term, gt_term], BOOL)
            result = self._encode(or_term)
            self._tseitin_cache[term.id] = result
            return result

        self._atoms[atom_id] = constraint
        self._term_to_atom[term.id] = atom_id
        self._atom_to_term[atom_id] = term
        self._tseitin_cache[term.id] = atom_id
        return atom_id

    def _encode_uf_equality(self, term: App) -> int:
        """Encode UF equality/disequality."""
        atom_id = self.sat.new_var()

        left, right = term.args[0], term.args[1]
        self._register_uf_term(left)
        self._register_uf_term(right)

        if term.op == Op.EQ:
            self._atoms[atom_id] = ('uf_eq', left.id, right.id)
        elif term.op == Op.NEQ:
            eq_term = App(Op.EQ, [left, right], BOOL)
            eq_lit = self._encode_uf_equality(eq_term)
            self._tseitin_cache[term.id] = -eq_lit
            return -eq_lit

        self._term_to_atom[term.id] = atom_id
        self._atom_to_term[atom_id] = term
        self._tseitin_cache[term.id] = atom_id
        return atom_id

    def _register_uf_term(self, term: Term):
        """Register a term in the congruence closure."""
        if isinstance(term, UFApp):
            for arg in term.args:
                self._register_uf_term(arg)
            arg_ids = [a.id for a in term.args]
            self.cc.add_term(term.id, term.func.name, arg_ids)
        elif isinstance(term, (Var, IntConst)):
            self.cc.add_term(term.id)

    # --- DPLL(T) Main Loop ---

    def check(self) -> SMTResult:
        """Check satisfiability of all assertions."""
        self._model = None
        self._unsat_core = None

        # Fresh SAT solver for each check
        self.sat = SATSolver()
        self.simplex = SimplexSolver()
        self.cc = CongruenceClosure()
        self._tseitin_cache = {}
        self._atoms = {}
        self._term_to_atom = {}
        self._atom_to_term = {}

        # Re-register variables
        for name, var in self._vars.items():
            if var.sort in (INT, REAL):
                self.simplex.add_variable(name)

        # Encode all assertions
        top_lits = []
        for assertion in self._assertions:
            lit = self._encode(assertion)
            top_lits.append(lit)

        # Assert all top-level constraints
        for lit in top_lits:
            self.sat.add_clause([lit])

        # DPLL(T) loop
        result = self._dpll_t()
        self._result = result
        return result

    def _dpll_t(self) -> SMTResult:
        """DPLL(T) algorithm: iterate between SAT solving and theory checking."""
        max_iterations = 100

        for iteration in range(max_iterations):
            # Step 1: SAT solve
            sat_result = self.sat.solve()

            if sat_result == SATResult.UNSAT:
                return SMTResult.UNSAT

            if sat_result != SATResult.SAT:
                return SMTResult.UNKNOWN

            # Step 2: Extract Boolean model
            bool_model = self.sat.model()
            if bool_model is None:
                return SMTResult.UNKNOWN

            # Step 3: Check theory consistency
            theory_ok, conflict_clause = self._check_theory(bool_model)

            if theory_ok:
                # Build model
                self._build_model(bool_model)
                return SMTResult.SAT

            # Step 4: Add theory lemma (conflict clause) and retry
            if conflict_clause:
                self.sat.add_clause(conflict_clause)
                # Reset SAT solver state for re-solving
                # We need to re-solve with the new clause
                # Create fresh solver with all existing clauses
                old_clauses = [(c.literals, c.learned) for c in self.sat.clauses]
                self.sat = SATSolver()
                for lits, learned in old_clauses:
                    clause_ints = [int(l) for l in lits]
                    self.sat.add_clause(clause_ints)
            else:
                return SMTResult.UNKNOWN

        return SMTResult.UNKNOWN

    def _check_theory(self, bool_model: dict) -> tuple:
        """Check if the Boolean model is theory-consistent.
        Returns (is_consistent, conflict_clause_or_None)."""

        # Reset theory solvers
        simplex = SimplexSolver()
        cc = CongruenceClosure()

        # Re-register variables
        for name, var in self._vars.items():
            if var.sort in (INT, REAL):
                simplex.add_variable(name)

        active_atoms = []
        neq_checks = []  # (expr, atom_id) for negated equalities

        for atom_id, atom_info in self._atoms.items():
            if atom_id not in bool_model:
                continue
            is_true = bool_model[atom_id]

            if isinstance(atom_info, Constraint):
                if is_true:
                    # Constraint is active
                    simplex.add_constraint(atom_info.expr, atom_info.kind, atom_id)
                    active_atoms.append((atom_id, True))
                else:
                    # Negated constraint
                    neg_constraint = self._negate_constraint(atom_info)
                    if neg_constraint == 'NEQ':
                        # NOT (expr == 0) => expr != 0
                        # Store for post-check: after simplex, verify model satisfies NEQ
                        neq_checks.append((atom_info.expr, atom_id))
                        active_atoms.append((atom_id, False))
                    elif neg_constraint is not None:
                        simplex.add_constraint(neg_constraint.expr, neg_constraint.kind, atom_id)
                        active_atoms.append((atom_id, False))

            elif isinstance(atom_info, tuple) and atom_info[0] == 'uf_eq':
                _, left_id, right_id = atom_info
                # Register the actual terms from the original assertion
                if atom_id in self._atom_to_term:
                    orig = self._atom_to_term[atom_id]
                    if isinstance(orig, App):
                        for arg in orig.args:
                            self._register_uf_term_in_cc(cc, arg)
                else:
                    cc.add_term(left_id)
                    cc.add_term(right_id)
                if is_true:
                    if not cc.merge(left_id, right_id):
                        clause = [-atom_id]
                        return False, clause
                    active_atoms.append((atom_id, True))
                else:
                    cc.add_disequality(left_id, right_id, atom_id)
                    active_atoms.append((atom_id, False))

        # Check Simplex
        if not simplex.check():
            conflict = simplex.get_conflict()
            if conflict:
                # Build conflict clause: negation of the conjunction of active atoms
                clause = []
                for aid, was_true in active_atoms:
                    if aid in conflict:
                        clause.append(-aid if was_true else aid)
                if clause:
                    return False, clause
            # Fallback: negate all active atoms
            clause = [-aid if was_true else aid for aid, was_true in active_atoms]
            return False, clause if clause else None

        # Propagate arithmetic equalities to congruence closure
        # If x == y in the simplex model (via EQ constraint), tell the CC
        for atom_id, atom_info in self._atoms.items():
            if atom_id not in bool_model:
                continue
            if not bool_model[atom_id]:
                continue
            if isinstance(atom_info, Constraint) and atom_info.kind == ConstraintKind.EQ:
                # This equality is true. Check if it's a simple x == y
                expr = atom_info.expr
                if len(expr.coeffs) == 2 and expr.const == 0:
                    vars_in_expr = list(expr.coeffs.items())
                    v1_name, c1 = vars_in_expr[0]
                    v2_name, c2 = vars_in_expr[1]
                    if c1 == Fraction(1) and c2 == Fraction(-1):
                        # v1 - v2 == 0, so v1 == v2
                        if v1_name in self._vars and v2_name in self._vars:
                            t1 = self._vars[v1_name]
                            t2 = self._vars[v2_name]
                            cc.add_term(t1.id)
                            cc.add_term(t2.id)
                            cc.merge(t1.id, t2.id)
                    elif c1 == Fraction(-1) and c2 == Fraction(1):
                        if v1_name in self._vars and v2_name in self._vars:
                            t1 = self._vars[v1_name]
                            t2 = self._vars[v2_name]
                            cc.add_term(t1.id)
                            cc.add_term(t2.id)
                            cc.merge(t1.id, t2.id)

        # Check NEQ constraints against the simplex model
        for expr, atom_id in neq_checks:
            val = Fraction(0)
            all_assigned = True
            for v, c in expr.coeffs.items():
                sv = simplex.get_value(v)
                if sv is None:
                    all_assigned = False
                    break
                val += c * sv
            if all_assigned:
                val += expr.const
                if val == 0:
                    # Violates NEQ -- the current boolean assignment implies expr=0
                    # but we need expr != 0. Block this Boolean assignment.
                    # The conflict involves: the NEQ atom (false) plus all active
                    # constraints that force the value.
                    clause = [-aid if was_true else aid for aid, was_true in active_atoms]
                    return False, clause if clause else None

        # Check congruence closure
        if not cc.check():
            clause = [-aid if was_true else aid for aid, was_true in active_atoms]
            return False, clause if clause else None

        # Store theory assignment for model building
        self._theory_simplex = simplex
        self._theory_cc = cc
        return True, None

    def _register_uf_term_in_cc(self, cc: CongruenceClosure, term: Term):
        """Register a term and all its sub-terms in a congruence closure instance."""
        if term.id in cc.parent:
            return  # Already registered
        if isinstance(term, UFApp):
            for arg in term.args:
                self._register_uf_term_in_cc(cc, arg)
            arg_ids = [a.id for a in term.args]
            cc.add_term(term.id, term.func.name, arg_ids)
        elif isinstance(term, (Var, IntConst, BoolConst)):
            cc.add_term(term.id)
        elif isinstance(term, App):
            # For comparison apps, register sub-terms
            for arg in term.args:
                self._register_uf_term_in_cc(cc, arg)
            cc.add_term(term.id)

    def _negate_constraint(self, constraint: Constraint) -> Optional[Constraint]:
        """Negate a linear constraint. Returns None for disjunctive negations (NEQ)."""
        if constraint.kind == ConstraintKind.LE:
            # NOT (expr <= 0) => expr > 0 => -expr < 0 => -expr <= -1 (integers)
            neg_expr = -constraint.expr
            return Constraint(neg_expr, ConstraintKind.LT, constraint.atom_id)
        elif constraint.kind == ConstraintKind.LT:
            # NOT (expr < 0) => expr >= 0 => -expr <= 0
            neg_expr = -constraint.expr
            return Constraint(neg_expr, ConstraintKind.LE, constraint.atom_id)
        elif constraint.kind == ConstraintKind.EQ:
            # NOT (expr == 0) => expr != 0 => expr > 0 OR expr < 0
            # Disjunctive -- return special marker
            return 'NEQ'
        return None

    def _build_model(self, bool_model: dict):
        """Build the satisfying model from SAT + theory assignments."""
        self._model = {}

        for name, var in self._vars.items():
            if var.sort in (INT, REAL):
                val = self._theory_simplex.get_value(name)
                if val is not None:
                    if var.sort == INT:
                        # Round to integer if needed
                        self._model[name] = int(val)
                    else:
                        self._model[name] = float(val)
                else:
                    self._model[name] = 0
            elif var.sort == BOOL:
                # Find SAT variable
                if var.id in self._tseitin_cache:
                    sat_var = self._tseitin_cache[var.id]
                    if abs(sat_var) in bool_model:
                        self._model[name] = bool_model[abs(sat_var)] if sat_var > 0 else not bool_model[abs(sat_var)]
                    else:
                        self._model[name] = False
                else:
                    self._model[name] = False

    # --- Model access ---

    def model(self) -> Optional[dict]:
        """Get the satisfying assignment after a SAT result."""
        return self._model

    def result(self) -> SMTResult:
        return self._result

    def unsat_core(self) -> Optional[list]:
        return self._unsat_core

    # --- String representation ---

    def __repr__(self):
        return f"SMTSolver(assertions={len(self._assertions)}, vars={len(self._vars)})"


# --- Convenience API (Z3-like) ---

def Int(name: str) -> tuple:
    """For use outside a solver context -- returns (solver, var)."""
    s = SMTSolver()
    return s.Int(name)


def Ints(names: str) -> list:
    """Create multiple int variables. Returns list of vars.
    Must be used with a solver: s = SMTSolver(); x, y = s.Ints('x y')"""
    pass  # Implemented as method


# Add Ints method to SMTSolver
def _ints(self, names: str) -> list:
    return [self.Int(n) for n in names.split()]

def _bools(self, names: str) -> list:
    return [self.Bool(n) for n in names.split()]

SMTSolver.Ints = _ints
SMTSolver.Bools = _bools


# --- SMT-LIB2 Parser (subset) ---

def parse_smtlib2(text: str) -> tuple:
    """Parse a subset of SMT-LIB2 format. Returns (solver, assertions)."""
    solver = SMTSolver()
    tokens = _tokenize_smt(text)
    pos = [0]

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def consume():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def expect(t):
        got = consume()
        if got != t:
            raise ValueError(f"Expected {t}, got {got}")

    def parse_expr():
        t = peek()
        if t == '(':
            consume()
            cmd = consume()

            if cmd in ('+', '-', '*'):
                op = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL}[cmd]
                args = []
                while peek() != ')':
                    args.append(parse_expr())
                consume()  # )
                if cmd == '-' and len(args) == 1:
                    return App(Op.NEG, args, INT)
                result = args[0]
                for a in args[1:]:
                    result = App(op, [result, a], INT)
                return result

            if cmd in ('<=', '<', '>=', '>', '='):
                op = {'<=': Op.LE, '<': Op.LT, '>=': Op.GE, '>': Op.GT, '=': Op.EQ}[cmd]
                args = []
                while peek() != ')':
                    args.append(parse_expr())
                consume()  # )
                if len(args) == 2:
                    return App(op, args, BOOL)
                # Chain: (= a b c) means a=b AND b=c
                constraints = []
                for i in range(len(args) - 1):
                    constraints.append(App(op, [args[i], args[i+1]], BOOL))
                return solver.And(*constraints)

            if cmd == 'and':
                args = []
                while peek() != ')':
                    args.append(parse_expr())
                consume()
                return solver.And(*args)

            if cmd == 'or':
                args = []
                while peek() != ')':
                    args.append(parse_expr())
                consume()
                return solver.Or(*args)

            if cmd == 'not':
                arg = parse_expr()
                expect(')')
                return solver.Not(arg)

            if cmd == '=>':
                a = parse_expr()
                b = parse_expr()
                expect(')')
                return solver.Implies(a, b)

            if cmd == 'ite':
                c = parse_expr()
                t = parse_expr()
                e = parse_expr()
                expect(')')
                return solver.If(c, t, e)

            if cmd == 'distinct':
                args = []
                while peek() != ')':
                    args.append(parse_expr())
                consume()
                return solver.Distinct(*args)

            raise ValueError(f"Unknown command: {cmd}")

        else:
            consume()
            # Number?
            try:
                return IntConst(int(t))
            except ValueError:
                pass
            if t == 'true':
                return BoolConst(True)
            if t == 'false':
                return BoolConst(False)
            # Variable reference
            if t in solver._vars:
                return solver._vars[t]
            # Create as Int by default
            return solver.Int(t)

    while pos[0] < len(tokens):
        t = peek()
        if t == '(':
            consume()
            cmd = consume()

            if cmd == 'declare-const':
                name = consume()
                sort_name = consume()
                expect(')')
                if sort_name == 'Int':
                    solver.Int(name)
                elif sort_name == 'Bool':
                    solver.Bool(name)
                elif sort_name == 'Real':
                    solver.Real(name)

            elif cmd == 'declare-fun':
                name = consume()
                expect('(')
                domain = []
                while peek() != ')':
                    s = consume()
                    if s == 'Int':
                        domain.append(INT)
                    elif s == 'Bool':
                        domain.append(BOOL)
                consume()  # )
                range_name = consume()
                expect(')')
                if domain:
                    range_sort = INT if range_name == 'Int' else BOOL
                    solver.Function(name, *domain, range_sort)
                else:
                    if range_name == 'Int':
                        solver.Int(name)
                    elif range_name == 'Bool':
                        solver.Bool(name)

            elif cmd == 'assert':
                expr = parse_expr()
                expect(')')
                solver.add(expr)

            elif cmd == 'check-sat':
                expect(')')

            elif cmd == 'get-model':
                expect(')')

            else:
                # Skip unknown
                depth = 1
                while depth > 0:
                    t = consume()
                    if t == '(':
                        depth += 1
                    elif t == ')':
                        depth -= 1
        else:
            consume()

    return solver


def _tokenize_smt(text: str) -> list:
    """Tokenize SMT-LIB2 text."""
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in ' \t\n\r':
            i += 1
        elif c == ';':
            # Comment
            while i < len(text) and text[i] != '\n':
                i += 1
        elif c == '(':
            tokens.append('(')
            i += 1
        elif c == ')':
            tokens.append(')')
            i += 1
        elif c == '"':
            # String literal
            j = i + 1
            while j < len(text) and text[j] != '"':
                j += 1
            tokens.append(text[i:j+1])
            i = j + 1
        else:
            j = i
            while j < len(text) and text[j] not in ' \t\n\r()':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens
