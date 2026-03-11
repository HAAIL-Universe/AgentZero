"""
C124: Linear Programming -- Simplex Method & LP Solver

A complete linear programming solver featuring:
1. StandardLP -- LP in standard form (minimize c^T x, Ax <= b, x >= 0)
2. SimplexSolver -- revised simplex with Bland's anti-cycling rule
3. TwoPhaseSimplexSolver -- handles infeasible starting points
4. DualSimplexSolver -- dual simplex for post-optimality changes
5. LPBuilder -- user-friendly LP construction with named variables
6. MILPSolver -- branch-and-bound mixed-integer LP solver

Algorithms: simplex tableau, two-phase method, dual simplex, branch & bound
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import math


class LPStatus(Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NOT_SOLVED = "not_solved"


@dataclass
class LPResult:
    status: LPStatus
    objective: Optional[float] = None
    variables: Optional[dict] = None
    dual_values: Optional[list] = None
    iterations: int = 0

    def __repr__(self):
        if self.status == LPStatus.OPTIMAL:
            return f"LPResult(OPTIMAL, obj={self.objective:.6f}, vars={self.variables})"
        return f"LPResult({self.status.value})"


# ---------------------------------------------------------------------------
# StandardLP: represents LP in standard form
# ---------------------------------------------------------------------------

class StandardLP:
    """
    LP in standard form:
      minimize   c^T x
      subject to Ax <= b
                 x >= 0

    c: objective coefficients (length n)
    A: constraint matrix (m x n)
    b: RHS values (length m, must be >= 0 for direct simplex)
    """

    def __init__(self, c, A, b):
        self.c = [float(v) for v in c]
        self.A = [[float(v) for v in row] for row in A]
        self.b = [float(v) for v in b]
        self.n = len(c)  # number of decision variables
        self.m = len(b)  # number of constraints

        if len(A) != self.m:
            raise ValueError(f"A has {len(A)} rows but b has {self.m} entries")
        for i, row in enumerate(A):
            if len(row) != self.n:
                raise ValueError(f"Row {i} has {len(row)} cols but expected {self.n}")


# ---------------------------------------------------------------------------
# SimplexSolver: tableau-based simplex with Bland's rule
# ---------------------------------------------------------------------------

EPS = 1e-9

class SimplexSolver:
    """
    Solves LP in standard form using the simplex method.
    Assumes b >= 0 (use TwoPhaseSimplexSolver if not guaranteed).
    Uses Bland's rule for anti-cycling.
    """

    def solve(self, lp):
        m, n = lp.m, lp.n

        # Check b >= 0
        for i in range(m):
            if lp.b[i] < -EPS:
                return LPResult(status=LPStatus.INFEASIBLE)

        # Build initial tableau with slack variables
        # Columns: x1..xn, s1..sm
        # Rows: constraints + objective
        tableau = []
        for i in range(m):
            row = lp.A[i][:] + [0.0] * m + [lp.b[i]]
            row[n + i] = 1.0
            tableau.append(row)

        # Objective row: c^T for original vars, 0 for slacks, 0 for RHS
        obj_row = lp.c[:] + [0.0] * m + [0.0]
        tableau.append(obj_row)

        # Basis: initially the slack variables
        basis = list(range(n, n + m))
        total_vars = n + m

        iterations = 0
        max_iter = 10000

        while iterations < max_iter:
            iterations += 1

            # Find entering variable (Bland's rule: smallest index with negative reduced cost)
            pivot_col = -1
            for j in range(total_vars):
                if tableau[m][j] < -EPS:
                    pivot_col = j
                    break

            if pivot_col == -1:
                # Optimal
                break

            # Find leaving variable (minimum ratio test, Bland's for ties)
            pivot_row = -1
            min_ratio = float('inf')
            for i in range(m):
                if tableau[i][pivot_col] > EPS:
                    ratio = tableau[i][-1] / tableau[i][pivot_col]
                    if ratio < min_ratio - EPS:
                        min_ratio = ratio
                        pivot_row = i
                    elif abs(ratio - min_ratio) < EPS:
                        # Bland's: pick row with smallest basis index
                        if basis[i] < basis[pivot_row]:
                            pivot_row = i

            if pivot_row == -1:
                return LPResult(status=LPStatus.UNBOUNDED, iterations=iterations)

            # Pivot
            self._pivot(tableau, pivot_row, pivot_col, m)
            basis[pivot_row] = pivot_col

        # Extract solution
        solution = [0.0] * n
        for i in range(m):
            if basis[i] < n:
                solution[basis[i]] = tableau[i][-1]

        # Tableau convention: RHS of obj row stores -z (negated objective)
        obj_val = -tableau[m][-1]

        # Dual values: reduced cost of slack i = -y_i
        dual_values = [-tableau[m][n + i] for i in range(m)]

        variables = {f"x{i}": solution[i] for i in range(n)}

        return LPResult(
            status=LPStatus.OPTIMAL,
            objective=obj_val,
            variables=variables,
            dual_values=dual_values,
            iterations=iterations,
        )

    def _pivot(self, tableau, pivot_row, pivot_col, m):
        pivot_val = tableau[pivot_row][pivot_col]
        cols = len(tableau[0])

        # Scale pivot row
        for j in range(cols):
            tableau[pivot_row][j] /= pivot_val

        # Eliminate column in all other rows (including objective)
        for i in range(m + 1):
            if i == pivot_row:
                continue
            factor = tableau[i][pivot_col]
            if abs(factor) < EPS:
                continue
            for j in range(cols):
                tableau[i][j] -= factor * tableau[pivot_row][j]


# ---------------------------------------------------------------------------
# TwoPhaseSimplexSolver: handles arbitrary RHS signs
# ---------------------------------------------------------------------------

class TwoPhaseSimplexSolver:
    """
    Two-phase simplex method.
    Phase 1: find a BFS by minimizing sum of artificial variables.
    Phase 2: optimize the original objective from the BFS.
    """

    def solve(self, lp):
        m, n = lp.m, lp.n

        # Make b >= 0 by flipping rows with negative b
        A = [row[:] for row in lp.A]
        b = lp.b[:]
        flipped = [False] * m
        for i in range(m):
            if b[i] < -EPS:
                A[i] = [-v for v in A[i]]
                b[i] = -b[i]
                flipped[i] = True

        # Check if we need phase 1 (only if any row was flipped)
        need_phase1 = any(flipped)

        if not need_phase1:
            # Direct simplex
            direct_lp = StandardLP(lp.c, A, b)
            return SimplexSolver().solve(direct_lp)

        # Phase 1: add artificial variables for flipped rows
        # Columns: x1..xn, s1..sm, a1..ak (artificial for flipped rows)
        art_indices = [i for i in range(m) if flipped[i]]
        k = len(art_indices)

        total_phase1 = n + m + k

        tableau = []
        art_col_map = {}  # row index -> column index of its artificial var
        art_idx = 0
        for i in range(m):
            row = A[i][:] + [0.0] * m + [0.0] * k + [b[i]]
            row[n + i] = 1.0  # slack
            if flipped[i]:
                # This row was flipped -- slack goes wrong way, need artificial
                row[n + i] = -1.0  # slack is subtracted (flipped constraint)
                col = n + m + art_idx
                row[col] = 1.0
                art_col_map[i] = col
                art_idx += 1
            tableau.append(row)

        # Phase 1 objective: minimize sum of artificial variables
        obj_row = [0.0] * (total_phase1 + 1)
        for i in art_indices:
            col = art_col_map[i]
            obj_row[col] = 1.0
        tableau.append(obj_row)

        # Basis: slacks for non-flipped, artificials for flipped
        basis = []
        for i in range(m):
            if flipped[i]:
                basis.append(art_col_map[i])
            else:
                basis.append(n + i)

        # Make objective row canonical (subtract artificial basis rows)
        for i in art_indices:
            col = art_col_map[i]
            factor = tableau[m][col]  # should be 1.0 initially, but after subtraction...
            # Actually, we need to subtract each artificial-basis row from obj
            for j in range(total_phase1 + 1):
                tableau[m][j] -= tableau[i][j]

        # Solve phase 1
        iterations = 0
        max_iter = 10000

        while iterations < max_iter:
            iterations += 1

            pivot_col = -1
            for j in range(total_phase1):
                if tableau[m][j] < -EPS:
                    pivot_col = j
                    break
            if pivot_col == -1:
                break

            pivot_row = -1
            min_ratio = float('inf')
            for i in range(m):
                if tableau[i][pivot_col] > EPS:
                    ratio = tableau[i][-1] / tableau[i][pivot_col]
                    if ratio < min_ratio - EPS:
                        min_ratio = ratio
                        pivot_row = i
                    elif abs(ratio - min_ratio) < EPS:
                        if basis[i] < basis[pivot_row]:
                            pivot_row = i
            if pivot_row == -1:
                break  # Unbounded in phase 1 shouldn't happen

            self._pivot(tableau, pivot_row, pivot_col, m)
            basis[pivot_row] = pivot_col

        # Check phase 1 objective (RHS stores -z, so negate)
        phase1_obj = -tableau[m][-1]
        if phase1_obj > EPS:
            return LPResult(status=LPStatus.INFEASIBLE, iterations=iterations)

        # Pivot out any degenerate artificial variables still in basis
        art_col_set = set(art_col_map.values())
        for i in range(m):
            if basis[i] in art_col_set:
                # Find a non-artificial column with non-zero coefficient to pivot on
                pivot_col = -1
                for j in range(n + m):
                    if abs(tableau[i][j]) > EPS:
                        pivot_col = j
                        break
                if pivot_col != -1:
                    self._pivot(tableau, i, pivot_col, m)
                    basis[i] = pivot_col

        # Phase 2: remove artificial columns, replace objective
        # Build phase 2 tableau (drop artificial columns)
        phase2_cols = n + m  # original + slack
        tableau2 = []
        for i in range(m):
            row2 = tableau[i][:phase2_cols] + [tableau[i][-1]]
            tableau2.append(row2)

        # Phase 2 objective
        obj2 = lp.c[:] + [0.0] * m + [0.0]
        tableau2.append(obj2)

        # Make objective row canonical w.r.t. current basis
        for i in range(m):
            bv = basis[i]
            if bv < phase2_cols:
                factor = tableau2[m][bv]
                if abs(factor) > EPS:
                    for j in range(phase2_cols + 1):
                        tableau2[m][j] -= factor * tableau2[i][j]

        # Solve phase 2
        while iterations < max_iter:
            iterations += 1

            pivot_col = -1
            for j in range(phase2_cols):
                if tableau2[m][j] < -EPS:
                    pivot_col = j
                    break
            if pivot_col == -1:
                break

            pivot_row = -1
            min_ratio = float('inf')
            for i in range(m):
                if tableau2[i][pivot_col] > EPS:
                    ratio = tableau2[i][-1] / tableau2[i][pivot_col]
                    if ratio < min_ratio - EPS:
                        min_ratio = ratio
                        pivot_row = i
                    elif abs(ratio - min_ratio) < EPS:
                        if basis[i] < basis[pivot_row]:
                            pivot_row = i
            if pivot_row == -1:
                return LPResult(status=LPStatus.UNBOUNDED, iterations=iterations)

            self._pivot(tableau2, pivot_row, pivot_col, m)
            basis[pivot_row] = pivot_col

        # Extract solution
        solution = [0.0] * n
        for i in range(m):
            if basis[i] < n:
                solution[basis[i]] = tableau2[i][-1]

        obj_val = -tableau2[m][-1]
        dual_values = [-tableau2[m][n + i] for i in range(m)]
        variables = {f"x{i}": solution[i] for i in range(n)}

        return LPResult(
            status=LPStatus.OPTIMAL,
            objective=obj_val,
            variables=variables,
            dual_values=dual_values,
            iterations=iterations,
        )

    def _pivot(self, tableau, pivot_row, pivot_col, m):
        pivot_val = tableau[pivot_row][pivot_col]
        cols = len(tableau[0])
        for j in range(cols):
            tableau[pivot_row][j] /= pivot_val
        for i in range(m + 1):
            if i == pivot_row:
                continue
            factor = tableau[i][pivot_col]
            if abs(factor) < EPS:
                continue
            for j in range(cols):
                tableau[i][j] -= factor * tableau[pivot_row][j]


# ---------------------------------------------------------------------------
# DualSimplexSolver: dual simplex for post-optimality analysis
# ---------------------------------------------------------------------------

class DualSimplexSolver:
    """
    Dual simplex method. Starts with a dual-feasible tableau
    (all reduced costs >= 0) and iterates to primal feasibility.
    Useful for adding constraints to an optimal solution.
    """

    def solve(self, lp):
        m, n = lp.m, lp.n

        # Build tableau
        tableau = []
        for i in range(m):
            row = lp.A[i][:] + [0.0] * m + [lp.b[i]]
            row[n + i] = 1.0
            tableau.append(row)

        obj_row = lp.c[:] + [0.0] * m + [0.0]
        tableau.append(obj_row)

        basis = list(range(n, n + m))
        total_vars = n + m
        iterations = 0
        max_iter = 10000

        # Check dual feasibility (all reduced costs >= 0 for non-basic)
        dual_feasible = all(tableau[m][j] >= -EPS for j in range(total_vars))

        if not dual_feasible:
            # Fall back to two-phase
            return TwoPhaseSimplexSolver().solve(lp)

        while iterations < max_iter:
            iterations += 1

            # Find leaving variable: row with most negative RHS (Bland's: smallest index)
            pivot_row = -1
            min_rhs = -EPS
            for i in range(m):
                if tableau[i][-1] < min_rhs:
                    min_rhs = tableau[i][-1]
                    pivot_row = i

            if pivot_row == -1:
                # Primal feasible -- optimal
                break

            # Find entering variable: min ratio of reduced cost / |tableau[pivot_row][j]|
            # for negative entries in pivot row
            pivot_col = -1
            min_ratio = float('inf')
            for j in range(total_vars):
                if tableau[pivot_row][j] < -EPS:
                    ratio = abs(tableau[m][j] / tableau[pivot_row][j])
                    if ratio < min_ratio - EPS:
                        min_ratio = ratio
                        pivot_col = j
                    elif abs(ratio - min_ratio) < EPS:
                        if j < (pivot_col if pivot_col >= 0 else float('inf')):
                            pivot_col = j

            if pivot_col == -1:
                return LPResult(status=LPStatus.INFEASIBLE, iterations=iterations)

            # Pivot
            self._pivot(tableau, pivot_row, pivot_col, m)
            basis[pivot_row] = pivot_col

        # Extract solution
        solution = [0.0] * n
        for i in range(m):
            if basis[i] < n:
                solution[basis[i]] = tableau[i][-1]

        obj_val = -tableau[m][-1]
        dual_values = [-tableau[m][n + i] for i in range(m)]
        variables = {f"x{i}": solution[i] for i in range(n)}

        return LPResult(
            status=LPStatus.OPTIMAL,
            objective=obj_val,
            variables=variables,
            dual_values=dual_values,
            iterations=iterations,
        )

    def _pivot(self, tableau, pivot_row, pivot_col, m):
        pivot_val = tableau[pivot_row][pivot_col]
        cols = len(tableau[0])
        for j in range(cols):
            tableau[pivot_row][j] /= pivot_val
        for i in range(m + 1):
            if i == pivot_row:
                continue
            factor = tableau[i][pivot_col]
            if abs(factor) < EPS:
                continue
            for j in range(cols):
                tableau[i][j] -= factor * tableau[pivot_row][j]


# ---------------------------------------------------------------------------
# LPBuilder: user-friendly LP construction
# ---------------------------------------------------------------------------

class Sense(Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"


class ConstraintOp(Enum):
    LEQ = "<="
    GEQ = ">="
    EQ = "=="


@dataclass
class Variable:
    name: str
    index: int
    lower: float = 0.0
    upper: float = float('inf')
    is_integer: bool = False


@dataclass
class Constraint:
    coeffs: dict  # var_name -> coefficient
    op: ConstraintOp
    rhs: float
    name: str = ""


class LPBuilder:
    """
    User-friendly LP construction with named variables.

    Usage:
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 5, y: 4})
        lp.add_constraint({x: 6, y: 4}, ConstraintOp.LEQ, 24)
        lp.add_constraint({x: 1, y: 2}, ConstraintOp.LEQ, 6)
        result = lp.solve()
    """

    def __init__(self, sense=Sense.MINIMIZE):
        self.sense = sense
        self.variables = {}  # name -> Variable
        self.var_order = []  # ordered list of names
        self.constraints = []
        self.objective = {}  # var_name -> coefficient
        self._next_idx = 0

    def add_var(self, name, lower=0.0, upper=float('inf'), is_integer=False):
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists")
        var = Variable(name=name, index=self._next_idx, lower=lower,
                       upper=upper, is_integer=is_integer)
        self.variables[name] = var
        self.var_order.append(name)
        self._next_idx += 1
        return name

    def set_objective(self, coeffs):
        self.objective = {}
        for var_name, coeff in coeffs.items():
            if var_name not in self.variables:
                raise ValueError(f"Unknown variable '{var_name}'")
            self.objective[var_name] = float(coeff)

    def add_constraint(self, coeffs, op, rhs, name=""):
        for var_name in coeffs:
            if var_name not in self.variables:
                raise ValueError(f"Unknown variable '{var_name}'")
        self.constraints.append(Constraint(
            coeffs={k: float(v) for k, v in coeffs.items()},
            op=op,
            rhs=float(rhs),
            name=name,
        ))

    def _to_standard_form(self):
        """Convert to StandardLP (minimize, Ax <= b, x >= 0)."""
        n = len(self.var_order)

        # Objective coefficients
        c = [0.0] * n
        for var_name, coeff in self.objective.items():
            idx = self.variables[var_name].index
            if self.sense == Sense.MAXIMIZE:
                c[idx] = -coeff  # negate for maximization
            else:
                c[idx] = coeff

        # Convert constraints
        A_rows = []
        b_vals = []

        for con in self.constraints:
            row = [0.0] * n
            for var_name, coeff in con.coeffs.items():
                row[self.variables[var_name].index] = coeff

            if con.op == ConstraintOp.LEQ:
                A_rows.append(row)
                b_vals.append(con.rhs)
            elif con.op == ConstraintOp.GEQ:
                # Negate: -a^T x <= -b
                A_rows.append([-v for v in row])
                b_vals.append(-con.rhs)
            elif con.op == ConstraintOp.EQ:
                # Split into <= and >=
                A_rows.append(row[:])
                b_vals.append(con.rhs)
                A_rows.append([-v for v in row])
                b_vals.append(-con.rhs)

        # Add upper bound constraints
        for var_name in self.var_order:
            var = self.variables[var_name]
            if var.upper < float('inf'):
                row = [0.0] * n
                row[var.index] = 1.0
                A_rows.append(row)
                b_vals.append(var.upper)

        return StandardLP(c, A_rows, b_vals)

    def solve(self):
        std_lp = self._to_standard_form()

        # Check if any integer variables
        has_integer = any(v.is_integer for v in self.variables.values())
        if has_integer:
            result = MILPSolver().solve(self)
            return result

        result = TwoPhaseSimplexSolver().solve(std_lp)

        # Post-process: map back to named variables and fix maximization
        if result.status == LPStatus.OPTIMAL:
            named_vars = {}
            for i, var_name in enumerate(self.var_order):
                named_vars[var_name] = result.variables.get(f"x{i}", 0.0)
            result.variables = named_vars

            if self.sense == Sense.MAXIMIZE:
                result.objective = -result.objective

        return result


# ---------------------------------------------------------------------------
# MILPSolver: branch-and-bound for mixed-integer LP
# ---------------------------------------------------------------------------

class MILPSolver:
    """
    Branch-and-bound solver for mixed-integer linear programming.
    Solves LP relaxation at each node, branches on fractional integer variables.
    """

    def solve(self, builder):
        self.builder = builder
        self.integer_vars = {name for name, var in builder.variables.items()
                             if var.is_integer}
        self.best_obj = float('inf')
        self.best_vars = None
        self.iterations = 0
        self.max_iter = 50000

        # Solve root relaxation
        result = self._solve_relaxation(builder)
        if result.status != LPStatus.OPTIMAL:
            return LPResult(status=result.status, iterations=self.iterations)

        # Branch and bound
        self._branch_and_bound(builder, result)

        if self.best_vars is None:
            return LPResult(status=LPStatus.INFEASIBLE, iterations=self.iterations)

        obj = self.best_obj
        if builder.sense == Sense.MAXIMIZE:
            obj = -obj

        return LPResult(
            status=LPStatus.OPTIMAL,
            objective=obj,
            variables=dict(self.best_vars),
            iterations=self.iterations,
        )

    def _solve_relaxation(self, builder):
        """Solve LP relaxation (ignore integrality)."""
        # Create a copy without integer constraints
        relaxed = LPBuilder(sense=Sense.MINIMIZE)  # Always minimize internally
        for name in builder.var_order:
            var = builder.variables[name]
            relaxed.add_var(name, lower=var.lower, upper=var.upper, is_integer=False)

        # Set objective (already handle sense conversion)
        obj = {}
        for var_name, coeff in builder.objective.items():
            if builder.sense == Sense.MAXIMIZE:
                obj[var_name] = -coeff
            else:
                obj[var_name] = coeff
        relaxed.set_objective(obj)

        for con in builder.constraints:
            relaxed.add_constraint(con.coeffs, con.op, con.rhs, con.name)

        return relaxed.solve()

    def _is_integer_feasible(self, variables):
        for name in self.integer_vars:
            val = variables.get(name, 0.0)
            if abs(val - round(val)) > EPS:
                return False
        return True

    def _find_branching_var(self, variables):
        """Find most fractional integer variable."""
        best_var = None
        best_frac = 0.0
        for name in self.integer_vars:
            val = variables.get(name, 0.0)
            frac = abs(val - round(val))
            if frac > EPS and frac > best_frac:
                best_frac = frac
                best_var = name
        return best_var

    def _branch_and_bound(self, builder, relaxation_result):
        """Iterative branch-and-bound using a stack."""
        # Each node is a list of extra constraints (var_name, op, rhs)
        stack = [([], relaxation_result)]

        while stack and self.iterations < self.max_iter:
            extra_constraints, node_result = stack.pop()
            self.iterations += 1

            if node_result.status != LPStatus.OPTIMAL:
                continue

            # Prune: relaxation worse than best known
            obj = node_result.objective
            if obj >= self.best_obj - EPS:
                continue

            if self._is_integer_feasible(node_result.variables):
                if obj < self.best_obj - EPS:
                    self.best_obj = obj
                    self.best_vars = node_result.variables
                continue

            # Branch
            branch_var = self._find_branching_var(node_result.variables)
            if branch_var is None:
                continue

            val = node_result.variables[branch_var]
            floor_val = math.floor(val)
            ceil_val = math.ceil(val)

            # Create two child nodes
            for bound_val, op in [(floor_val, ConstraintOp.LEQ),
                                  (ceil_val, ConstraintOp.GEQ)]:
                child_constraints = extra_constraints + [(branch_var, op, bound_val)]
                child_result = self._solve_node(builder, child_constraints)
                if child_result.status == LPStatus.OPTIMAL:
                    if child_result.objective < self.best_obj - EPS:
                        stack.append((child_constraints, child_result))

    def _solve_node(self, builder, extra_constraints):
        """Solve LP relaxation with extra branching constraints."""
        relaxed = LPBuilder(sense=Sense.MINIMIZE)
        for name in builder.var_order:
            var = builder.variables[name]
            relaxed.add_var(name, lower=var.lower, upper=var.upper, is_integer=False)

        obj = {}
        for var_name, coeff in builder.objective.items():
            if builder.sense == Sense.MAXIMIZE:
                obj[var_name] = -coeff
            else:
                obj[var_name] = coeff
        relaxed.set_objective(obj)

        for con in builder.constraints:
            relaxed.add_constraint(con.coeffs, con.op, con.rhs, con.name)

        # Add branching constraints
        for var_name, op, rhs in extra_constraints:
            relaxed.add_constraint({var_name: 1.0}, op, rhs)

        result = relaxed.solve()
        return result


# ---------------------------------------------------------------------------
# Sensitivity analysis utilities
# ---------------------------------------------------------------------------

def sensitivity_rhs(lp, constraint_index, delta):
    """
    Analyze how changing RHS of a constraint affects the optimal solution.
    Returns result with new RHS = b[i] + delta.
    """
    new_b = lp.b[:]
    new_b[constraint_index] += delta
    new_lp = StandardLP(lp.c, lp.A, new_b)
    return TwoPhaseSimplexSolver().solve(new_lp)


def sensitivity_obj(lp, var_index, delta):
    """
    Analyze how changing an objective coefficient affects the optimal solution.
    Returns result with new c[j] = c[j] + delta.
    """
    new_c = lp.c[:]
    new_c[var_index] += delta
    new_lp = StandardLP(new_c, lp.A, lp.b)
    return TwoPhaseSimplexSolver().solve(new_lp)


# ---------------------------------------------------------------------------
# Classic LP problems as convenience constructors
# ---------------------------------------------------------------------------

def transportation_problem(supply, demand, costs):
    """
    Solve a transportation problem.
    supply: list of supply amounts (m sources)
    demand: list of demand amounts (n destinations)
    costs: m x n cost matrix

    Returns LPResult with flow variables named "x_i_j".
    """
    m_src = len(supply)
    n_dst = len(demand)

    lp = LPBuilder(sense=Sense.MINIMIZE)

    # Create variables
    var_names = {}
    for i in range(m_src):
        for j in range(n_dst):
            name = f"x_{i}_{j}"
            lp.add_var(name)
            var_names[(i, j)] = name

    # Objective: minimize total cost
    obj = {}
    for i in range(m_src):
        for j in range(n_dst):
            obj[var_names[(i, j)]] = costs[i][j]
    lp.set_objective(obj)

    # Supply constraints: sum_j x_ij <= supply_i
    for i in range(m_src):
        coeffs = {var_names[(i, j)]: 1.0 for j in range(n_dst)}
        lp.add_constraint(coeffs, ConstraintOp.LEQ, supply[i])

    # Demand constraints: sum_i x_ij >= demand_j
    for j in range(n_dst):
        coeffs = {var_names[(i, j)]: 1.0 for i in range(m_src)}
        lp.add_constraint(coeffs, ConstraintOp.GEQ, demand[j])

    return lp.solve()


def diet_problem(nutrients_min, food_nutrients, food_costs):
    """
    Solve a diet problem.
    nutrients_min: list of minimum nutrient requirements
    food_nutrients: matrix [food][nutrient] of nutrient content per unit of food
    food_costs: cost per unit of each food

    Returns LPResult minimizing cost while meeting nutrient requirements.
    """
    n_foods = len(food_costs)
    n_nutrients = len(nutrients_min)

    lp = LPBuilder(sense=Sense.MINIMIZE)

    food_vars = [lp.add_var(f"food_{i}") for i in range(n_foods)]

    # Objective: minimize cost
    obj = {food_vars[i]: food_costs[i] for i in range(n_foods)}
    lp.set_objective(obj)

    # Nutrient constraints: sum_i food_nutrients[i][j] * x_i >= min_j
    for j in range(n_nutrients):
        coeffs = {food_vars[i]: food_nutrients[i][j] for i in range(n_foods)}
        lp.add_constraint(coeffs, ConstraintOp.GEQ, nutrients_min[j])

    return lp.solve()
