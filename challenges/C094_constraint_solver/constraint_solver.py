"""
C094: Constraint Solver -- CSP solver composing C035 (SAT) + C037 (SMT).

Finite-domain constraint satisfaction with:
- Variables with finite integer domains
- Constraint types: equality, inequality, comparison, alldiff, table, arithmetic, callback
- Arc consistency (AC-3) for domain reduction
- Backtracking search with MRV + LCV heuristics
- Forward checking and MAC (Maintaining Arc Consistency)
- SAT encoding for Boolean CSPs
- SMT integration for arithmetic reasoning
- Modeling helpers: Sudoku, N-Queens, graph coloring, scheduling, magic squares
"""

import sys
import os
from enum import Enum
from collections import deque
from itertools import product as cartesian_product

# Compose C035 and C037
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C035_sat_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C037_smt_solver'))

from sat_solver import Solver as SATSolver, SolverResult as SATResult
from smt_solver import SMTSolver, SMTResult


# --- Result ---

class CSPResult(Enum):
    SOLVED = "solved"
    UNSATISFIABLE = "unsatisfiable"
    UNKNOWN = "unknown"


# --- Variable ---

class Variable:
    """Finite-domain variable."""

    def __init__(self, name, domain):
        self.name = name
        self.domain = set(domain)
        if not self.domain:
            raise ValueError(f"Variable '{name}' has empty domain")

    def __repr__(self):
        return f"Variable({self.name}, {sorted(self.domain)})"


# --- Constraints ---

class Constraint:
    """Base constraint class."""

    def __init__(self, variables):
        self.variables = list(variables)

    def satisfied(self, assignment):
        """Check if constraint is satisfied given (partial) assignment."""
        raise NotImplementedError

    def propagate(self, var, val, domains):
        """Propagate assignment var=val. Returns set of (var, removed_val) or None if wipeout."""
        return set()

    def get_arcs(self):
        """Return list of (xi, xj) arcs for AC-3."""
        arcs = []
        for i, vi in enumerate(self.variables):
            for j, vj in enumerate(self.variables):
                if i != j:
                    arcs.append((vi, vj, self))
        return arcs

    def revise(self, xi, xj, domains):
        """Remove values from domains[xi] that have no support in domains[xj].
        Returns set of removed values."""
        removed = set()
        for vi in list(domains[xi]):
            has_support = False
            for vj in domains[xj]:
                test = {xi: vi, xj: vj}
                if self._check_partial(test):
                    has_support = True
                    break
            if not has_support:
                domains[xi].discard(vi)
                removed.add(vi)
        return removed

    def _check_partial(self, assignment):
        """Check if partial assignment is consistent with constraint."""
        # Only check if all variables are assigned
        if all(v in assignment for v in self.variables):
            return self.satisfied(assignment)
        return True


class EqualityConstraint(Constraint):
    """x == y"""
    def __init__(self, x, y):
        super().__init__([x, y])

    def satisfied(self, assignment):
        x, y = self.variables
        if x in assignment and y in assignment:
            return assignment[x] == assignment[y]
        return True

    def propagate(self, var, val, domains):
        removed = set()
        x, y = self.variables
        other = y if var == x else x if var == y else None
        if other is None:
            return removed
        for v in list(domains[other]):
            if v != val:
                domains[other].discard(v)
                removed.add((other, v))
        if not domains[other]:
            return None
        return removed


class InequalityConstraint(Constraint):
    """x != y"""
    def __init__(self, x, y):
        super().__init__([x, y])

    def satisfied(self, assignment):
        x, y = self.variables
        if x in assignment and y in assignment:
            return assignment[x] != assignment[y]
        return True

    def propagate(self, var, val, domains):
        removed = set()
        x, y = self.variables
        other = y if var == x else x if var == y else None
        if other is None:
            return removed
        if val in domains[other]:
            if len(domains[other]) == 1:
                # Would wipe out
                return None
            domains[other].discard(val)
            removed.add((other, val))
        return removed


class ComparisonConstraint(Constraint):
    """x op y where op is <, <=, >, >="""
    def __init__(self, x, y, op):
        super().__init__([x, y])
        if op not in ('<', '<=', '>', '>='):
            raise ValueError(f"Invalid comparison operator: {op}")
        self.op = op

    def satisfied(self, assignment):
        x, y = self.variables
        if x in assignment and y in assignment:
            a, b = assignment[x], assignment[y]
            if self.op == '<': return a < b
            if self.op == '<=': return a <= b
            if self.op == '>': return a > b
            if self.op == '>=': return a >= b
        return True

    def propagate(self, var, val, domains):
        removed = set()
        x, y = self.variables
        if var == x:
            # x = val, constrain y
            for v in list(domains[y]):
                if self.op == '<' and not (val < v): domains[y].discard(v); removed.add((y, v))
                elif self.op == '<=' and not (val <= v): domains[y].discard(v); removed.add((y, v))
                elif self.op == '>' and not (val > v): domains[y].discard(v); removed.add((y, v))
                elif self.op == '>=' and not (val >= v): domains[y].discard(v); removed.add((y, v))
            if not domains[y]:
                return None
        elif var == y:
            # y = val, constrain x
            for v in list(domains[x]):
                if self.op == '<' and not (v < val): domains[x].discard(v); removed.add((x, v))
                elif self.op == '<=' and not (v <= val): domains[x].discard(v); removed.add((x, v))
                elif self.op == '>' and not (v > val): domains[x].discard(v); removed.add((x, v))
                elif self.op == '>=' and not (v >= val): domains[x].discard(v); removed.add((x, v))
            if not domains[x]:
                return None
        return removed


class AllDifferentConstraint(Constraint):
    """All variables must take different values."""
    def __init__(self, variables):
        super().__init__(variables)

    def satisfied(self, assignment):
        vals = [assignment[v] for v in self.variables if v in assignment]
        return len(vals) == len(set(vals))

    def propagate(self, var, val, domains):
        removed = set()
        for other in self.variables:
            if other == var:
                continue
            if val in domains[other]:
                if len(domains[other]) == 1:
                    return None
                domains[other].discard(val)
                removed.add((other, val))
        return removed


class TableConstraint(Constraint):
    """Extensional constraint: list of allowed tuples."""
    def __init__(self, variables, allowed_tuples):
        super().__init__(variables)
        self.allowed = set(tuple(t) for t in allowed_tuples)

    def satisfied(self, assignment):
        if all(v in assignment for v in self.variables):
            vals = tuple(assignment[v] for v in self.variables)
            return vals in self.allowed
        return True

    def _check_partial(self, assignment):
        if all(v in assignment for v in self.variables):
            vals = tuple(assignment[v] for v in self.variables)
            return vals in self.allowed
        # For partial: check if any allowed tuple is consistent
        assigned_indices = [(i, assignment[v]) for i, v in enumerate(self.variables) if v in assignment]
        if not assigned_indices:
            return True
        for t in self.allowed:
            if all(t[i] == val for i, val in assigned_indices):
                return True
        return False

    def propagate(self, var, val, domains):
        removed = set()
        var_idx = self.variables.index(var)
        # Filter allowed tuples by var=val
        consistent = [t for t in self.allowed if t[var_idx] == val]
        for i, other in enumerate(self.variables):
            if other == var:
                continue
            allowed_vals = {t[i] for t in consistent}
            for v in list(domains[other]):
                if v not in allowed_vals:
                    domains[other].discard(v)
                    removed.add((other, v))
            if not domains[other]:
                return None
        return removed


class ArithmeticConstraint(Constraint):
    """Linear arithmetic constraint: sum(coeffs[i] * vars[i]) op rhs.
    coeffs: dict mapping var_name -> coefficient
    op: '==', '!=', '<', '<=', '>', '>='
    rhs: integer constant
    """
    def __init__(self, coeffs, op, rhs):
        self.coeffs = dict(coeffs)
        self.op = op
        self.rhs = rhs
        super().__init__(list(self.coeffs.keys()))

    def satisfied(self, assignment):
        if not all(v in assignment for v in self.variables):
            return True
        total = sum(self.coeffs[v] * assignment[v] for v in self.variables)
        if self.op == '==': return total == self.rhs
        if self.op == '!=': return total != self.rhs
        if self.op == '<': return total < self.rhs
        if self.op == '<=': return total <= self.rhs
        if self.op == '>': return total > self.rhs
        if self.op == '>=': return total >= self.rhs
        return False

    def propagate(self, var, val, domains):
        """Bounds propagation for linear constraints."""
        removed = set()
        unassigned = [v for v in self.variables if v != var and len(domains[v]) > 1]
        if not unassigned:
            return removed

        # For each unassigned variable, compute bounds
        for target in unassigned:
            # Fixed contribution from assigned vars (including var=val)
            fixed = self.coeffs[var] * val
            for v in self.variables:
                if v == var or v == target:
                    continue
                if len(domains[v]) == 1:
                    fixed += self.coeffs[v] * next(iter(domains[v]))
                else:
                    # Can't propagate precisely with multiple unassigned
                    continue

            # Only propagate if target is the only remaining free variable
            other_free = [v for v in self.variables if v != var and v != target and len(domains[v]) > 1]
            if other_free:
                continue

            c = self.coeffs[target]
            # fixed + c * target op rhs  =>  c * target op (rhs - fixed)
            remaining = self.rhs - fixed
            for v in list(domains[target]):
                total = c * v
                ok = True
                if self.op == '==' and total != remaining: ok = False
                elif self.op == '!=' and total == remaining: ok = False
                elif self.op == '<' and not (total + fixed + c * v - c * v < self.rhs):
                    # Re-check: fixed + c*v op rhs
                    ok = (fixed + c * v) < self.rhs
                elif self.op == '<=' and (fixed + c * v) > self.rhs: ok = False
                elif self.op == '>' and (fixed + c * v) <= self.rhs: ok = False
                elif self.op == '>=' and (fixed + c * v) < self.rhs: ok = False
                elif self.op == '==' and (fixed + c * v) != self.rhs: ok = False
                elif self.op == '!=' and (fixed + c * v) == self.rhs: ok = False

                if not ok:
                    domains[target].discard(v)
                    removed.add((target, v))
            if not domains[target]:
                return None
        return removed


class CallbackConstraint(Constraint):
    """User-defined constraint with a callback function."""
    def __init__(self, variables, callback):
        super().__init__(variables)
        self.callback = callback

    def satisfied(self, assignment):
        if all(v in assignment for v in self.variables):
            vals = {v: assignment[v] for v in self.variables}
            return self.callback(vals)
        return True


class SumConstraint(Constraint):
    """sum(variables) == target"""
    def __init__(self, variables, target):
        super().__init__(variables)
        self.target = target

    def satisfied(self, assignment):
        if all(v in assignment for v in self.variables):
            return sum(assignment[v] for v in self.variables) == self.target
        return True

    def propagate(self, var, val, domains):
        removed = set()
        assigned_sum = 0
        unassigned = []
        for v in self.variables:
            if v == var:
                assigned_sum += val
            elif len(domains[v]) == 1:
                assigned_sum += next(iter(domains[v]))
            else:
                unassigned.append(v)

        if len(unassigned) == 0:
            return removed

        if len(unassigned) == 1:
            target_val = self.target - assigned_sum
            other = unassigned[0]
            for v in list(domains[other]):
                if v != target_val:
                    domains[other].discard(v)
                    removed.add((other, v))
            if not domains[other]:
                return None
        else:
            # Bounds check for remaining
            remaining_target = self.target - assigned_sum
            for target_var in unassigned:
                others = [v for v in unassigned if v != target_var]
                min_others = sum(min(domains[v]) for v in others)
                max_others = sum(max(domains[v]) for v in others)
                # target_var + sum(others) = remaining_target
                # target_var = remaining_target - sum(others)
                min_val = remaining_target - max_others
                max_val = remaining_target - min_others
                for v in list(domains[target_var]):
                    if v < min_val or v > max_val:
                        domains[target_var].discard(v)
                        removed.add((target_var, v))
                if not domains[target_var]:
                    return None
        return removed


# --- Search Strategy ---

class SearchStrategy(Enum):
    BACKTRACKING = "backtracking"
    FORWARD_CHECKING = "forward_checking"
    MAC = "mac"  # Maintaining Arc Consistency


# --- CSP Solver ---

class CSPSolver:
    """Finite-domain constraint satisfaction problem solver."""

    def __init__(self):
        self.variables = {}  # name -> Variable
        self.constraints = []
        self.var_constraints = {}  # var_name -> [constraint indices]
        self.strategy = SearchStrategy.MAC
        self.stats = {
            'backtracks': 0,
            'nodes': 0,
            'propagations': 0,
            'solutions_found': 0,
        }

    def add_variable(self, name, domain):
        """Add a variable with finite domain."""
        var = Variable(name, domain)
        self.variables[name] = var
        self.var_constraints[name] = []
        return name

    def add_constraint(self, constraint):
        """Add a constraint."""
        idx = len(self.constraints)
        self.constraints.append(constraint)
        for v in constraint.variables:
            if v in self.var_constraints:
                self.var_constraints[v].append(idx)

    def add_equality(self, x, y):
        self.add_constraint(EqualityConstraint(x, y))

    def add_inequality(self, x, y):
        self.add_constraint(InequalityConstraint(x, y))

    def add_comparison(self, x, y, op):
        self.add_constraint(ComparisonConstraint(x, y, op))

    def add_alldiff(self, variables):
        self.add_constraint(AllDifferentConstraint(variables))

    def add_table(self, variables, allowed):
        self.add_constraint(TableConstraint(variables, allowed))

    def add_arithmetic(self, coeffs, op, rhs):
        self.add_constraint(ArithmeticConstraint(coeffs, op, rhs))

    def add_sum(self, variables, target):
        self.add_constraint(SumConstraint(variables, target))

    def add_callback(self, variables, fn):
        self.add_constraint(CallbackConstraint(variables, fn))

    def solve(self, strategy=None):
        """Solve the CSP. Returns (CSPResult, assignment or None)."""
        if strategy:
            self.strategy = strategy
        self.stats = {'backtracks': 0, 'nodes': 0, 'propagations': 0, 'solutions_found': 0}

        # Initial domains
        domains = {name: set(var.domain) for name, var in self.variables.items()}

        # Initial arc consistency
        if self.strategy == SearchStrategy.MAC:
            if not self._ac3(domains):
                return CSPResult.UNSATISFIABLE, None

        # Check for already-solved (all singletons)
        if all(len(d) == 1 for d in domains.values()):
            assignment = {name: next(iter(d)) for name, d in domains.items()}
            if self._is_consistent(assignment):
                self.stats['solutions_found'] = 1
                return CSPResult.SOLVED, assignment

        result = self._backtrack({}, domains)
        if result is not None:
            self.stats['solutions_found'] = 1
            return CSPResult.SOLVED, result
        return CSPResult.UNSATISFIABLE, None

    def solve_all(self, max_solutions=None):
        """Find all solutions (or up to max_solutions)."""
        self.stats = {'backtracks': 0, 'nodes': 0, 'propagations': 0, 'solutions_found': 0}
        domains = {name: set(var.domain) for name, var in self.variables.items()}

        if not self._ac3(domains):
            return []

        solutions = []
        self._backtrack_all({}, domains, solutions, max_solutions)
        self.stats['solutions_found'] = len(solutions)
        return solutions

    def _backtrack(self, assignment, domains):
        """Backtracking search with constraint propagation."""
        if len(assignment) == len(self.variables):
            if self._is_consistent(assignment):
                return dict(assignment)
            return None

        self.stats['nodes'] += 1
        var = self._select_variable(assignment, domains)
        if var is None:
            return None

        for val in self._order_values(var, domains):
            if self._is_value_consistent(var, val, assignment):
                assignment[var] = val
                saved_domains = {k: set(v) for k, v in domains.items()}

                domains[var] = {val}
                ok = True

                if self.strategy == SearchStrategy.FORWARD_CHECKING:
                    ok = self._forward_check(var, val, domains)
                elif self.strategy == SearchStrategy.MAC:
                    ok = self._propagate_and_ac3(var, val, domains)
                else:
                    ok = self._propagate(var, val, domains)

                if ok:
                    result = self._backtrack(assignment, domains)
                    if result is not None:
                        return result

                del assignment[var]
                for k in domains:
                    domains[k] = saved_domains[k]
                self.stats['backtracks'] += 1

        return None

    def _backtrack_all(self, assignment, domains, solutions, max_solutions):
        """Find all solutions."""
        if max_solutions and len(solutions) >= max_solutions:
            return

        if len(assignment) == len(self.variables):
            if self._is_consistent(assignment):
                solutions.append(dict(assignment))
            return

        self.stats['nodes'] += 1
        var = self._select_variable(assignment, domains)
        if var is None:
            return

        for val in self._order_values(var, domains):
            if self._is_value_consistent(var, val, assignment):
                assignment[var] = val
                saved_domains = {k: set(v) for k, v in domains.items()}
                domains[var] = {val}

                ok = self._propagate_and_ac3(var, val, domains)
                if ok:
                    self._backtrack_all(assignment, domains, solutions, max_solutions)

                del assignment[var]
                for k in domains:
                    domains[k] = saved_domains[k]
                self.stats['backtracks'] += 1

                if max_solutions and len(solutions) >= max_solutions:
                    return

    def _select_variable(self, assignment, domains):
        """MRV (Minimum Remaining Values) heuristic with degree tie-breaking."""
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None

        best = None
        best_size = float('inf')
        best_degree = -1

        for v in unassigned:
            size = len(domains[v])
            if size == 0:
                return None  # Wipeout detected
            degree = len(self.var_constraints.get(v, []))
            if size < best_size or (size == best_size and degree > best_degree):
                best = v
                best_size = size
                best_degree = degree

        return best

    def _order_values(self, var, domains):
        """LCV (Least Constraining Value) heuristic."""
        if len(domains[var]) <= 1:
            return list(domains[var])

        # Count how many values each choice eliminates
        scores = []
        for val in domains[var]:
            eliminations = 0
            for ci in self.var_constraints.get(var, []):
                c = self.constraints[ci]
                for other in c.variables:
                    if other != var and len(domains[other]) > 1:
                        for ov in domains[other]:
                            test = {var: val, other: ov}
                            if not c._check_partial(test):
                                eliminations += 1
            scores.append((eliminations, val))

        scores.sort()
        return [val for _, val in scores]

    def _propagate(self, var, val, domains):
        """Simple constraint propagation."""
        self.stats['propagations'] += 1
        for ci in self.var_constraints.get(var, []):
            c = self.constraints[ci]
            result = c.propagate(var, val, domains)
            if result is None:
                return False
        return True

    def _forward_check(self, var, val, domains):
        """Forward checking: remove inconsistent values from neighbors."""
        self.stats['propagations'] += 1
        for ci in self.var_constraints.get(var, []):
            c = self.constraints[ci]
            for other in c.variables:
                if other == var:
                    continue
                if len(domains[other]) == 1 and next(iter(domains[other])) in domains[other]:
                    continue  # Already assigned
                for ov in list(domains[other]):
                    test = {var: val, other: ov}
                    # Also include any other assigned vars
                    if not c._check_partial(test):
                        domains[other].discard(ov)
                if not domains[other]:
                    return False
        return True

    def _propagate_and_ac3(self, var, val, domains):
        """Propagate then enforce arc consistency."""
        if not self._propagate(var, val, domains):
            return False
        return self._ac3(domains, initial_var=var)

    def _ac3(self, domains, initial_var=None):
        """AC-3 arc consistency algorithm."""
        queue = deque()

        if initial_var:
            # Only arcs affected by initial_var
            for ci in self.var_constraints.get(initial_var, []):
                c = self.constraints[ci]
                for v in c.variables:
                    if v != initial_var:
                        queue.append((v, initial_var, c))
        else:
            # All arcs
            for c in self.constraints:
                for arc in c.get_arcs():
                    queue.append(arc)

        while queue:
            xi, xj, c = queue.popleft()
            if xi not in domains or xj not in domains:
                continue
            removed = c.revise(xi, xj, domains)
            if removed:
                if not domains[xi]:
                    return False
                # Add arcs (xk, xi) for all constraints involving xi
                for ci2 in self.var_constraints.get(xi, []):
                    c2 = self.constraints[ci2]
                    for xk in c2.variables:
                        if xk != xi and xk != xj:
                            queue.append((xk, xi, c2))
        return True

    def _is_consistent(self, assignment):
        """Check if assignment satisfies all constraints."""
        for c in self.constraints:
            if not c.satisfied(assignment):
                return False
        return True

    def _is_value_consistent(self, var, val, assignment):
        """Check if var=val is consistent with current assignment."""
        test = dict(assignment)
        test[var] = val
        for ci in self.var_constraints.get(var, []):
            if not self.constraints[ci].satisfied(test):
                return False
        return True

    # --- SAT Encoding ---

    def to_sat(self):
        """Encode CSP as SAT problem using direct encoding.
        Returns (SATSolver, decode_fn) or None if encoding fails."""
        sat = SATSolver()

        # Map: (var_name, value) -> SAT variable
        var_map = {}
        var_id = 0

        for name, variable in self.variables.items():
            for val in sorted(variable.domain):
                var_id += 1
                sat.new_var()
                var_map[(name, val)] = var_id

        # Reverse map for decoding
        rev_map = {v: k for k, v in var_map.items()}

        # At least one value per variable
        for name, variable in self.variables.items():
            clause = [var_map[(name, val)] for val in sorted(variable.domain)]
            sat.add_clause(clause)

        # At most one value per variable (pairwise)
        for name, variable in self.variables.items():
            vals = sorted(variable.domain)
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    sat.add_clause([-var_map[(name, vals[i])], -var_map[(name, vals[j])]])

        # Encode constraints
        for c in self.constraints:
            if isinstance(c, InequalityConstraint):
                x, y = c.variables
                for val in self.variables[x].domain:
                    if val in self.variables[y].domain:
                        if (x, val) in var_map and (y, val) in var_map:
                            sat.add_clause([-var_map[(x, val)], -var_map[(y, val)]])

            elif isinstance(c, EqualityConstraint):
                x, y = c.variables
                for vx in self.variables[x].domain:
                    for vy in self.variables[y].domain:
                        if vx != vy:
                            if (x, vx) in var_map and (y, vy) in var_map:
                                sat.add_clause([-var_map[(x, vx)], -var_map[(y, vy)]])

            elif isinstance(c, AllDifferentConstraint):
                for i in range(len(c.variables)):
                    for j in range(i + 1, len(c.variables)):
                        x, y = c.variables[i], c.variables[j]
                        common = self.variables[x].domain & self.variables[y].domain
                        for val in common:
                            if (x, val) in var_map and (y, val) in var_map:
                                sat.add_clause([-var_map[(x, val)], -var_map[(y, val)]])

            elif isinstance(c, TableConstraint):
                # For each variable value, it must appear in some allowed tuple
                x_vars = c.variables
                vals_list = [sorted(self.variables[v].domain) for v in x_vars]
                # Forbid all disallowed combinations
                for combo in cartesian_product(*vals_list):
                    if tuple(combo) not in c.allowed:
                        clause = [-var_map[(x_vars[i], combo[i])] for i in range(len(x_vars))
                                  if (x_vars[i], combo[i]) in var_map]
                        if clause:
                            sat.add_clause(clause)

            # For other constraint types, we use a general encoding
            elif isinstance(c, (ComparisonConstraint, ArithmeticConstraint, SumConstraint)):
                if len(c.variables) == 2:
                    x, y = c.variables
                    for vx in self.variables[x].domain:
                        for vy in self.variables[y].domain:
                            test = {x: vx, y: vy}
                            if not c.satisfied(test):
                                clause = []
                                if (x, vx) in var_map:
                                    clause.append(-var_map[(x, vx)])
                                if (y, vy) in var_map:
                                    clause.append(-var_map[(y, vy)])
                                if clause:
                                    sat.add_clause(clause)

        def decode(model):
            if model is None:
                return None
            assignment = {}
            for var_id, val in model.items():
                if val and var_id in rev_map:
                    name, value = rev_map[var_id]
                    assignment[name] = value
            return assignment

        return sat, decode

    def solve_with_sat(self):
        """Solve using SAT encoding."""
        result = self.to_sat()
        if result is None:
            return CSPResult.UNKNOWN, None
        sat, decode = result
        sat_result = sat.solve()
        if sat_result == SATResult.SAT:
            assignment = decode(sat.model())
            return CSPResult.SOLVED, assignment
        elif sat_result == SATResult.UNSAT:
            return CSPResult.UNSATISFIABLE, None
        return CSPResult.UNKNOWN, None

    # --- SMT Integration ---

    def solve_with_smt(self):
        """Solve using SMT solver (good for arithmetic constraints)."""
        smt = SMTSolver()
        smt_vars = {}

        for name, var in self.variables.items():
            smt_vars[name] = smt.Int(name)
            # Domain constraint: OR of equalities
            domain_list = sorted(var.domain)
            if len(domain_list) == 1:
                smt.add(smt_vars[name] == smt.IntVal(domain_list[0]))
            else:
                # x >= min and x <= max
                smt.add(smt_vars[name] >= smt.IntVal(min(domain_list)))
                smt.add(smt_vars[name] <= smt.IntVal(max(domain_list)))
                # If domain has gaps, add explicit membership
                full_range = set(range(min(domain_list), max(domain_list) + 1))
                excluded = full_range - var.domain
                for ex in excluded:
                    smt.add(smt_vars[name] != smt.IntVal(ex))

        for c in self.constraints:
            if isinstance(c, EqualityConstraint):
                x, y = c.variables
                smt.add(smt_vars[x] == smt_vars[y])

            elif isinstance(c, InequalityConstraint):
                x, y = c.variables
                smt.add(smt_vars[x] != smt_vars[y])

            elif isinstance(c, ComparisonConstraint):
                x, y = c.variables
                if c.op == '<': smt.add(smt_vars[x] < smt_vars[y])
                elif c.op == '<=': smt.add(smt_vars[x] <= smt_vars[y])
                elif c.op == '>': smt.add(smt_vars[x] > smt_vars[y])
                elif c.op == '>=': smt.add(smt_vars[x] >= smt_vars[y])

            elif isinstance(c, AllDifferentConstraint):
                smt.add(smt.Distinct(*[smt_vars[v] for v in c.variables]))

            elif isinstance(c, ArithmeticConstraint):
                # Build linear expression: coeff1*var1 + coeff2*var2 + ...
                terms = list(c.coeffs.items())
                # Start with first term to avoid IntVal(0) + ... nesting
                v0, c0 = terms[0]
                if c0 == 1:
                    expr = smt_vars[v0]
                elif c0 == -1:
                    expr = -smt_vars[v0]
                else:
                    expr = smt.IntVal(c0) * smt_vars[v0]
                for v, coeff in terms[1:]:
                    if coeff == 1:
                        expr = expr + smt_vars[v]
                    elif coeff == -1:
                        expr = expr - smt_vars[v]
                    else:
                        expr = expr + smt.IntVal(coeff) * smt_vars[v]
                rhs = smt.IntVal(c.rhs)
                if c.op == '==': smt.add(expr == rhs)
                elif c.op == '!=': smt.add(expr != rhs)
                elif c.op == '<': smt.add(expr < rhs)
                elif c.op == '<=': smt.add(expr <= rhs)
                elif c.op == '>': smt.add(expr > rhs)
                elif c.op == '>=': smt.add(expr >= rhs)

            elif isinstance(c, SumConstraint):
                expr = smt.IntVal(0)
                for v in c.variables:
                    expr = expr + smt_vars[v]
                smt.add(expr == smt.IntVal(c.target))

        result = smt.check()
        if result == SMTResult.SAT:
            model = smt.model()
            assignment = {}
            for name in self.variables:
                if name in model:
                    val = model[name]
                    if isinstance(val, (int, float)):
                        assignment[name] = int(val)
                    else:
                        assignment[name] = val
            return CSPResult.SOLVED, assignment
        elif result == SMTResult.UNSAT:
            return CSPResult.UNSATISFIABLE, None
        return CSPResult.UNKNOWN, None


# --- Modeling Helpers ---

def sudoku(grid):
    """Create Sudoku CSP from 9x9 grid (0 = empty).
    Returns (CSPSolver, decode_fn) where decode_fn converts assignment to 9x9 grid."""
    csp = CSPSolver()

    # Variables: cell_r_c with domain 1-9
    for r in range(9):
        for c in range(9):
            name = f"c{r}{c}"
            if grid[r][c] != 0:
                csp.add_variable(name, [grid[r][c]])
            else:
                csp.add_variable(name, range(1, 10))

    # Row constraints
    for r in range(9):
        csp.add_alldiff([f"c{r}{c}" for c in range(9)])

    # Column constraints
    for c in range(9):
        csp.add_alldiff([f"c{r}{c}" for r in range(9)])

    # Box constraints
    for br in range(3):
        for bc in range(3):
            cells = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    cells.append(f"c{r}{c}")
            csp.add_alldiff(cells)

    def decode(assignment):
        if assignment is None:
            return None
        result = [[0] * 9 for _ in range(9)]
        for r in range(9):
            for c in range(9):
                result[r][c] = assignment[f"c{r}{c}"]
        return result

    return csp, decode


def n_queens(n):
    """Create N-Queens CSP.
    Variables: q0..q(n-1), each with domain 0..n-1 (column for each row).
    Returns (CSPSolver, decode_fn)."""
    csp = CSPSolver()

    for i in range(n):
        csp.add_variable(f"q{i}", range(n))

    # All different columns
    csp.add_alldiff([f"q{i}" for i in range(n)])

    # Diagonal constraints
    for i in range(n):
        for j in range(i + 1, n):
            diff = j - i
            # |qi - qj| != |i - j|
            csp.add_callback(
                [f"q{i}", f"q{j}"],
                lambda a, qi=f"q{i}", qj=f"q{j}", d=diff: abs(a[qi] - a[qj]) != d
            )

    def decode(assignment):
        if assignment is None:
            return None
        return [assignment[f"q{i}"] for i in range(n)]

    return csp, decode


def graph_coloring(edges, num_nodes, num_colors):
    """Create graph coloring CSP.
    Returns (CSPSolver, decode_fn)."""
    csp = CSPSolver()

    for i in range(num_nodes):
        csp.add_variable(f"n{i}", range(num_colors))

    for u, v in edges:
        csp.add_inequality(f"n{u}", f"n{v}")

    def decode(assignment):
        if assignment is None:
            return None
        return {i: assignment[f"n{i}"] for i in range(num_nodes)}

    return csp, decode


def scheduling(tasks, precedences=None, deadlines=None, durations=None, num_slots=None):
    """Create scheduling CSP.
    tasks: list of task names
    precedences: list of (before, after) pairs
    deadlines: dict task_name -> max_slot
    durations: dict task_name -> duration (default 1)
    num_slots: total number of time slots
    Returns (CSPSolver, decode_fn)."""
    if durations is None:
        durations = {t: 1 for t in tasks}
    if num_slots is None:
        num_slots = sum(durations.get(t, 1) for t in tasks)

    csp = CSPSolver()

    for t in tasks:
        dur = durations.get(t, 1)
        max_start = num_slots - dur
        csp.add_variable(t, range(0, max_start + 1))

    # Precedence constraints: before must finish before after starts
    if precedences:
        for before, after in precedences:
            dur_before = durations.get(before, 1)
            # after >= before + dur_before
            csp.add_callback(
                [before, after],
                lambda a, b=before, af=after, d=dur_before: a[af] >= a[b] + d
            )

    # Deadline constraints
    if deadlines:
        for t, deadline in deadlines.items():
            dur = durations.get(t, 1)
            # t + dur <= deadline => t <= deadline - dur
            max_start = deadline - dur
            # Restrict domain
            old_domain = csp.variables[t].domain
            csp.variables[t].domain = {v for v in old_domain if v <= max_start}

    # No-overlap constraints for tasks with duration > 0
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t1, t2 = tasks[i], tasks[j]
            d1, d2 = durations.get(t1, 1), durations.get(t2, 1)
            csp.add_callback(
                [t1, t2],
                lambda a, x=t1, y=t2, dx=d1, dy=d2:
                    a[x] + dx <= a[y] or a[y] + dy <= a[x]
            )

    def decode(assignment):
        if assignment is None:
            return None
        schedule = {}
        for t in tasks:
            start = assignment[t]
            dur = durations.get(t, 1)
            schedule[t] = (start, start + dur)
        return schedule

    return csp, decode


def magic_square(n):
    """Create magic square CSP.
    n x n grid with values 1..n^2, rows/cols/diags sum to magic constant.
    Returns (CSPSolver, decode_fn)."""
    magic_const = n * (n * n + 1) // 2
    csp = CSPSolver()

    # Variables
    names = []
    for r in range(n):
        row_names = []
        for c in range(n):
            name = f"m{r}{c}"
            csp.add_variable(name, range(1, n * n + 1))
            row_names.append(name)
        names.append(row_names)

    # All different
    all_vars = [name for row in names for name in row]
    csp.add_alldiff(all_vars)

    # Row sums
    for r in range(n):
        csp.add_sum(names[r], magic_const)

    # Column sums
    for c in range(n):
        csp.add_sum([names[r][c] for r in range(n)], magic_const)

    # Diagonal sums
    csp.add_sum([names[i][i] for i in range(n)], magic_const)
    csp.add_sum([names[i][n - 1 - i] for i in range(n)], magic_const)

    def decode(assignment):
        if assignment is None:
            return None
        return [[assignment[names[r][c]] for c in range(n)] for r in range(n)]

    return csp, decode


def latin_square(n, partial=None):
    """Create Latin square CSP.
    n x n grid where each row and column contains 1..n exactly once.
    partial: dict of (r, c) -> value for pre-filled cells.
    Returns (CSPSolver, decode_fn)."""
    csp = CSPSolver()

    names = []
    for r in range(n):
        row_names = []
        for c in range(n):
            name = f"l{r}{c}"
            if partial and (r, c) in partial:
                csp.add_variable(name, [partial[(r, c)]])
            else:
                csp.add_variable(name, range(1, n + 1))
            row_names.append(name)
        names.append(row_names)

    # Row alldiff
    for r in range(n):
        csp.add_alldiff(names[r])

    # Column alldiff
    for c in range(n):
        csp.add_alldiff([names[r][c] for r in range(n)])

    def decode(assignment):
        if assignment is None:
            return None
        return [[assignment[names[r][c]] for c in range(n)] for r in range(n)]

    return csp, decode


def knapsack(items, capacity):
    """Create 0/1 knapsack as CSP.
    items: list of (weight, value) tuples
    capacity: max weight
    Returns (CSPSolver, decode_fn) -- finds feasible solutions, not optimal."""
    csp = CSPSolver()

    for i in range(len(items)):
        csp.add_variable(f"x{i}", [0, 1])

    # Weight constraint: sum(x_i * w_i) <= capacity
    coeffs = {f"x{i}": items[i][0] for i in range(len(items))}
    csp.add_arithmetic(coeffs, '<=', capacity)

    def decode(assignment):
        if assignment is None:
            return None
        selected = [i for i in range(len(items)) if assignment[f"x{i}"] == 1]
        total_weight = sum(items[i][0] for i in selected)
        total_value = sum(items[i][1] for i in selected)
        return {'selected': selected, 'weight': total_weight, 'value': total_value}

    return csp, decode
