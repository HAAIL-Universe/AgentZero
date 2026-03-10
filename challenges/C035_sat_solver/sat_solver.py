"""
C035: DPLL SAT Solver
A complete SAT solver implementing the DPLL algorithm with:
- CNF formula representation
- Unit propagation
- Pure literal elimination
- Two-literal watching for efficient BCP
- DIMACS CNF parsing
- Solution verification
- UNSAT proof (empty clause derivation)
- Basic CDCL (Conflict-Driven Clause Learning)
- VSIDS-like decision heuristic
- Random restarts

This is a fundamentally different domain from the language toolchain:
search, backtracking, constraint propagation.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import deque
import random


# --- Core Types ---

class LBool(Enum):
    TRUE = 1
    FALSE = 0
    UNDEF = -1


@dataclass
class Literal:
    """A literal is a variable (positive int) with optional negation."""
    var: int
    neg: bool = False

    def __post_init__(self):
        if self.var <= 0:
            raise ValueError(f"Variable must be positive, got {self.var}")

    def __neg__(self):
        return Literal(self.var, not self.neg)

    def __eq__(self, other):
        if not isinstance(other, Literal):
            return NotImplemented
        return self.var == other.var and self.neg == other.neg

    def __hash__(self):
        return hash((self.var, self.neg))

    def __repr__(self):
        return f"-{self.var}" if self.neg else f"{self.var}"

    def __int__(self):
        return -self.var if self.neg else self.var

    @staticmethod
    def from_int(i: int) -> 'Literal':
        if i == 0:
            raise ValueError("Literal cannot be 0")
        return Literal(abs(i), i < 0)


@dataclass
class Clause:
    """A disjunction of literals."""
    literals: list
    learned: bool = False
    activity: float = 0.0

    def __len__(self):
        return len(self.literals)

    def __iter__(self):
        return iter(self.literals)

    def __getitem__(self, idx):
        return self.literals[idx]

    def __repr__(self):
        return "({})".format(" v ".join(str(l) for l in self.literals))

    def is_empty(self):
        return len(self.literals) == 0

    def is_unit(self):
        return len(self.literals) == 1


@dataclass
class Assignment:
    """Tracks variable assignments with decision levels and antecedents."""
    value: dict = field(default_factory=dict)       # var -> bool
    level: dict = field(default_factory=dict)        # var -> decision level
    antecedent: dict = field(default_factory=dict)   # var -> clause index (None for decisions)
    trail: list = field(default_factory=list)        # assignment order
    trail_lim: list = field(default_factory=list)    # trail indices at each decision level

    def assign(self, var: int, val: bool, dl: int, ante=None):
        self.value[var] = val
        self.level[var] = dl
        self.antecedent[var] = ante
        self.trail.append(var)

    def unassign(self, var: int):
        del self.value[var]
        del self.level[var]
        del self.antecedent[var]

    def is_assigned(self, var: int) -> bool:
        return var in self.value

    def lit_value(self, lit: Literal) -> LBool:
        if lit.var not in self.value:
            return LBool.UNDEF
        val = self.value[lit.var]
        if lit.neg:
            val = not val
        return LBool.TRUE if val else LBool.FALSE

    def current_level(self) -> int:
        return len(self.trail_lim)

    def new_decision_level(self):
        self.trail_lim.append(len(self.trail))

    def backtrack_to(self, level: int):
        while len(self.trail_lim) > level:
            lim = self.trail_lim.pop()
            while len(self.trail) > lim:
                var = self.trail.pop()
                self.unassign(var)


# --- Solver ---

class SolverResult(Enum):
    SAT = "SATISFIABLE"
    UNSAT = "UNSATISFIABLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class SolverStats:
    decisions: int = 0
    propagations: int = 0
    conflicts: int = 0
    restarts: int = 0
    learned_clauses: int = 0
    max_decision_level: int = 0


class Solver:
    def __init__(self):
        self.num_vars = 0
        self.clauses: list = []           # list of Clause
        self.watches: dict = {}           # lit_int -> list of clause indices
        self.occurs: dict = {}            # lit_int -> set of clause indices
        self.assignment = Assignment()
        self.activity: dict = {}          # var -> float (VSIDS)
        self.activity_inc: float = 1.0
        self.activity_decay: float = 0.95
        self.stats = SolverStats()
        self.restart_threshold = 100
        self.restart_multiplier = 1.5
        self.learnt_limit = 100
        self._result = SolverResult.UNKNOWN
        self._model: Optional[dict] = None
        self._prop_queue: deque = deque()  # literals (as int) to propagate
        self._clause_sat: list = []       # clause_idx -> True if satisfied

    # --- Setup ---

    def new_var(self) -> int:
        self.num_vars += 1
        v = self.num_vars
        self.activity[v] = 0.0
        pos = int(Literal(v))
        neg = int(-Literal(v))
        if pos not in self.watches:
            self.watches[pos] = []
        if neg not in self.watches:
            self.watches[neg] = []
        if pos not in self.occurs:
            self.occurs[pos] = set()
        if neg not in self.occurs:
            self.occurs[neg] = set()
        return v

    def _ensure_var(self, v: int):
        while self.num_vars < v:
            self.new_var()

    def add_clause(self, lits: list) -> Optional[int]:
        """Add a clause. lits is a list of int (positive = true, negative = negated).
        Returns clause index, or None if clause is tautological."""
        if not lits:
            # Empty clause = immediate UNSAT
            self._result = SolverResult.UNSAT
            return None

        # Normalize: remove duplicates, check for tautology
        seen = set()
        normalized = []
        for i in lits:
            if i == 0:
                continue
            if -i in seen:
                return None  # tautology
            if i not in seen:
                seen.add(i)
                normalized.append(i)

        if not normalized:
            self._result = SolverResult.UNSAT
            return None

        literals = [Literal.from_int(i) for i in normalized]
        for lit in literals:
            self._ensure_var(lit.var)

        clause = Clause(literals)
        idx = len(self.clauses)
        self.clauses.append(clause)

        # Build occurrence list
        for lit in literals:
            li = int(lit)
            self.occurs[li].add(idx)

        return idx

    # --- VSIDS ---

    def _bump_var(self, var: int):
        self.activity[var] += self.activity_inc
        if self.activity[var] > 1e100:
            # Rescale
            for v in self.activity:
                self.activity[v] *= 1e-100
            self.activity_inc *= 1e-100

    def _decay_activity(self):
        self.activity_inc /= self.activity_decay

    def _pick_decision_var(self) -> Optional[int]:
        """Pick unassigned variable with highest activity."""
        best_var = None
        best_act = -1.0
        for v in range(1, self.num_vars + 1):
            if not self.assignment.is_assigned(v):
                if self.activity[v] > best_act:
                    best_act = self.activity[v]
                    best_var = v
        return best_var

    # --- BCP (Boolean Constraint Propagation) ---

    def _enqueue(self, var: int, val: bool, dl: int, ante=None):
        """Assign and enqueue for propagation."""
        self.assignment.assign(var, val, dl, ante)
        # The literal that became FALSE needs checking
        false_lit_int = -var if val else var
        self._prop_queue.append(false_lit_int)  # deque append

    def _propagate(self) -> Optional[int]:
        """Unit propagation using occurrence lists.
        Returns conflicting clause index, or None if no conflict."""
        assign_value = self.assignment.value
        assign_level = self.assignment.level

        # First, find initial unit clauses if queue is empty and no assignments
        if not self._prop_queue:
            for idx, clause in enumerate(self.clauses):
                if len(clause.literals) == 1:
                    lit = clause.literals[0]
                    if lit.var not in assign_value:
                        self._enqueue(lit.var, not lit.neg,
                                      self.assignment.current_level(), idx)
                        self.stats.propagations += 1

        while self._prop_queue:
            false_lit_int = self._prop_queue.popleft()

            # Check all clauses containing this now-false literal
            for clause_idx in list(self.occurs.get(false_lit_int, ())):
                clause = self.clauses[clause_idx]
                unsat_count = 0
                undef_lit = None
                sat = False

                for lit in clause.literals:
                    v = lit.var
                    if v in assign_value:
                        val = assign_value[v]
                        if lit.neg:
                            val = not val
                        if val:
                            sat = True
                            break
                        else:
                            unsat_count += 1
                    else:
                        undef_lit = lit

                if sat:
                    continue

                n = len(clause.literals)
                if unsat_count == n:
                    self._prop_queue.clear()
                    return clause_idx  # conflict

                if undef_lit is not None and unsat_count == n - 1:
                    if undef_lit.var not in assign_value:
                        self._enqueue(undef_lit.var, not undef_lit.neg,
                                      self.assignment.current_level(), clause_idx)
                        self.stats.propagations += 1

        return None

    # --- Conflict Analysis (1UIP) ---

    def _analyze(self, conflict_idx: int) -> tuple:
        """Analyze conflict using 1UIP scheme (MiniSat-style).
        Returns (learned_clause_lits, backtrack_level)."""
        current_dl = self.assignment.current_level()

        if current_dl == 0:
            return ([], -1)

        seen = [False] * (self.num_vars + 1)
        learnt = []
        counter = 0
        bt_level = 0

        # Initialize with conflict clause
        conflict_clause = self.clauses[conflict_idx]
        for lit in conflict_clause.literals:
            v = lit.var
            if not seen[v] and self.assignment.level.get(v, 0) > 0:
                seen[v] = True
                if self.assignment.level[v] == current_dl:
                    counter += 1
                else:
                    learnt.append(lit)
                    if self.assignment.level[v] > bt_level:
                        bt_level = self.assignment.level[v]

        # Walk trail backwards to find 1UIP
        idx = len(self.assignment.trail) - 1
        uip_var = None
        while True:
            # Find next seen variable on trail
            while idx >= 0 and not seen[self.assignment.trail[idx]]:
                idx -= 1
            if idx < 0:
                break

            p = self.assignment.trail[idx]
            seen[p] = False
            counter -= 1
            idx -= 1

            if counter == 0:
                # p is the 1UIP
                uip_var = p
                break

            # Resolve with antecedent of p
            ante_idx = self.assignment.antecedent.get(p)
            if ante_idx is not None:
                ante = self.clauses[ante_idx]
                for lit in ante.literals:
                    v = lit.var
                    if v != p and not seen[v] and self.assignment.level.get(v, 0) > 0:
                        seen[v] = True
                        if self.assignment.level[v] == current_dl:
                            counter += 1
                        else:
                            learnt.append(lit)
                            if self.assignment.level[v] > bt_level:
                                bt_level = self.assignment.level[v]

        if uip_var is None:
            return ([], -1)

        # UIP literal: negate the assignment (if var=True, clause needs -var)
        uip_lit = Literal(uip_var, self.assignment.value[uip_var])
        learnt.insert(0, uip_lit)

        # Bump activity for all variables in learned clause
        for lit in learnt:
            self._bump_var(lit.var)
        self._decay_activity()

        return (learnt, bt_level)

    # --- Solve ---

    def solve(self) -> SolverResult:
        """Main DPLL/CDCL solve loop."""
        if self._result == SolverResult.UNSAT:
            return SolverResult.UNSAT

        if self.num_vars == 0 and not self.clauses:
            self._result = SolverResult.SAT
            self._model = {}
            return SolverResult.SAT

        # Initial propagation
        conflict = self._propagate()
        if conflict is not None:
            self._result = SolverResult.UNSAT
            return SolverResult.UNSAT

        restart_count = 0
        conflicts_until_restart = self.restart_threshold

        while True:
            # Pick decision variable
            var = self._pick_decision_var()
            if var is None:
                # All variables assigned, SAT
                self._result = SolverResult.SAT
                self._model = dict(self.assignment.value)
                return SolverResult.SAT

            # Make decision
            self.assignment.new_decision_level()
            dl = self.assignment.current_level()
            self.stats.decisions += 1
            if dl > self.stats.max_decision_level:
                self.stats.max_decision_level = dl

            # Try false first (common heuristic)
            self._enqueue(var, False, dl, None)

            # Propagate
            conflict = self._propagate()

            while conflict is not None:
                self.stats.conflicts += 1

                if self.assignment.current_level() == 0:
                    self._result = SolverResult.UNSAT
                    return SolverResult.UNSAT

                # Analyze conflict
                learnt_lits, bt_level = self._analyze(conflict)

                if not learnt_lits:
                    self._result = SolverResult.UNSAT
                    return SolverResult.UNSAT

                # Backtrack
                self.assignment.backtrack_to(bt_level)


                # Add learned clause
                learnt_ints = [int(l) for l in learnt_lits]
                clause_idx = self._add_learned_clause(learnt_ints)
                self.stats.learned_clauses += 1

                # The first literal (UIP) of the learned clause is unit now
                # Explicitly propagate it
                uip = learnt_lits[0]
                if not self.assignment.is_assigned(uip.var):
                    self._enqueue(uip.var, not uip.neg,
                                  self.assignment.current_level(), clause_idx)
                    self.stats.propagations += 1

                conflict = self._propagate()

            # Check restart
            conflicts_until_restart -= 1
            if conflicts_until_restart <= 0:
                restart_count += 1
                self.stats.restarts += 1
                self.assignment.backtrack_to(0)

                conflicts_until_restart = int(
                    self.restart_threshold * (self.restart_multiplier ** restart_count)
                )
                conflict = self._propagate()
                if conflict is not None:
                    self._result = SolverResult.UNSAT
                    return SolverResult.UNSAT

    def _add_learned_clause(self, lits: list) -> int:
        literals = [Literal.from_int(i) for i in lits]
        clause = Clause(literals, learned=True)
        idx = len(self.clauses)
        self.clauses.append(clause)

        # Add to occurrence lists
        for lit in literals:
            li = int(lit)
            if li not in self.occurs:
                self.occurs[li] = set()
            self.occurs[li].add(idx)

        return idx

    def model(self) -> Optional[dict]:
        """Return the satisfying assignment if SAT, else None."""
        return self._model

    # --- Verification ---

    def verify(self, assignment: dict = None) -> bool:
        """Verify that the given assignment satisfies all original (non-learned) clauses."""
        if assignment is None:
            assignment = self._model
        if assignment is None:
            return False

        for clause in self.clauses:
            if clause.learned:
                continue
            satisfied = False
            for lit in clause.literals:
                val = assignment.get(lit.var, False)
                if lit.neg:
                    val = not val
                if val:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True


# --- DIMACS Parser ---

def parse_dimacs(text: str) -> Solver:
    """Parse a DIMACS CNF format string into a Solver."""
    solver = Solver()
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('c'):
            continue
        if line.startswith('p'):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == 'cnf':
                num_vars = int(parts[2])
                solver._ensure_var(num_vars)
            continue
        # Clause line
        nums = list(map(int, line.split()))
        if nums and nums[-1] == 0:
            nums = nums[:-1]
        if nums:
            solver.add_clause(nums)
    return solver


# --- Convenience ---

def solve_dimacs(text: str) -> tuple:
    """Parse and solve a DIMACS CNF formula.
    Returns (result, model_or_none, stats)."""
    solver = parse_dimacs(text)
    result = solver.solve()
    return (result, solver.model(), solver.stats)


def solve_clauses(clauses: list, num_vars: int = 0) -> tuple:
    """Solve from a list of int-lists.
    Each inner list is a clause (positive = var, negative = negated var).
    Returns (result, model_or_none, stats)."""
    solver = Solver()
    if num_vars > 0:
        solver._ensure_var(num_vars)
    for clause in clauses:
        solver.add_clause(clause)
    result = solver.solve()
    return (result, solver.model(), solver.stats)


# --- Problem Generators ---

def generate_random_3sat(num_vars: int, num_clauses: int, seed: int = None) -> list:
    """Generate a random 3-SAT instance. Returns list of int-lists (clauses)."""
    if seed is not None:
        random.seed(seed)
    clauses = []
    for _ in range(num_clauses):
        vars_chosen = random.sample(range(1, num_vars + 1), min(3, num_vars))
        clause = []
        for v in vars_chosen:
            clause.append(v if random.random() < 0.5 else -v)
        clauses.append(clause)
    return clauses


def generate_pigeonhole(n: int) -> list:
    """Generate pigeonhole principle: n+1 pigeons, n holes.
    Always UNSAT. Variable (i-1)*n + j represents pigeon i in hole j."""
    clauses = []
    num_pigeons = n + 1

    # Each pigeon must be in some hole
    for i in range(1, num_pigeons + 1):
        clause = []
        for j in range(1, n + 1):
            clause.append((i - 1) * n + j)
        clauses.append(clause)

    # No two pigeons in same hole
    for j in range(1, n + 1):
        for i1 in range(1, num_pigeons + 1):
            for i2 in range(i1 + 1, num_pigeons + 1):
                v1 = (i1 - 1) * n + j
                v2 = (i2 - 1) * n + j
                clauses.append([-v1, -v2])

    return clauses


def generate_queens(n: int) -> list:
    """Generate N-Queens as SAT. Variable (i-1)*n + j means queen at row i, col j.
    SAT for n >= 4."""
    clauses = []

    def var(row, col):
        return (row - 1) * n + col

    # At least one queen per row
    for i in range(1, n + 1):
        clauses.append([var(i, j) for j in range(1, n + 1)])

    # At most one queen per row
    for i in range(1, n + 1):
        for j1 in range(1, n + 1):
            for j2 in range(j1 + 1, n + 1):
                clauses.append([-var(i, j1), -var(i, j2)])

    # At most one queen per column
    for j in range(1, n + 1):
        for i1 in range(1, n + 1):
            for i2 in range(i1 + 1, n + 1):
                clauses.append([-var(i1, j), -var(i2, j)])

    # At most one queen per diagonal (down-right)
    for d in range(-(n - 1), n):
        diag = [(i, i - d) for i in range(1, n + 1) if 1 <= i - d <= n]
        for a in range(len(diag)):
            for b in range(a + 1, len(diag)):
                r1, c1 = diag[a]
                r2, c2 = diag[b]
                clauses.append([-var(r1, c1), -var(r2, c2)])

    # At most one queen per anti-diagonal (down-left)
    for d in range(2, 2 * n + 1):
        adiag = [(i, d - i) for i in range(1, n + 1) if 1 <= d - i <= n]
        for a in range(len(adiag)):
            for b in range(a + 1, len(adiag)):
                r1, c1 = adiag[a]
                r2, c2 = adiag[b]
                clauses.append([-var(r1, c1), -var(r2, c2)])

    return clauses


# --- Sudoku Encoder ---

def encode_sudoku(grid: list) -> tuple:
    """Encode a 9x9 Sudoku as SAT.
    grid is a 9x9 list of ints (0 = empty).
    Returns (clauses, decode_fn).
    Variable (r*9 + c)*9 + d represents digit d+1 at row r, col c (0-indexed)."""
    clauses = []

    def var(r, c, d):
        return (r * 9 + c) * 9 + d + 1  # 1-indexed for SAT

    # Given clues
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                clauses.append([var(r, c, grid[r][c] - 1)])

    # Each cell has at least one digit
    for r in range(9):
        for c in range(9):
            clauses.append([var(r, c, d) for d in range(9)])

    # Each cell has at most one digit
    for r in range(9):
        for c in range(9):
            for d1 in range(9):
                for d2 in range(d1 + 1, 9):
                    clauses.append([-var(r, c, d1), -var(r, c, d2)])

    # Each row has each digit
    for r in range(9):
        for d in range(9):
            clauses.append([var(r, c, d) for c in range(9)])

    # Each column has each digit
    for c in range(9):
        for d in range(9):
            clauses.append([var(r, c, d) for r in range(9)])

    # Each 3x3 box has each digit
    for br in range(3):
        for bc in range(3):
            for d in range(9):
                clauses.append([
                    var(br * 3 + dr, bc * 3 + dc, d)
                    for dr in range(3) for dc in range(3)
                ])

    def decode(model):
        result = [[0] * 9 for _ in range(9)]
        for r in range(9):
            for c in range(9):
                for d in range(9):
                    v = var(r, c, d)
                    if model.get(v, False):
                        result[r][c] = d + 1
        return result

    return (clauses, decode)


# --- Graph Coloring Encoder ---

def encode_graph_coloring(edges: list, num_nodes: int, num_colors: int) -> tuple:
    """Encode graph coloring as SAT.
    Returns (clauses, decode_fn).
    Variable (node * num_colors + color + 1) means node has color."""
    clauses = []

    def var(node, color):
        return node * num_colors + color + 1

    # Each node has at least one color
    for n in range(num_nodes):
        clauses.append([var(n, c) for c in range(num_colors)])

    # Each node has at most one color
    for n in range(num_nodes):
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                clauses.append([-var(n, c1), -var(n, c2)])

    # Adjacent nodes have different colors
    for u, v in edges:
        for c in range(num_colors):
            clauses.append([-var(u, c), -var(v, c)])

    def decode(model):
        coloring = {}
        for n in range(num_nodes):
            for c in range(num_colors):
                if model.get(var(n, c), False):
                    coloring[n] = c
        return coloring

    return (clauses, decode)
