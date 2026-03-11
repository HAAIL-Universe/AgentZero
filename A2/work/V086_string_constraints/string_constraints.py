"""V086: String Constraint Solver

Solves string constraints using automata-based reasoning (V081 SFA) combined
with integer length reasoning (C037 SMT). Supports:
- Regex membership: x in R
- Word equations: x . y = z (concatenation)
- Length constraints: len(x) = n, len(x) >= n
- String equality/inequality: x = "abc", x != y
- Contains/prefix/suffix: x contains "ab", x startswith "pre"
- Character-at constraints: x[i] = 'a'

Architecture:
- Each string variable is tracked as an SFA (over-approximation of possible values)
- Regex constraints narrow the SFA via intersection
- Word equations solved via product construction
- Length constraints bridge to C037 SMT integer reasoning
- Satisfying assignments extracted via SFA accepted word generation
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Set, Any

# Add dependency paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V081_symbolic_automata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V084_symbolic_regex'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from symbolic_automata import (
    SFA, SFATransition, CharAlgebra, Pred, PredKind,
    PTrue, PFalse, PChar, PRange, PAnd, POr, PNot,
    sfa_intersection, sfa_union, sfa_complement, sfa_difference,
    sfa_is_equivalent, sfa_is_subset, sfa_from_string, sfa_from_char_class,
    sfa_from_range, sfa_any_char, sfa_epsilon, sfa_empty,
    sfa_concat, sfa_star, sfa_plus, sfa_optional,
    shortest_accepted, sfa_stats,
)
from symbolic_regex import (
    compile_regex, compile_regex_dfa, parse_regex, regex_equivalent,
)
from smt_solver import SMTSolver, SMTResult, Var as SMTVar, IntConst, App, Op, INT, BOOL


# ---------------------------------------------------------------------------
# Constraint AST
# ---------------------------------------------------------------------------

class ConstraintKind(Enum):
    """Types of string constraints."""
    REGEX = auto()          # x in R (regex membership)
    EQUALS_CONST = auto()   # x = "abc"
    NOT_EQUALS_CONST = auto()  # x != "abc"
    EQUALS_VAR = auto()     # x = y
    NOT_EQUALS_VAR = auto() # x != y
    CONCAT = auto()         # x . y = z
    LENGTH_EQ = auto()      # len(x) = n
    LENGTH_LE = auto()      # len(x) <= n
    LENGTH_GE = auto()      # len(x) >= n
    LENGTH_RANGE = auto()   # lo <= len(x) <= hi
    CONTAINS = auto()       # x contains "sub"
    PREFIX = auto()         # x starts with "pre"
    SUFFIX = auto()         # x ends with "suf"
    CHAR_AT = auto()        # x[i] = c
    IN_SET = auto()         # x in {"a", "b", "c"} (finite set)
    NOT_EMPTY = auto()      # x != ""


@dataclass(frozen=True)
class StringVar:
    """A string variable."""
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class StringConstraint:
    """A constraint on string variables."""
    kind: ConstraintKind
    var: str                          # primary variable name
    var2: Optional[str] = None        # secondary variable (for equals_var, concat)
    var3: Optional[str] = None        # tertiary variable (for concat: var.var2 = var3)
    pattern: Optional[str] = None     # regex pattern or string constant
    value: Optional[int] = None       # integer value (for length)
    value2: Optional[int] = None      # second integer (for length_range hi)
    char: Optional[str] = None        # character (for char_at)
    strings: Optional[tuple] = None   # string set (for in_set)


# ---------------------------------------------------------------------------
# Constraint constructors (convenience API)
# ---------------------------------------------------------------------------

def str_var(name: str) -> StringVar:
    """Create a string variable."""
    return StringVar(name)


def regex_constraint(var: str, pattern: str) -> StringConstraint:
    """x in L(pattern)"""
    return StringConstraint(kind=ConstraintKind.REGEX, var=var, pattern=pattern)


def equals_const(var: str, value: str) -> StringConstraint:
    """x = value"""
    return StringConstraint(kind=ConstraintKind.EQUALS_CONST, var=var, pattern=value)


def not_equals_const(var: str, value: str) -> StringConstraint:
    """x != value"""
    return StringConstraint(kind=ConstraintKind.NOT_EQUALS_CONST, var=var, pattern=value)


def equals_var(var1: str, var2: str) -> StringConstraint:
    """x = y"""
    return StringConstraint(kind=ConstraintKind.EQUALS_VAR, var=var1, var2=var2)


def not_equals_var(var1: str, var2: str) -> StringConstraint:
    """x != y"""
    return StringConstraint(kind=ConstraintKind.NOT_EQUALS_VAR, var=var1, var2=var2)


def concat_eq(x: str, y: str, z: str) -> StringConstraint:
    """x . y = z"""
    return StringConstraint(kind=ConstraintKind.CONCAT, var=x, var2=y, var3=z)


def length_eq(var: str, n: int) -> StringConstraint:
    """len(x) = n"""
    return StringConstraint(kind=ConstraintKind.LENGTH_EQ, var=var, value=n)


def length_le(var: str, n: int) -> StringConstraint:
    """len(x) <= n"""
    return StringConstraint(kind=ConstraintKind.LENGTH_LE, var=var, value=n)


def length_ge(var: str, n: int) -> StringConstraint:
    """len(x) >= n"""
    return StringConstraint(kind=ConstraintKind.LENGTH_GE, var=var, value=n)


def length_range(var: str, lo: int, hi: int) -> StringConstraint:
    """lo <= len(x) <= hi"""
    return StringConstraint(kind=ConstraintKind.LENGTH_RANGE, var=var, value=lo, value2=hi)


def contains(var: str, substring: str) -> StringConstraint:
    """x contains substring"""
    return StringConstraint(kind=ConstraintKind.CONTAINS, var=var, pattern=substring)


def prefix(var: str, pre: str) -> StringConstraint:
    """x starts with pre"""
    return StringConstraint(kind=ConstraintKind.PREFIX, var=var, pattern=pre)


def suffix(var: str, suf: str) -> StringConstraint:
    """x ends with suf"""
    return StringConstraint(kind=ConstraintKind.SUFFIX, var=var, pattern=suf)


def char_at(var: str, index: int, c: str) -> StringConstraint:
    """x[index] = c"""
    return StringConstraint(kind=ConstraintKind.CHAR_AT, var=var, value=index, char=c)


def in_set(var: str, strings: list) -> StringConstraint:
    """x in {s1, s2, ...}"""
    return StringConstraint(kind=ConstraintKind.IN_SET, var=var, strings=tuple(strings))


def not_empty(var: str) -> StringConstraint:
    """x != '' """
    return StringConstraint(kind=ConstraintKind.NOT_EMPTY, var=var)


# ---------------------------------------------------------------------------
# SFA builders for constraints
# ---------------------------------------------------------------------------

def _sigma_star(algebra: CharAlgebra) -> SFA:
    """Build Sigma* -- accepts any string."""
    return SFA(
        states={0},
        initial=0,
        accepting={0},
        transitions=[SFATransition(0, PTrue(), 0)],
        algebra=algebra,
    )


def _length_exactly(n: int, algebra: CharAlgebra) -> SFA:
    """SFA accepting strings of exactly length n."""
    if n < 0:
        return sfa_empty(algebra)
    states = set(range(n + 1))
    transitions = [SFATransition(i, PTrue(), i + 1) for i in range(n)]
    return SFA(states=states, initial=0, accepting={n}, transitions=transitions, algebra=algebra)


def _length_at_most(n: int, algebra: CharAlgebra) -> SFA:
    """SFA accepting strings of length <= n."""
    if n < 0:
        return sfa_empty(algebra)
    states = set(range(n + 1))
    accepting = set(range(n + 1))
    transitions = [SFATransition(i, PTrue(), i + 1) for i in range(n)]
    return SFA(states=states, initial=0, accepting=accepting, transitions=transitions, algebra=algebra)


def _length_at_least(n: int, algebra: CharAlgebra) -> SFA:
    """SFA accepting strings of length >= n."""
    if n <= 0:
        return _sigma_star(algebra)
    # States 0..n, state n is accepting with self-loop
    states = set(range(n + 1))
    transitions = [SFATransition(i, PTrue(), i + 1) for i in range(n)]
    transitions.append(SFATransition(n, PTrue(), n))  # self-loop
    return SFA(states=states, initial=0, accepting={n}, transitions=transitions, algebra=algebra)


def _length_between(lo: int, hi: int, algebra: CharAlgebra) -> SFA:
    """SFA accepting strings with lo <= length <= hi."""
    ge = _length_at_least(lo, algebra)
    le = _length_at_most(hi, algebra)
    return sfa_intersection(ge, le)


def _contains_sfa(substring: str, algebra: CharAlgebra) -> SFA:
    """SFA accepting strings containing substring. Uses Sigma* . literal . Sigma*."""
    if not substring:
        return _sigma_star(algebra)
    lit = sfa_from_string(substring, algebra)
    star = _sigma_star(algebra)
    return sfa_concat(sfa_concat(star, lit), star)


def _prefix_sfa(pre: str, algebra: CharAlgebra) -> SFA:
    """SFA accepting strings starting with pre."""
    if not pre:
        return _sigma_star(algebra)
    lit = sfa_from_string(pre, algebra)
    star = _sigma_star(algebra)
    return sfa_concat(lit, star)


def _suffix_sfa(suf: str, algebra: CharAlgebra) -> SFA:
    """SFA accepting strings ending with suf."""
    if not suf:
        return _sigma_star(algebra)
    lit = sfa_from_string(suf, algebra)
    star = _sigma_star(algebra)
    return sfa_concat(star, lit)


def _char_at_sfa(index: int, c: str, algebra: CharAlgebra) -> SFA:
    """SFA accepting strings where position index has character c.
    Sigma^index . c . Sigma*"""
    if index < 0:
        return sfa_empty(algebra)
    prefix_sfa = _length_exactly(index, algebra)
    char_sfa = sfa_from_string(c, algebra)
    star = _sigma_star(algebra)
    return sfa_concat(sfa_concat(prefix_sfa, char_sfa), star)


def _in_set_sfa(strings: tuple, algebra: CharAlgebra) -> SFA:
    """SFA accepting exactly the given set of strings."""
    if not strings:
        return sfa_empty(algebra)
    result = sfa_from_string(strings[0], algebra)
    for s in strings[1:]:
        result = sfa_union(result, sfa_from_string(s, algebra))
    return result


# ---------------------------------------------------------------------------
# String Constraint Solver
# ---------------------------------------------------------------------------

class SolveResult(Enum):
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass
class StringSolution:
    """Result of solving string constraints."""
    result: SolveResult
    assignment: Optional[Dict[str, str]] = None  # var -> concrete string
    unsat_core: Optional[List[int]] = None        # indices of conflicting constraints
    stats: Dict[str, Any] = field(default_factory=dict)


class StringConstraintSolver:
    """Solver for string constraints using SFA + SMT.

    Each string variable maintains an SFA (automaton) representing the set of
    possible values. Constraints narrow the SFA via intersection. Length
    constraints are tracked both as SFA constraints and as SMT integer constraints.
    Word equations (concatenation) use product construction.

    Solving procedure:
    1. For each constraint, build an SFA restriction for the relevant variable(s)
    2. Intersect each variable's SFA with its accumulated constraints
    3. Check emptiness -- if any variable's SFA is empty, UNSAT
    4. For word equations, verify length consistency via SMT
    5. Extract satisfying assignment from non-empty SFAs
    """

    def __init__(self, alphabet: str = None):
        """Initialize solver.

        Args:
            alphabet: Characters in the alphabet. Default: printable ASCII.
        """
        self.algebra = CharAlgebra(alphabet)
        self.constraints: List[StringConstraint] = []
        self.variables: Set[str] = set()
        # Per-variable SFA (starts as Sigma*)
        self._var_sfa: Dict[str, SFA] = {}
        # Length constraints for SMT
        self._length_constraints: List[Tuple[str, str, int]] = []  # (var, op, value)
        # Concat constraints (processed separately)
        self._concat_constraints: List[StringConstraint] = []
        # Equality/inequality between variables
        self._var_eq_constraints: List[Tuple[str, str, bool]] = []  # (v1, v2, equal?)

    def _ensure_var(self, name: str):
        """Ensure variable exists with Sigma* SFA."""
        if name not in self.variables:
            self.variables.add(name)
            self._var_sfa[name] = _sigma_star(self.algebra)

    def add(self, constraint: StringConstraint):
        """Add a constraint to the solver."""
        self.constraints.append(constraint)
        self._ensure_var(constraint.var)
        if constraint.var2:
            self._ensure_var(constraint.var2)
        if constraint.var3:
            self._ensure_var(constraint.var3)

    def add_all(self, constraints: List[StringConstraint]):
        """Add multiple constraints."""
        for c in constraints:
            self.add(c)

    def _apply_sfa_constraint(self, var: str, sfa: SFA):
        """Intersect variable's SFA with the given constraint SFA."""
        self._var_sfa[var] = sfa_intersection(self._var_sfa[var], sfa)

    def _process_constraint(self, c: StringConstraint) -> bool:
        """Process a single constraint. Returns False if immediately UNSAT."""
        kind = c.kind

        if kind == ConstraintKind.REGEX:
            regex_sfa = compile_regex(c.pattern, self.algebra)
            self._apply_sfa_constraint(c.var, regex_sfa)

        elif kind == ConstraintKind.EQUALS_CONST:
            eq_sfa = sfa_from_string(c.pattern, self.algebra)
            self._apply_sfa_constraint(c.var, eq_sfa)

        elif kind == ConstraintKind.NOT_EQUALS_CONST:
            eq_sfa = sfa_from_string(c.pattern, self.algebra)
            neq_sfa = sfa_complement(eq_sfa.determinize())
            self._apply_sfa_constraint(c.var, neq_sfa)

        elif kind == ConstraintKind.EQUALS_VAR:
            self._var_eq_constraints.append((c.var, c.var2, True))

        elif kind == ConstraintKind.NOT_EQUALS_VAR:
            self._var_eq_constraints.append((c.var, c.var2, False))

        elif kind == ConstraintKind.CONCAT:
            self._concat_constraints.append(c)

        elif kind == ConstraintKind.LENGTH_EQ:
            len_sfa = _length_exactly(c.value, self.algebra)
            self._apply_sfa_constraint(c.var, len_sfa)
            self._length_constraints.append((c.var, '==', c.value))

        elif kind == ConstraintKind.LENGTH_LE:
            len_sfa = _length_at_most(c.value, self.algebra)
            self._apply_sfa_constraint(c.var, len_sfa)
            self._length_constraints.append((c.var, '<=', c.value))

        elif kind == ConstraintKind.LENGTH_GE:
            len_sfa = _length_at_least(c.value, self.algebra)
            self._apply_sfa_constraint(c.var, len_sfa)
            self._length_constraints.append((c.var, '>=', c.value))

        elif kind == ConstraintKind.LENGTH_RANGE:
            len_sfa = _length_between(c.value, c.value2, self.algebra)
            self._apply_sfa_constraint(c.var, len_sfa)
            self._length_constraints.append((c.var, '>=', c.value))
            self._length_constraints.append((c.var, '<=', c.value2))

        elif kind == ConstraintKind.CONTAINS:
            con_sfa = _contains_sfa(c.pattern, self.algebra)
            self._apply_sfa_constraint(c.var, con_sfa)

        elif kind == ConstraintKind.PREFIX:
            pre_sfa = _prefix_sfa(c.pattern, self.algebra)
            self._apply_sfa_constraint(c.var, pre_sfa)

        elif kind == ConstraintKind.SUFFIX:
            suf_sfa = _suffix_sfa(c.pattern, self.algebra)
            self._apply_sfa_constraint(c.var, suf_sfa)

        elif kind == ConstraintKind.CHAR_AT:
            ca_sfa = _char_at_sfa(c.value, c.char, self.algebra)
            self._apply_sfa_constraint(c.var, ca_sfa)

        elif kind == ConstraintKind.IN_SET:
            set_sfa = _in_set_sfa(c.strings, self.algebra)
            self._apply_sfa_constraint(c.var, set_sfa)

        elif kind == ConstraintKind.NOT_EMPTY:
            ne_sfa = _length_at_least(1, self.algebra)
            self._apply_sfa_constraint(c.var, ne_sfa)

        # Quick emptiness check
        if self._var_sfa[c.var].is_empty():
            return False
        return True

    def _solve_concat(self, c: StringConstraint, assignment: Dict[str, str]) -> bool:
        """Solve a concat constraint x . y = z.

        Strategy:
        1. If x and y both assigned: check x+y in z's SFA
        2. If x assigned and not y (or vice versa): use known part
        3. If z assigned/known: try all splits
        4. General: build concat SFA(x).SFA(y) and intersect with SFA(z)
        """
        x_var, y_var, z_var = c.var, c.var2, c.var3
        x_sfa = self._var_sfa[x_var]
        y_sfa = self._var_sfa[y_var]
        z_sfa = self._var_sfa[z_var]

        x_val = assignment.get(x_var)
        y_val = assignment.get(y_var)
        z_val = assignment.get(z_var)

        # Case 1: x and y both assigned
        if x_val is not None and y_val is not None:
            result = x_val + y_val
            if z_sfa.accepts(result):
                if z_val is None or z_val == result:
                    assignment[z_var] = result
                    return True
            return False

        # Case 2: x assigned, y not -- find y such that x+y in z's SFA
        if x_val is not None and y_val is None:
            # Build suffix SFA: what y can be such that x_val+y is accepted by z
            x_lit = sfa_from_string(x_val, self.algebra)
            concat_sfa = sfa_concat(x_lit, y_sfa)
            combined = sfa_intersection(concat_sfa, z_sfa)
            w = shortest_accepted(combined)
            if w is None:
                return False
            full = ''.join(w) if w else ''
            y_part = full[len(x_val):]
            if z_val is None or z_val == full:
                assignment[y_var] = y_part
                assignment[z_var] = full
                return True
            return False

        # Case 3: y assigned, x not
        if y_val is not None and x_val is None:
            y_lit = sfa_from_string(y_val, self.algebra)
            concat_sfa = sfa_concat(x_sfa, y_lit)
            combined = sfa_intersection(concat_sfa, z_sfa)
            w = shortest_accepted(combined)
            if w is None:
                return False
            full = ''.join(w) if w else ''
            x_part = full[:len(full) - len(y_val)]
            if z_val is None or z_val == full:
                assignment[x_var] = x_part
                assignment[z_var] = full
                return True
            return False

        # Case 4: z assigned -- try all splits
        if z_val is not None:
            for i in range(len(z_val) + 1):
                x_part = z_val[:i]
                y_part = z_val[i:]
                if x_sfa.accepts(x_part) and y_sfa.accepts(y_part):
                    assignment[x_var] = x_part
                    assignment[y_var] = y_part
                    return True
            return False

        # Case 5: nothing assigned -- build concat and intersect
        concat_sfa = sfa_concat(x_sfa, y_sfa)
        combined = sfa_intersection(concat_sfa, z_sfa)
        z_word = shortest_accepted(combined)
        if z_word is None:
            return False
        z_str = ''.join(z_word) if z_word else ''
        # Split z_str to find valid x, y parts
        for i in range(len(z_str) + 1):
            x_part = z_str[:i]
            y_part = z_str[i:]
            if x_sfa.accepts(x_part) and y_sfa.accepts(y_part):
                assignment[x_var] = x_part
                assignment[y_var] = y_part
                assignment[z_var] = z_str
                return True
        return False

    def _check_var_equalities(self, assignment: Dict[str, str]) -> bool:
        """Check variable equality/inequality constraints against assignment."""
        for v1, v2, equal in self._var_eq_constraints:
            if v1 in assignment and v2 in assignment:
                if equal and assignment[v1] != assignment[v2]:
                    return False
                if not equal and assignment[v1] == assignment[v2]:
                    return False
        return True

    def _extract_assignment(self) -> Optional[Dict[str, str]]:
        """Extract a satisfying assignment from the SFAs."""
        assignment = {}

        # Identify variables involved in concat constraints
        concat_vars = set()
        for c in self._concat_constraints:
            concat_vars.add(c.var)
            concat_vars.add(c.var2)
            concat_vars.add(c.var3)

        # Handle equality constraints first: propagate SFA intersections
        for v1, v2, equal in self._var_eq_constraints:
            if equal:
                common_sfa = sfa_intersection(self._var_sfa[v1], self._var_sfa[v2])
                if common_sfa.is_empty():
                    return None
                self._var_sfa[v1] = common_sfa
                self._var_sfa[v2] = common_sfa

        # Handle concat constraints before extracting words
        for c in self._concat_constraints:
            if not self._solve_concat(c, assignment):
                return None

        # Extract shortest word for remaining unassigned variables
        for var in self.variables:
            if var in assignment:
                continue
            sfa = self._var_sfa[var]
            word = shortest_accepted(sfa)
            if word is None:
                return None
            assignment[var] = ''.join(word) if word else ''

        # Handle equality for assigned vars
        for v1, v2, equal in self._var_eq_constraints:
            if equal:
                # Both should have same value from shared SFA
                if v1 in assignment and v2 not in assignment:
                    assignment[v2] = assignment[v1]
                elif v2 in assignment and v1 not in assignment:
                    assignment[v1] = assignment[v2]
                elif v1 in assignment and v2 in assignment:
                    if assignment[v1] != assignment[v2]:
                        # Try to reconcile
                        common_sfa = sfa_intersection(self._var_sfa[v1], self._var_sfa[v2])
                        word = shortest_accepted(common_sfa)
                        if word is None:
                            return None
                        val = ''.join(word) if word else ''
                        assignment[v1] = val
                        assignment[v2] = val
            else:
                # v1 != v2
                if v1 in assignment and v2 in assignment:
                    if assignment[v1] == assignment[v2]:
                        # Try alternate for v2
                        exclude = sfa_from_string(assignment[v1], self.algebra)
                        diff_sfa = sfa_difference(
                            self._var_sfa[v2].determinize(), exclude.determinize()
                        )
                        word = shortest_accepted(diff_sfa)
                        if word is not None:
                            assignment[v2] = ''.join(word) if word else ''
                        else:
                            # Try alternate for v1
                            exclude = sfa_from_string(assignment[v2], self.algebra)
                            diff_sfa = sfa_difference(
                                self._var_sfa[v1].determinize(), exclude.determinize()
                            )
                            word = shortest_accepted(diff_sfa)
                            if word is not None:
                                assignment[v1] = ''.join(word) if word else ''
                            else:
                                return None

        if not self._check_var_equalities(assignment):
            return None

        return assignment

    def check(self) -> StringSolution:
        """Solve all constraints.

        Returns:
            StringSolution with result, assignment (if SAT), and stats.
        """
        stats = {'constraints': len(self.constraints), 'variables': len(self.variables)}

        # Reset SFAs
        for var in self.variables:
            self._var_sfa[var] = _sigma_star(self.algebra)
        self._length_constraints = []
        self._concat_constraints = []
        self._var_eq_constraints = []

        # Process each constraint
        unsat_idx = None
        for i, c in enumerate(self.constraints):
            if not self._process_constraint(c):
                stats['unsat_at'] = i
                return StringSolution(
                    result=SolveResult.UNSAT,
                    unsat_core=[i],
                    stats=stats,
                )

        # Check each variable's SFA for emptiness
        for var in self.variables:
            if self._var_sfa[var].is_empty():
                stats['empty_var'] = var
                return StringSolution(result=SolveResult.UNSAT, stats=stats)

        # Extract assignment
        assignment = self._extract_assignment()
        if assignment is None:
            return StringSolution(result=SolveResult.UNSAT, stats=stats)

        # Verify length constraints via SMT
        if self._length_constraints:
            smt = SMTSolver()
            len_vars = {}
            for var in self.variables:
                lv = smt.Int(f'len_{var}')
                len_vars[var] = lv
                smt.add(lv >= IntConst(0))
                # Set actual length from assignment
                if var in assignment:
                    smt.add(lv == IntConst(len(assignment[var])))

            for var, op, val in self._length_constraints:
                lv = len_vars[var]
                if op == '==':
                    smt.add(lv == IntConst(val))
                elif op == '<=':
                    smt.add(lv <= IntConst(val))
                elif op == '>=':
                    smt.add(lv >= IntConst(val))

            if smt.check() != SMTResult.SAT:
                return StringSolution(result=SolveResult.UNSAT, stats=stats)

        stats['sfa_states'] = {var: self._var_sfa[var].count_states() for var in self.variables}
        return StringSolution(
            result=SolveResult.SAT,
            assignment=assignment,
            stats=stats,
        )

    def get_var_sfa(self, var: str) -> Optional[SFA]:
        """Get the current SFA for a variable (after check())."""
        return self._var_sfa.get(var)

    def get_var_language_size(self, var: str, max_length: int = 5) -> Optional[int]:
        """Estimate language size for a variable up to max_length."""
        from symbolic_automata import count_accepting_paths
        sfa = self._var_sfa.get(var)
        if sfa is None:
            return None
        return count_accepting_paths(sfa, max_length)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def solve_constraints(constraints: List[StringConstraint],
                      alphabet: str = None) -> StringSolution:
    """Solve a list of string constraints.

    Args:
        constraints: List of StringConstraint objects.
        alphabet: Optional alphabet string.

    Returns:
        StringSolution with result and assignment.
    """
    solver = StringConstraintSolver(alphabet)
    solver.add_all(constraints)
    return solver.check()


def check_regex_membership(var: str, pattern: str,
                           extra_constraints: List[StringConstraint] = None,
                           alphabet: str = None) -> StringSolution:
    """Check if there exists a string matching regex and all extra constraints."""
    constraints = [regex_constraint(var, pattern)]
    if extra_constraints:
        constraints.extend(extra_constraints)
    return solve_constraints(constraints, alphabet)


def check_word_equation(x: str, y: str, z: str,
                        x_constraints: List[StringConstraint] = None,
                        y_constraints: List[StringConstraint] = None,
                        z_constraints: List[StringConstraint] = None,
                        alphabet: str = None) -> StringSolution:
    """Check if word equation x . y = z is satisfiable under constraints."""
    constraints = [concat_eq(x, y, z)]
    for extra in [x_constraints, y_constraints, z_constraints]:
        if extra:
            constraints.extend(extra)
    return solve_constraints(constraints, alphabet)


def find_string_matching(pattern: str, length: int = None,
                         containing: str = None,
                         starting_with: str = None,
                         ending_with: str = None,
                         alphabet: str = None) -> Optional[str]:
    """Find a string matching the given regex pattern with optional constraints.

    Returns the string if found, None otherwise.
    """
    constraints = [regex_constraint('x', pattern)]
    if length is not None:
        constraints.append(length_eq('x', length))
    if containing is not None:
        constraints.append(contains('x', containing))
    if starting_with is not None:
        constraints.append(prefix('x', starting_with))
    if ending_with is not None:
        constraints.append(suffix('x', ending_with))
    result = solve_constraints(constraints, alphabet)
    if result.result == SolveResult.SAT:
        return result.assignment.get('x')
    return None


def check_string_disjointness(pattern1: str, pattern2: str,
                               alphabet: str = None) -> StringSolution:
    """Check if two regex languages are disjoint (no common string)."""
    constraints = [
        regex_constraint('x', pattern1),
        regex_constraint('x', pattern2),
    ]
    return solve_constraints(constraints, alphabet)


def enumerate_solutions(constraints: List[StringConstraint],
                        var: str, max_count: int = 10,
                        alphabet: str = None) -> List[str]:
    """Enumerate up to max_count distinct solutions for a variable.

    Uses iterative exclusion: find solution, exclude it, repeat.
    """
    all_constraints = list(constraints)
    solutions = []
    excluded = []

    for _ in range(max_count):
        current = list(all_constraints)
        for ex in excluded:
            current.append(not_equals_const(var, ex))
        result = solve_constraints(current, alphabet)
        if result.result != SolveResult.SAT:
            break
        val = result.assignment.get(var, '')
        solutions.append(val)
        excluded.append(val)

    return solutions


def check_implication(premise_constraints: List[StringConstraint],
                      conclusion_constraint: StringConstraint,
                      alphabet: str = None) -> bool:
    """Check if premise constraints imply the conclusion constraint.

    Returns True if every solution of premises also satisfies conclusion.
    Uses: premise AND NOT(conclusion) is UNSAT => implication holds.
    """
    var = conclusion_constraint.var
    kind = conclusion_constraint.kind
    alg = CharAlgebra(alphabet)

    # Build negation of conclusion as an SFA on var
    neg_sfa = None
    if kind == ConstraintKind.REGEX:
        regex_sfa = compile_regex(conclusion_constraint.pattern, alg).determinize()
        neg_sfa = sfa_complement(regex_sfa)
    elif kind == ConstraintKind.EQUALS_CONST:
        eq_sfa = sfa_from_string(conclusion_constraint.pattern, alg).determinize()
        neg_sfa = sfa_complement(eq_sfa)
    elif kind == ConstraintKind.LENGTH_EQ:
        eq_sfa = _length_exactly(conclusion_constraint.value, alg).determinize()
        neg_sfa = sfa_complement(eq_sfa)
    elif kind == ConstraintKind.LENGTH_GE:
        # NOT(len >= n) = len <= n-1
        neg_sfa = _length_at_most(conclusion_constraint.value - 1, alg)
    elif kind == ConstraintKind.LENGTH_LE:
        # NOT(len <= n) = len >= n+1
        neg_sfa = _length_at_least(conclusion_constraint.value + 1, alg)
    else:
        return False  # can't negate this constraint type

    # Solve: premises AND neg(conclusion)
    solver = StringConstraintSolver(alphabet)
    solver.add_all(premise_constraints)
    solver._ensure_var(var)
    # Add neg_sfa as additional constraint on the variable
    # We do this by adding it to the constraints list as processed after check resets
    # Instead, just solve directly: process premises, then intersect neg_sfa
    result = solver.check()
    if result.result == SolveResult.UNSAT:
        # Premises alone are UNSAT -> implication holds vacuously
        return True

    # Now check premises AND NOT(conclusion)
    # Re-solve with the negation added
    solver2 = StringConstraintSolver(alphabet)
    solver2.add_all(premise_constraints)
    # We need to add the negation as a constraint. Since we can't express
    # arbitrary SFA as a constraint, we process it during check.
    # Approach: override the var's SFA after processing all other constraints.
    # Save the neg_sfa and apply it in check.
    # Simplest: just use the solver internals directly.
    solver2._ensure_var(var)
    # Process all regular constraints
    for c in solver2.constraints:
        solver2._process_constraint(c)
    # Apply negation SFA
    solver2._apply_sfa_constraint(var, neg_sfa)
    # Check emptiness
    if solver2._var_sfa[var].is_empty():
        return True
    # Try to extract assignment
    assignment = solver2._extract_assignment()
    return assignment is None


def string_solver_stats(solver: StringConstraintSolver) -> dict:
    """Get statistics about the solver state."""
    stats = {
        'variables': len(solver.variables),
        'constraints': len(solver.constraints),
        'var_names': sorted(solver.variables),
    }
    for var in solver.variables:
        sfa = solver._var_sfa.get(var)
        if sfa:
            stats[f'{var}_states'] = sfa.count_states()
            stats[f'{var}_transitions'] = sfa.count_transitions()
    return stats
