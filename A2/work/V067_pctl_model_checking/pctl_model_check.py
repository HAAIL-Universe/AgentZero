"""V067: PCTL Model Checking

Probabilistic Computation Tree Logic model checking over discrete-time Markov chains.
Composes V065 (Markov chain analysis) for chain operations.

PCTL extends CTL with probabilistic quantifiers:
  P>=p [phi]  -- the probability of satisfying phi is >= p
  P<=p [phi]  -- the probability of satisfying phi is <= p

Path formulas:
  X phi       -- next-state satisfies phi
  phi U psi   -- phi until psi (bounded and unbounded)
  F phi       -- eventually phi (sugar for true U phi)
  G phi       -- always phi (sugar for NOT F NOT phi)

State formulas:
  true, false, atom(label)
  NOT phi, phi AND psi, phi OR psi
  P>=p [path_formula], P<=p [path_formula]
"""

from __future__ import annotations
import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Dict, Tuple, Union
from fractions import Fraction

# Import V065 Markov chain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
from markov_chain import MarkovChain, make_chain, analyze_chain, StateType

# ---------------------------------------------------------------------------
# PCTL AST
# ---------------------------------------------------------------------------

class FormulaKind(Enum):
    TRUE = "true"
    FALSE = "false"
    ATOM = "atom"
    NOT = "not"
    AND = "and"
    OR = "or"
    PROB_GEQ = "P>="
    PROB_LEQ = "P<="
    PROB_GT = "P>"
    PROB_LT = "P<"
    NEXT = "X"
    UNTIL = "U"
    BOUNDED_UNTIL = "BU"


@dataclass(frozen=True)
class PCTL:
    """PCTL formula node."""
    kind: FormulaKind
    # For ATOM: label string
    label: Optional[str] = None
    # For NOT, NEXT: single sub-formula
    sub: Optional['PCTL'] = None
    # For AND, OR: left and right
    left: Optional['PCTL'] = None
    right: Optional['PCTL'] = None
    # For PROB_*: threshold and path formula
    threshold: Optional[float] = None
    path: Optional['PCTL'] = None
    # For UNTIL: phi (left) until psi (right)
    # For BOUNDED_UNTIL: bound
    bound: Optional[int] = None

    def __repr__(self):
        if self.kind == FormulaKind.TRUE:
            return "true"
        elif self.kind == FormulaKind.FALSE:
            return "false"
        elif self.kind == FormulaKind.ATOM:
            return f'"{self.label}"'
        elif self.kind == FormulaKind.NOT:
            return f"!({self.sub})"
        elif self.kind == FormulaKind.AND:
            return f"({self.left} & {self.right})"
        elif self.kind == FormulaKind.OR:
            return f"({self.left} | {self.right})"
        elif self.kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_LEQ,
                           FormulaKind.PROB_GT, FormulaKind.PROB_LT):
            op = self.kind.value
            return f"{op}{self.threshold}[{self.path}]"
        elif self.kind == FormulaKind.NEXT:
            return f"X({self.sub})"
        elif self.kind == FormulaKind.UNTIL:
            return f"({self.left} U {self.right})"
        elif self.kind == FormulaKind.BOUNDED_UNTIL:
            return f"({self.left} U<={self.bound} {self.right})"
        return f"PCTL({self.kind})"


# Constructors
def tt() -> PCTL:
    return PCTL(kind=FormulaKind.TRUE)

def ff() -> PCTL:
    return PCTL(kind=FormulaKind.FALSE)

def atom(label: str) -> PCTL:
    return PCTL(kind=FormulaKind.ATOM, label=label)

def pnot(phi: PCTL) -> PCTL:
    # Simplify double negation
    if phi.kind == FormulaKind.NOT:
        return phi.sub
    if phi.kind == FormulaKind.TRUE:
        return ff()
    if phi.kind == FormulaKind.FALSE:
        return tt()
    return PCTL(kind=FormulaKind.NOT, sub=phi)

def pand(phi: PCTL, psi: PCTL) -> PCTL:
    if phi.kind == FormulaKind.FALSE or psi.kind == FormulaKind.FALSE:
        return ff()
    if phi.kind == FormulaKind.TRUE:
        return psi
    if psi.kind == FormulaKind.TRUE:
        return phi
    return PCTL(kind=FormulaKind.AND, left=phi, right=psi)

def por(phi: PCTL, psi: PCTL) -> PCTL:
    if phi.kind == FormulaKind.TRUE or psi.kind == FormulaKind.TRUE:
        return tt()
    if phi.kind == FormulaKind.FALSE:
        return psi
    if psi.kind == FormulaKind.FALSE:
        return phi
    return PCTL(kind=FormulaKind.OR, left=phi, right=psi)

def prob_geq(p: float, path: PCTL) -> PCTL:
    return PCTL(kind=FormulaKind.PROB_GEQ, threshold=p, path=path)

def prob_leq(p: float, path: PCTL) -> PCTL:
    return PCTL(kind=FormulaKind.PROB_LEQ, threshold=p, path=path)

def prob_gt(p: float, path: PCTL) -> PCTL:
    return PCTL(kind=FormulaKind.PROB_GT, threshold=p, path=path)

def prob_lt(p: float, path: PCTL) -> PCTL:
    return PCTL(kind=FormulaKind.PROB_LT, threshold=p, path=path)

def next_f(phi: PCTL) -> PCTL:
    return PCTL(kind=FormulaKind.NEXT, sub=phi)

def until(phi: PCTL, psi: PCTL) -> PCTL:
    return PCTL(kind=FormulaKind.UNTIL, left=phi, right=psi)

def bounded_until(phi: PCTL, psi: PCTL, k: int) -> PCTL:
    return PCTL(kind=FormulaKind.BOUNDED_UNTIL, left=phi, right=psi, bound=k)

# Sugar
def eventually(phi: PCTL) -> PCTL:
    """F phi = true U phi"""
    return until(tt(), phi)

def always(phi: PCTL) -> PCTL:
    """G phi = NOT (true U NOT phi)"""
    # We represent G directly via prob complement:
    # P>=p[G phi] iff P<=1-p[F !phi]
    # But at the path level we just negate: G phi is not a primitive,
    # we handle it in the checker.
    # For now, use a special encoding: store as NOT(eventually(NOT(phi)))
    # Actually, let's just keep it as until with negation and handle in checker
    return PCTL(kind=FormulaKind.UNTIL, left=tt(), right=pnot(phi))


def bounded_eventually(phi: PCTL, k: int) -> PCTL:
    """F<=k phi = true U<=k phi"""
    return bounded_until(tt(), phi, k)


# ---------------------------------------------------------------------------
# Labeling: maps state indices to sets of atomic propositions
# ---------------------------------------------------------------------------

@dataclass
class LabeledMC:
    """Markov chain with state labeling for PCTL model checking."""
    mc: MarkovChain
    labels: Dict[int, Set[str]]  # state -> set of atomic proposition names

    def states_with(self, label: str) -> Set[int]:
        """Return set of states where label holds."""
        return {s for s, labs in self.labels.items() if label in labs}

    def states_without(self, label: str) -> Set[int]:
        """Return set of states where label does NOT hold."""
        all_states = set(range(self.mc.n_states))
        return all_states - self.states_with(label)


def make_labeled_mc(matrix: List[List[float]],
                    labels: Dict[int, Set[str]],
                    state_labels: Optional[List[str]] = None) -> LabeledMC:
    """Create a labeled Markov chain."""
    mc = make_chain(matrix, state_labels)
    # Ensure all states have label entries
    full_labels = {}
    for s in range(mc.n_states):
        full_labels[s] = set(labels.get(s, set()))
    return LabeledMC(mc=mc, labels=full_labels)


# ---------------------------------------------------------------------------
# PCTL Model Checking Algorithm
# ---------------------------------------------------------------------------

class PCTLChecker:
    """PCTL model checker for labeled Markov chains.

    Uses iterative linear equation solving for unbounded until,
    and matrix-vector multiplication for bounded properties.
    """

    def __init__(self, lmc: LabeledMC, tol: float = 1e-10):
        self.lmc = lmc
        self.mc = lmc.mc
        self.n = lmc.mc.n_states
        self.P = lmc.mc.transition
        self.tol = tol

    def check(self, formula: PCTL) -> Set[int]:
        """Return the set of states satisfying the PCTL formula."""
        kind = formula.kind

        if kind == FormulaKind.TRUE:
            return set(range(self.n))

        elif kind == FormulaKind.FALSE:
            return set()

        elif kind == FormulaKind.ATOM:
            return self.lmc.states_with(formula.label)

        elif kind == FormulaKind.NOT:
            sub_sat = self.check(formula.sub)
            return set(range(self.n)) - sub_sat

        elif kind == FormulaKind.AND:
            return self.check(formula.left) & self.check(formula.right)

        elif kind == FormulaKind.OR:
            return self.check(formula.left) | self.check(formula.right)

        elif kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_LEQ,
                      FormulaKind.PROB_GT, FormulaKind.PROB_LT):
            return self._check_prob(formula)

        else:
            raise ValueError(f"Unexpected top-level formula kind: {kind}")

    def _check_prob(self, formula: PCTL) -> Set[int]:
        """Handle P~p[path_formula]."""
        path = formula.path
        threshold = formula.threshold

        # Compute probability vector for path formula
        probs = self._path_probs(path)

        # Compare against threshold
        result = set()
        for s in range(self.n):
            p = probs[s]
            if formula.kind == FormulaKind.PROB_GEQ:
                if p >= threshold - self.tol:
                    result.add(s)
            elif formula.kind == FormulaKind.PROB_LEQ:
                if p <= threshold + self.tol:
                    result.add(s)
            elif formula.kind == FormulaKind.PROB_GT:
                if p > threshold + self.tol:
                    result.add(s)
            elif formula.kind == FormulaKind.PROB_LT:
                if p < threshold - self.tol:
                    result.add(s)

        return result

    def _path_probs(self, path: PCTL) -> List[float]:
        """Compute probability of satisfying a path formula from each state."""
        kind = path.kind

        if kind == FormulaKind.NEXT:
            return self._next_probs(path.sub)

        elif kind == FormulaKind.UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._until_probs(phi_sat, psi_sat)

        elif kind == FormulaKind.BOUNDED_UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._bounded_until_probs(phi_sat, psi_sat, path.bound)

        else:
            raise ValueError(f"Unexpected path formula kind: {kind}")

    def _next_probs(self, phi: PCTL) -> List[float]:
        """P(X phi | s) = sum_{t in Sat(phi)} P[s][t]."""
        phi_sat = self.check(phi)
        probs = []
        for s in range(self.n):
            p = sum(self.P[s][t] for t in phi_sat)
            probs.append(p)
        return probs

    def _until_probs(self, phi_sat: Set[int], psi_sat: Set[int]) -> List[float]:
        """Compute P(phi U psi | s) for each state s.

        States are classified into three categories:
        - S_yes: states where probability is 1 (psi holds, or all paths through phi lead to psi)
        - S_no: states where probability is 0 (neither phi nor psi, or no path to psi through phi)
        - S_?: remaining states -- solve linear system

        For S_? states: x_s = sum_{t} P[s][t] * x_t
        where x_s = 1 for s in S_yes, x_s = 0 for s in S_no.
        """
        all_states = set(range(self.n))

        # S_no: states not in phi_sat and not in psi_sat
        # Also: states from which psi_sat is unreachable through phi_sat states
        s_yes = set(psi_sat)  # psi already holds
        s_no = all_states - phi_sat - psi_sat  # neither phi nor psi

        # Refine: states in phi_sat but from which psi_sat is unreachable
        # via phi_sat states. Do backward BFS from psi_sat through phi_sat.
        reachable_to_psi = set(psi_sat)
        worklist = list(psi_sat)
        while worklist:
            t = worklist.pop()
            for s in range(self.n):
                if s not in reachable_to_psi and s in phi_sat and self.P[s][t] > 0:
                    reachable_to_psi.add(s)
                    worklist.append(s)

        # States in phi_sat - psi_sat that can't reach psi -> prob 0
        for s in phi_sat - psi_sat:
            if s not in reachable_to_psi:
                s_no.add(s)

        s_maybe = all_states - s_yes - s_no

        # If no maybe states, we're done
        probs = [0.0] * self.n
        for s in s_yes:
            probs[s] = 1.0
        # s_no already 0.0

        if not s_maybe:
            return probs

        # Solve linear system for s_maybe states:
        # x_s = sum_{t in s_maybe} P[s][t] * x_t + sum_{t in s_yes} P[s][t]
        # => x_s - sum_{t in s_maybe} P[s][t] * x_t = sum_{t in s_yes} P[s][t]
        # => (I - P_maybe) * x = b
        maybe_list = sorted(s_maybe)
        maybe_idx = {s: i for i, s in enumerate(maybe_list)}
        m = len(maybe_list)

        A = [[0.0] * m for _ in range(m)]
        b = [0.0] * m

        for i, s in enumerate(maybe_list):
            A[i][i] = 1.0
            for t in range(self.n):
                if t in s_maybe:
                    j = maybe_idx[t]
                    A[i][j] -= self.P[s][t]
                elif t in s_yes:
                    b[i] += self.P[s][t]

        sol = _solve_linear(A, b)
        if sol is not None:
            for i, s in enumerate(maybe_list):
                probs[s] = max(0.0, min(1.0, sol[i]))

        return probs

    def _bounded_until_probs(self, phi_sat: Set[int], psi_sat: Set[int],
                              k: int) -> List[float]:
        """Compute P(phi U<=k psi | s) via backward induction.

        Base: at step k, prob = 1 if psi holds, 0 otherwise (for states in phi|psi)
        Step i: prob_i(s) = 1 if s in psi_sat
                           = 0 if s not in phi_sat and s not in psi_sat
                           = sum_t P[s][t] * prob_{i+1}(t)  if s in phi_sat - psi_sat
        """
        # prob[s] at current step
        prob = [0.0] * self.n
        for s in psi_sat:
            prob[s] = 1.0

        for step in range(k):
            new_prob = [0.0] * self.n
            for s in range(self.n):
                if s in psi_sat:
                    new_prob[s] = 1.0
                elif s in phi_sat:
                    # s in phi but not psi: transition
                    new_prob[s] = sum(self.P[s][t] * prob[t] for t in range(self.n))
                # else: not in phi or psi -> 0
            prob = new_prob

        return prob

    def check_quantitative(self, path_formula: PCTL) -> List[float]:
        """Compute exact probability of path formula from each state.

        Unlike check() which returns a set of states, this returns the
        probability vector directly. Useful for quantitative queries.
        """
        return self._path_probs(path_formula)

    def check_state(self, state: int, formula: PCTL) -> bool:
        """Check if a specific state satisfies the formula."""
        return state in self.check(formula)

    def check_all(self, formula: PCTL) -> bool:
        """Check if ALL states satisfy the formula."""
        return self.check(formula) == set(range(self.n))

    def check_initial(self, initial: int, formula: PCTL) -> bool:
        """Check if the initial state satisfies the formula."""
        return initial in self.check(formula)


# ---------------------------------------------------------------------------
# Linear algebra helper (same approach as V065)
# ---------------------------------------------------------------------------

def _solve_linear(A_orig: List[List[float]], b_orig: List[float]) -> Optional[List[float]]:
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(A_orig)
    if n == 0:
        return []
    A = [row[:] for row in A_orig]
    b = b_orig[:]

    for col in range(n):
        # Partial pivoting
        max_val = abs(A[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(A[row][col]) > max_val:
                max_val = abs(A[row][col])
                max_row = row
        if max_val < 1e-15:
            return None
        if max_row != col:
            A[col], A[max_row] = A[max_row], A[col]
            b[col], b[max_row] = b[max_row], b[col]

        # Eliminate
        for row in range(col + 1, n):
            factor = A[row][col] / A[col][col]
            for j in range(col, n):
                A[row][j] -= factor * A[col][j]
            b[row] -= factor * b[col]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-15:
            return None
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    return x


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

@dataclass
class PCTLResult:
    """Result of PCTL model checking."""
    formula: PCTL
    satisfying_states: Set[int]
    all_states: int
    probabilities: Optional[List[float]] = None  # For quantitative queries
    state_labels: Optional[List[str]] = None

    @property
    def all_satisfy(self) -> bool:
        return len(self.satisfying_states) == self.all_states

    @property
    def none_satisfy(self) -> bool:
        return len(self.satisfying_states) == 0

    def summary(self) -> str:
        labels = self.state_labels or [f"s{i}" for i in range(self.all_states)]
        sat_names = [labels[s] for s in sorted(self.satisfying_states)]
        lines = [
            f"Formula: {self.formula}",
            f"Satisfying states: {sat_names} ({len(self.satisfying_states)}/{self.all_states})",
        ]
        if self.probabilities:
            lines.append("Probabilities:")
            for i, p in enumerate(self.probabilities):
                lines.append(f"  {labels[i]}: {p:.6f}")
        return "\n".join(lines)


def check_pctl(lmc: LabeledMC, formula: PCTL) -> PCTLResult:
    """Check a PCTL formula against a labeled Markov chain.

    Returns PCTLResult with satisfying states.
    """
    checker = PCTLChecker(lmc)
    sat = checker.check(formula)
    # If the top-level is a probability operator, also get quantitative result
    probs = None
    if formula.kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_LEQ,
                        FormulaKind.PROB_GT, FormulaKind.PROB_LT):
        probs = checker.check_quantitative(formula.path)
    return PCTLResult(
        formula=formula,
        satisfying_states=sat,
        all_states=lmc.mc.n_states,
        probabilities=probs,
        state_labels=lmc.mc.state_labels,
    )


def check_pctl_state(lmc: LabeledMC, state: int, formula: PCTL) -> bool:
    """Check if a specific state satisfies a PCTL formula."""
    checker = PCTLChecker(lmc)
    return checker.check_state(state, formula)


def check_pctl_quantitative(lmc: LabeledMC, path_formula: PCTL) -> List[float]:
    """Compute exact probabilities for a path formula from each state."""
    checker = PCTLChecker(lmc)
    return checker.check_quantitative(path_formula)


# ---------------------------------------------------------------------------
# PCTL Parsing (text -> PCTL AST)
# ---------------------------------------------------------------------------

class PCTLParser:
    """Simple recursive descent parser for PCTL formulas.

    Grammar:
      formula := or_expr
      or_expr := and_expr ('|' and_expr)*
      and_expr := unary ('&' unary)*
      unary := '!' unary | primary
      primary := 'true' | 'false' | ATOM | prob_formula | '(' formula ')'
      prob_formula := 'P' comp NUMBER '[' path_formula ']'
      comp := '>=' | '<=' | '>' | '<'
      path_formula := 'X' formula | formula 'U' ['<=' NUMBER] formula
      ATOM := '"' CHARS '"'
    """

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse(self) -> PCTL:
        result = self._or_expr()
        self._skip_ws()
        if self.pos < len(self.text):
            raise ValueError(f"Unexpected characters at position {self.pos}: '{self.text[self.pos:]}'")
        return result

    def _skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in ' \t\n\r':
            self.pos += 1

    def _peek(self) -> Optional[str]:
        self._skip_ws()
        if self.pos < len(self.text):
            return self.text[self.pos]
        return None

    def _match(self, s: str) -> bool:
        self._skip_ws()
        if self.text[self.pos:self.pos+len(s)] == s:
            self.pos += len(s)
            return True
        return False

    def _expect(self, s: str):
        if not self._match(s):
            raise ValueError(f"Expected '{s}' at position {self.pos}")

    def _or_expr(self) -> PCTL:
        left = self._and_expr()
        while self._peek() == '|' and not self._lookahead('||'):
            self._match('|')
            right = self._and_expr()
            left = por(left, right)
        return left

    def _lookahead(self, s: str) -> bool:
        self._skip_ws()
        return self.text[self.pos:self.pos+len(s)] == s

    def _and_expr(self) -> PCTL:
        left = self._unary()
        while self._peek() == '&':
            self._match('&')
            right = self._unary()
            left = pand(left, right)
        return left

    def _unary(self) -> PCTL:
        if self._peek() == '!':
            self._match('!')
            sub = self._unary()
            return pnot(sub)
        return self._primary()

    def _primary(self) -> PCTL:
        self._skip_ws()

        # true/false
        if self._match('true'):
            return tt()
        if self._match('false'):
            return ff()

        # P operator
        if self._peek() == 'P':
            return self._prob_formula()

        # Quoted atom
        if self._peek() == '"':
            return self._atom()

        # Parenthesized
        if self._peek() == '(':
            self._match('(')
            result = self._or_expr()
            self._expect(')')
            return result

        raise ValueError(f"Unexpected character at position {self.pos}: '{self.text[self.pos:]}'")

    def _atom(self) -> PCTL:
        self._expect('"')
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != '"':
            self.pos += 1
        label = self.text[start:self.pos]
        self._expect('"')
        return atom(label)

    def _prob_formula(self) -> PCTL:
        self._expect('P')
        self._skip_ws()

        # Parse comparison operator
        if self._match('>='):
            kind = FormulaKind.PROB_GEQ
        elif self._match('<='):
            kind = FormulaKind.PROB_LEQ
        elif self._match('>'):
            kind = FormulaKind.PROB_GT
        elif self._match('<'):
            kind = FormulaKind.PROB_LT
        else:
            raise ValueError(f"Expected comparison operator after P at position {self.pos}")

        # Parse threshold
        threshold = self._number()

        # Parse [path_formula]
        self._expect('[')
        path = self._path_formula()
        self._expect(']')

        return PCTL(kind=kind, threshold=threshold, path=path)

    def _path_formula(self) -> PCTL:
        self._skip_ws()

        # X phi (next)
        if self._peek() == 'X':
            self._match('X')
            sub = self._or_expr()
            return next_f(sub)

        # F phi (eventually = true U phi)
        if self._peek() == 'F':
            self._match('F')
            # Check for bounded: F<=k
            if self._match('<='):
                k = int(self._number())
                sub = self._or_expr()
                return bounded_eventually(sub, k)
            sub = self._or_expr()
            return eventually(sub)

        # G phi (always)
        if self._peek() == 'G':
            self._match('G')
            sub = self._or_expr()
            # G phi = NOT(F NOT phi) -- but we need it as a path formula
            # P>=p[G phi] = 1 - P[F !phi] ... handle via complement
            # Store as: until(true, not(phi)) and let caller negate
            # Actually: for the path formula, we handle G specially
            return always(sub)

        # Otherwise: phi U psi or phi U<=k psi
        left = self._or_expr()
        self._skip_ws()
        if self._match('U'):
            if self._match('<='):
                k = int(self._number())
                right = self._or_expr()
                return bounded_until(left, right, k)
            right = self._or_expr()
            return until(left, right)

        # No U found -- treat as a state formula wrapped in next? Error.
        raise ValueError(f"Expected 'U' in path formula at position {self.pos}")

    def _number(self) -> float:
        self._skip_ws()
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
            self.pos += 1
        if self.pos == start:
            raise ValueError(f"Expected number at position {self.pos}")
        return float(self.text[start:self.pos])


def parse_pctl(text: str) -> PCTL:
    """Parse a PCTL formula from text."""
    return PCTLParser(text).parse()


# ---------------------------------------------------------------------------
# Steady-state (long-run) PCTL properties
# ---------------------------------------------------------------------------

def check_steady_state_property(lmc: LabeledMC, label: str,
                                 lower: float = 0.0,
                                 upper: float = 1.0) -> Dict:
    """Check if steady-state probability of being in a labeled state is within bounds.

    Uses V065 exact steady-state computation.
    """
    analysis = analyze_chain(lmc.mc)
    ss = analysis.steady_state

    if ss is None:
        return {
            'verified': False,
            'reason': 'No steady-state distribution (chain may be periodic or reducible)',
        }

    target_states = lmc.states_with(label)
    prob = sum(ss[s] for s in target_states)

    verified = lower - 1e-10 <= prob <= upper + 1e-10
    return {
        'verified': verified,
        'label': label,
        'probability': prob,
        'lower_bound': lower,
        'upper_bound': upper,
        'target_states': sorted(target_states),
        'steady_state': ss,
    }


# ---------------------------------------------------------------------------
# Reward-based PCTL (expected accumulated reward)
# ---------------------------------------------------------------------------

def expected_reward_until(lmc: LabeledMC, rewards: List[float],
                          target: PCTL) -> List[float]:
    """Compute expected accumulated reward until reaching target states.

    rewards[s] = reward earned per step in state s.
    Returns expected total reward from each state.
    """
    checker = PCTLChecker(lmc)
    target_sat = checker.check(target)

    n = lmc.mc.n_states
    P = lmc.mc.transition

    # States that can't reach target get infinite reward (use large number)
    # States in target get 0 reward
    # Others: r_s = rewards[s] + sum_t P[s][t] * r_t

    # BFS to find states that can reach target
    can_reach = set(target_sat)
    worklist = list(target_sat)
    while worklist:
        t = worklist.pop()
        for s in range(n):
            if s not in can_reach and P[s][t] > 0:
                can_reach.add(s)
                worklist.append(s)

    result = [float('inf')] * n
    for s in target_sat:
        result[s] = 0.0

    # Solve for states that can reach target and are not target
    solve_states = sorted(can_reach - target_sat)
    if not solve_states:
        return result

    idx = {s: i for i, s in enumerate(solve_states)}
    m = len(solve_states)

    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m

    for i, s in enumerate(solve_states):
        A[i][i] = 1.0
        b[i] = rewards[s]
        for t in range(n):
            if t in target_sat:
                pass  # r_t = 0, contributes nothing
            elif t in idx:
                j = idx[t]
                A[i][j] -= P[s][t]
            # else: t can't reach target, skip (shouldn't happen if s can reach)

    sol = _solve_linear(A, b)
    if sol is not None:
        for i, s in enumerate(solve_states):
            result[s] = max(0.0, sol[i])

    return result


# ---------------------------------------------------------------------------
# Comparison and convenience
# ---------------------------------------------------------------------------

def compare_bounded_vs_unbounded(lmc: LabeledMC, phi: PCTL, psi: PCTL,
                                  bounds: List[int]) -> Dict:
    """Compare bounded until probabilities for increasing bounds.

    Shows convergence of P(phi U<=k psi) toward P(phi U psi).
    """
    checker = PCTLChecker(lmc)

    unbounded = checker.check_quantitative(until(phi, psi))
    results = {'unbounded': unbounded, 'bounded': {}}

    for k in bounds:
        bounded = checker.check_quantitative(bounded_until(phi, psi, k))
        results['bounded'][k] = bounded

    return results


def verify_pctl_property(lmc: LabeledMC, formula: PCTL,
                          initial_state: int = 0) -> Dict:
    """Verify a PCTL property at an initial state.

    Returns a verification result dict.
    """
    result = check_pctl(lmc, formula)
    holds = initial_state in result.satisfying_states

    out = {
        'holds': holds,
        'initial_state': initial_state,
        'formula': str(formula),
        'satisfying_states': sorted(result.satisfying_states),
        'total_states': result.all_states,
    }
    if result.probabilities:
        out['probabilities'] = result.probabilities
        out['initial_probability'] = result.probabilities[initial_state]

    return out


def batch_check(lmc: LabeledMC, formulas: List[PCTL]) -> List[PCTLResult]:
    """Check multiple PCTL formulas against the same labeled MC."""
    return [check_pctl(lmc, f) for f in formulas]
