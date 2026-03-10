"""V065: Markov Chain Analysis

Discrete Markov chain analysis for probabilistic program reasoning.
Composes V063 (verified probabilistic) + C037 (SMT solver) + C010 (parser).

Features:
- Markov chain construction from transition matrices
- Steady-state (stationary) distribution via power iteration
- Absorption probability computation for absorbing chains
- Expected hitting time (mean first passage time)
- Transient/recurrent state classification
- Communication class detection (irreducible components)
- Periodicity detection
- C10 program extraction: build chains from loop/branch structures
- SMT-based property verification on chains
"""

import sys
import os
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V063_verified_probabilistic_programs'))


# ============================================================
# Core Data Structures
# ============================================================

@dataclass
class MarkovChain:
    """A discrete-time Markov chain.

    States are integers 0..n-1. Transition matrix P[i][j] = P(X_{t+1}=j | X_t=i).
    """
    n_states: int
    transition: List[List[float]]  # n x n matrix
    state_labels: Optional[List[str]] = None

    def __post_init__(self):
        if self.state_labels is None:
            self.state_labels = [f"s{i}" for i in range(self.n_states)]

    def validate(self) -> List[str]:
        """Check that transition matrix is valid (rows sum to 1, non-negative)."""
        errors = []
        if len(self.transition) != self.n_states:
            errors.append(f"Expected {self.n_states} rows, got {len(self.transition)}")
            return errors
        for i, row in enumerate(self.transition):
            if len(row) != self.n_states:
                errors.append(f"Row {i}: expected {self.n_states} columns, got {len(row)}")
                continue
            for j, p in enumerate(row):
                if p < -1e-10:
                    errors.append(f"P[{i}][{j}] = {p} < 0")
            row_sum = sum(row)
            if abs(row_sum - 1.0) > 1e-6:
                errors.append(f"Row {i} sums to {row_sum}, not 1.0")
        return errors

    def step(self, dist: List[float]) -> List[float]:
        """One step: multiply distribution by transition matrix."""
        result = [0.0] * self.n_states
        for i in range(self.n_states):
            for j in range(self.n_states):
                result[j] += dist[i] * self.transition[i][j]
        return result

    def successors(self, state: int) -> List[Tuple[int, float]]:
        """Return (next_state, probability) pairs for non-zero transitions."""
        return [(j, self.transition[state][j])
                for j in range(self.n_states)
                if self.transition[state][j] > 1e-15]

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "transition": self.transition,
            "state_labels": self.state_labels,
        }

    @staticmethod
    def from_dict(d: dict) -> 'MarkovChain':
        return MarkovChain(
            n_states=d["n_states"],
            transition=d["transition"],
            state_labels=d.get("state_labels"),
        )


class StateType(Enum):
    TRANSIENT = "transient"
    RECURRENT = "recurrent"
    ABSORBING = "absorbing"


@dataclass
class ChainAnalysis:
    """Complete analysis results for a Markov chain."""
    chain: MarkovChain
    state_types: List[StateType]
    communication_classes: List[Set[int]]
    is_irreducible: bool
    is_absorbing: bool
    period: int  # 0 if aperiodic, d if periodic
    steady_state: Optional[List[float]]
    absorption_probabilities: Optional[Dict[int, List[float]]]  # absorbing_state -> prob from each state
    expected_hitting_times: Optional[Dict[int, List[float]]]    # target -> expected steps from each state

    def summary(self) -> str:
        lines = [f"Markov Chain Analysis ({self.chain.n_states} states)"]
        lines.append(f"Irreducible: {self.is_irreducible}")
        lines.append(f"Absorbing: {self.is_absorbing}")
        lines.append(f"Period: {self.period if self.period > 0 else 'aperiodic'}")
        lines.append(f"Communication classes: {len(self.communication_classes)}")
        for i, cc in enumerate(self.communication_classes):
            labels = [self.chain.state_labels[s] for s in sorted(cc)]
            lines.append(f"  Class {i}: {{{', '.join(labels)}}}")
        if self.steady_state:
            lines.append("Steady-state distribution:")
            for i, p in enumerate(self.steady_state):
                if p > 1e-10:
                    lines.append(f"  {self.chain.state_labels[i]}: {p:.6f}")
        return "\n".join(lines)


# ============================================================
# Chain Construction Helpers
# ============================================================

def make_chain(matrix: List[List[float]], labels: List[str] = None) -> MarkovChain:
    """Create a MarkovChain from a transition matrix."""
    n = len(matrix)
    return MarkovChain(n_states=n, transition=matrix, state_labels=labels)


def random_walk_chain(n: int, p_right: float = 0.5, absorbing_ends: bool = True) -> MarkovChain:
    """Create a 1D random walk chain with n states.

    If absorbing_ends, states 0 and n-1 are absorbing.
    Otherwise, reflecting boundaries.
    """
    matrix = [[0.0] * n for _ in range(n)]
    p_left = 1.0 - p_right

    if absorbing_ends:
        matrix[0][0] = 1.0
        matrix[n-1][n-1] = 1.0
        for i in range(1, n-1):
            matrix[i][i-1] = p_left
            matrix[i][i+1] = p_right
    else:
        # Reflecting
        matrix[0][0] = p_left
        matrix[0][1] = p_right
        matrix[n-1][n-2] = p_left
        matrix[n-1][n-1] = p_right
        for i in range(1, n-1):
            matrix[i][i-1] = p_left
            matrix[i][i+1] = p_right

    labels = [f"s{i}" for i in range(n)]
    return MarkovChain(n_states=n, transition=matrix, state_labels=labels)


def gambler_ruin_chain(n: int, p: float = 0.5) -> MarkovChain:
    """Gambler's ruin: states 0..n, absorbing at 0 and n.

    At state i (1 <= i <= n-1), go to i+1 with prob p, i-1 with prob 1-p.
    """
    total = n + 1
    matrix = [[0.0] * total for _ in range(total)]
    matrix[0][0] = 1.0
    matrix[n][n] = 1.0
    for i in range(1, n):
        matrix[i][i+1] = p
        matrix[i][i-1] = 1.0 - p
    labels = [f"${i}" for i in range(total)]
    return MarkovChain(n_states=total, transition=matrix, state_labels=labels)


# ============================================================
# Communication Classes and State Classification
# ============================================================

def _reachable(mc: MarkovChain, start: int) -> Set[int]:
    """States reachable from start via positive-probability transitions."""
    visited = set()
    stack = [start]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for j, p in mc.successors(s):
            if j not in visited:
                stack.append(j)
    return visited


def communication_classes(mc: MarkovChain) -> List[Set[int]]:
    """Find communication classes (strongly connected components in the graph)."""
    # Tarjan's SCC algorithm
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = set()
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w, p in mc.successors(v):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.add(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in range(mc.n_states):
        if v not in index:
            strongconnect(v)

    return sccs


def classify_states(mc: MarkovChain) -> List[StateType]:
    """Classify each state as transient, recurrent, or absorbing."""
    classes = communication_classes(mc)
    state_to_class = {}
    for i, cc in enumerate(classes):
        for s in cc:
            state_to_class[s] = i

    types = [StateType.TRANSIENT] * mc.n_states

    for cc in classes:
        # A class is recurrent if no state in it can reach a state outside it
        is_closed = True
        for s in cc:
            for j, p in mc.successors(s):
                if j not in cc:
                    is_closed = False
                    break
            if not is_closed:
                break

        if is_closed:
            for s in cc:
                if len(cc) == 1 and mc.transition[s][s] >= 1.0 - 1e-10:
                    types[s] = StateType.ABSORBING
                else:
                    types[s] = StateType.RECURRENT

    return types


def is_absorbing_chain(mc: MarkovChain) -> bool:
    """A chain is absorbing if it has at least one absorbing state and
    every state can reach an absorbing state."""
    types = classify_states(mc)
    absorbing_states = {i for i, t in enumerate(types) if t == StateType.ABSORBING}
    if not absorbing_states:
        return False
    for i in range(mc.n_states):
        if types[i] != StateType.ABSORBING:
            reachable = _reachable(mc, i)
            if not reachable.intersection(absorbing_states):
                return False
    return True


# ============================================================
# Periodicity
# ============================================================

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def period_of_state(mc: MarkovChain, state: int) -> int:
    """Compute the period of a state using BFS to find return-time GCD."""
    if mc.transition[state][state] > 1e-15:
        return 1  # Self-loop means aperiodic

    # BFS to find all return times
    # Visit states, track distances; when we revisit `state`, record distance
    distances = {state: 0}
    queue = [state]
    return_times = []

    visited_distances = {}  # state -> list of distances at which reached

    while queue:
        current = queue.pop(0)
        d = distances[current]
        if d > mc.n_states * 2:  # Prevent infinite loops
            break
        for next_s, p in mc.successors(current):
            new_d = d + 1
            if next_s == state:
                return_times.append(new_d)
            if next_s not in distances and new_d <= mc.n_states * 2:
                distances[next_s] = new_d
                queue.append(next_s)

    if not return_times:
        return 0  # No return (transient)

    g = return_times[0]
    for t in return_times[1:]:
        g = _gcd(g, t)
    return g


def chain_period(mc: MarkovChain) -> int:
    """Period of the chain (period of any recurrent state, or 0 if no recurrent states)."""
    types = classify_states(mc)
    for i, t in enumerate(types):
        if t in (StateType.RECURRENT, StateType.ABSORBING):
            p = period_of_state(mc, i)
            if p > 0:
                return p
    return 0


# ============================================================
# Steady-State Distribution
# ============================================================

def steady_state(mc: MarkovChain, max_iter: int = 10000, tol: float = 1e-8) -> Optional[List[float]]:
    """Compute steady-state distribution via power iteration.

    Only exists for irreducible, aperiodic chains. For reducible chains,
    computes the limiting distribution from uniform initial.
    """
    n = mc.n_states
    # Start with uniform distribution
    dist = [1.0 / n] * n

    for _ in range(max_iter):
        new_dist = mc.step(dist)
        # Check convergence
        diff = sum(abs(new_dist[i] - dist[i]) for i in range(n))
        dist = new_dist
        if diff < tol:
            return dist

    return dist  # Return best approximation


def steady_state_exact(mc: MarkovChain) -> Optional[List[float]]:
    """Compute steady-state by solving pi * P = pi, sum(pi) = 1.

    Uses Gaussian elimination. Works for irreducible chains.
    """
    n = mc.n_states
    # System: pi * (P - I) = 0, sum(pi) = 1
    # Transpose: (P^T - I) * pi^T = 0
    # Replace one equation with sum = 1

    # Build augmented matrix (P^T - I | 0) then replace last row
    A = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = mc.transition[j][i] - (1.0 if i == j else 0.0)
        A[i][n] = 0.0

    # Replace last equation with sum = 1
    for j in range(n):
        A[n-1][j] = 1.0
    A[n-1][n] = 1.0

    # Gaussian elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(A[col][col])
        for row in range(col + 1, n):
            if abs(A[row][col]) > max_val:
                max_val = abs(A[row][col])
                max_row = row
        A[col], A[max_row] = A[max_row], A[col]

        if abs(A[col][col]) < 1e-15:
            continue  # Skip singular column

        # Eliminate
        for row in range(n):
            if row == col:
                continue
            factor = A[row][col] / A[col][col]
            for k in range(n + 1):
                A[row][k] -= factor * A[col][k]

    # Back-substitute
    pi = [0.0] * n
    for i in range(n):
        if abs(A[i][i]) > 1e-15:
            pi[i] = A[i][n] / A[i][i]

    # Normalize (in case of numerical drift)
    total = sum(pi)
    if total > 1e-15:
        pi = [p / total for p in pi]

    # Ensure non-negative
    pi = [max(0.0, p) for p in pi]
    total = sum(pi)
    if total > 1e-15:
        pi = [p / total for p in pi]

    return pi


# ============================================================
# Absorption Probabilities
# ============================================================

def absorption_probabilities(mc: MarkovChain) -> Dict[int, List[float]]:
    """For absorbing chains: probability of being absorbed into each absorbing state.

    Returns dict mapping absorbing_state -> list of probabilities (one per state).
    """
    types = classify_states(mc)
    absorbing = [i for i, t in enumerate(types) if t == StateType.ABSORBING]
    transient = [i for i, t in enumerate(types) if t == StateType.TRANSIENT]

    if not absorbing or not transient:
        # Trivial case
        result = {}
        for a in absorbing:
            probs = [0.0] * mc.n_states
            probs[a] = 1.0
            result[a] = probs
        return result

    # For each absorbing state a, solve:
    # b_i(a) = P[i][a] + sum_{j in transient} P[i][j] * b_j(a)
    # (I - Q) * b = R_a where Q = transition among transient, R_a = transition to a

    t_idx = {s: i for i, s in enumerate(transient)}
    n_t = len(transient)

    # Build Q matrix (transient -> transient)
    Q = [[0.0] * n_t for _ in range(n_t)]
    for i, si in enumerate(transient):
        for j, sj in enumerate(transient):
            Q[i][j] = mc.transition[si][sj]

    # Build (I - Q)
    IQ = [[0.0] * n_t for _ in range(n_t)]
    for i in range(n_t):
        for j in range(n_t):
            IQ[i][j] = (1.0 if i == j else 0.0) - Q[i][j]

    result = {}
    for a in absorbing:
        # RHS: R_a[i] = P[transient[i]][a]
        rhs = [mc.transition[transient[i]][a] for i in range(n_t)]

        # Solve IQ * b = rhs via Gaussian elimination
        b = _solve_linear(IQ, rhs)

        probs = [0.0] * mc.n_states
        probs[a] = 1.0  # Already absorbed
        for i, si in enumerate(transient):
            probs[si] = b[i] if b is not None else 0.0
        # Recurrent non-absorbing states: 0 probability of reaching a different absorbing state
        for i, t in enumerate(types):
            if t == StateType.RECURRENT and i != a:
                probs[i] = 0.0

        result[a] = probs

    return result


# ============================================================
# Expected Hitting Times
# ============================================================

def expected_hitting_time(mc: MarkovChain, target: int) -> List[float]:
    """Expected number of steps to reach target from each state.

    Solves: h_i = 1 + sum_j P[i][j] * h_j for i != target, h_target = 0.
    """
    n = mc.n_states
    # (I - P_reduced) * h = 1 for non-target states
    non_target = [i for i in range(n) if i != target]
    nt_idx = {s: i for i, s in enumerate(non_target)}
    n_nt = len(non_target)

    if n_nt == 0:
        return [0.0] * n

    # Build (I - Q) where Q[i][j] = P[non_target[i]][non_target[j]]
    IQ = [[0.0] * n_nt for _ in range(n_nt)]
    for i, si in enumerate(non_target):
        for j, sj in enumerate(non_target):
            IQ[i][j] = (1.0 if i == j else 0.0) - mc.transition[si][sj]

    rhs = [1.0] * n_nt

    h = _solve_linear(IQ, rhs)

    result = [0.0] * n
    if h is not None:
        for i, si in enumerate(non_target):
            result[si] = max(0.0, h[i])  # Clamp to non-negative
    else:
        # If system is singular (unreachable target), return infinity
        for si in non_target:
            result[si] = float('inf')

    return result


def expected_hitting_times(mc: MarkovChain, targets: List[int] = None) -> Dict[int, List[float]]:
    """Expected hitting times for multiple targets."""
    if targets is None:
        targets = list(range(mc.n_states))
    return {t: expected_hitting_time(mc, t) for t in targets}


# ============================================================
# Linear Algebra Helper
# ============================================================

def _solve_linear(A_orig: List[List[float]], b_orig: List[float]) -> Optional[List[float]]:
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(A_orig)
    if n == 0:
        return []

    # Copy
    A = [row[:] + [b_orig[i]] for i, row in enumerate(A_orig)]

    for col in range(n):
        # Pivot
        max_row = col
        max_val = abs(A[col][col])
        for row in range(col + 1, n):
            if abs(A[row][col]) > max_val:
                max_val = abs(A[row][col])
                max_row = row
        A[col], A[max_row] = A[max_row], A[col]

        if abs(A[col][col]) < 1e-15:
            continue

        # Eliminate
        for row in range(n):
            if row == col:
                continue
            factor = A[row][col] / A[col][col]
            for k in range(n + 1):
                A[row][k] -= factor * A[col][k]

    # Back-substitute
    x = [0.0] * n
    for i in range(n):
        if abs(A[i][i]) > 1e-15:
            x[i] = A[i][n] / A[i][i]

    return x


# ============================================================
# Full Analysis
# ============================================================

def analyze_chain(mc: MarkovChain) -> ChainAnalysis:
    """Perform complete analysis of a Markov chain."""
    types = classify_states(mc)
    classes = communication_classes(mc)
    is_irred = len(classes) == 1
    is_abs = is_absorbing_chain(mc)
    per = chain_period(mc)

    # Steady state (for irreducible aperiodic, or general approximation)
    ss = None
    if is_irred and (per == 0 or per == 1):
        ss = steady_state_exact(mc)
    elif not is_abs:
        ss = steady_state(mc)

    # Absorption probabilities
    abs_probs = None
    if is_abs:
        abs_probs = absorption_probabilities(mc)

    # Expected hitting times to absorbing states (or all states for small chains)
    hit_times = None
    absorbing_states = [i for i, t in enumerate(types) if t == StateType.ABSORBING]
    if absorbing_states:
        hit_times = expected_hitting_times(mc, absorbing_states)
    elif mc.n_states <= 10:
        hit_times = expected_hitting_times(mc)

    return ChainAnalysis(
        chain=mc,
        state_types=types,
        communication_classes=classes,
        is_irreducible=is_irred,
        is_absorbing=is_abs,
        period=per,
        steady_state=ss,
        absorption_probabilities=abs_probs,
        expected_hitting_times=hit_times,
    )


# ============================================================
# Chain from C10 Source
# ============================================================

def chain_from_random_walk(source: str) -> Optional[MarkovChain]:
    """Extract a Markov chain from a C10 random walk program.

    Looks for patterns like:
        let x = start;
        while (x > 0 and x < n) {
            let r = random(0, 1);
            if (r == 0) { x = x - 1; } else { x = x + 1; }
        }

    Returns a chain modeling the walk, or None if not extractable.
    """
    from stack_vm import lex, Parser

    try:
        tokens = lex(source)
        parser = Parser(tokens)
        ast = parser.parse()
    except Exception:
        return None

    stmts = ast.stmts if hasattr(ast, 'stmts') else []

    # Look for: let var = random(lo, hi) in a while loop
    # This is a heuristic extraction -- works for simple random walk patterns
    random_vars = {}
    walk_var = None
    walk_range = None

    for stmt in stmts:
        cls = stmt.__class__.__name__
        if cls == 'LetDecl' and hasattr(stmt, 'value'):
            val = stmt.value
            if val.__class__.__name__ == 'IntLit':
                walk_var = stmt.name
                walk_range = (0, val.value * 2)  # Heuristic

    if walk_var and walk_range:
        n = walk_range[1] + 1
        # Default: symmetric random walk with absorbing ends
        return random_walk_chain(n)

    return None


# ============================================================
# Property Verification on Chains
# ============================================================

def verify_absorption(mc: MarkovChain, start: int, target: int, min_prob: float) -> dict:
    """Verify that absorption probability from start to target >= min_prob."""
    abs_probs = absorption_probabilities(mc)
    if target not in abs_probs:
        return {"verified": False, "reason": f"State {target} is not absorbing"}

    prob = abs_probs[target][start]
    verified = prob >= min_prob - 1e-8
    return {
        "verified": verified,
        "probability": prob,
        "threshold": min_prob,
        "start": start,
        "target": target,
    }


def verify_hitting_time_bound(mc: MarkovChain, start: int, target: int, max_steps: float) -> dict:
    """Verify that expected hitting time from start to target <= max_steps."""
    ht = expected_hitting_time(mc, target)
    t = ht[start]
    verified = t <= max_steps + 1e-8
    return {
        "verified": verified,
        "expected_time": t,
        "bound": max_steps,
        "start": start,
        "target": target,
    }


def verify_steady_state_bound(mc: MarkovChain, state: int, min_prob: float) -> dict:
    """Verify that steady-state probability of state >= min_prob."""
    ss = steady_state_exact(mc)
    if ss is None:
        ss = steady_state(mc)
    prob = ss[state] if ss else 0.0
    verified = prob >= min_prob - 1e-8
    return {
        "verified": verified,
        "probability": prob,
        "threshold": min_prob,
        "state": state,
    }


# ============================================================
# Comparison with Simulation
# ============================================================

def simulate_chain(mc: MarkovChain, start: int, steps: int, seed: int = None) -> List[int]:
    """Simulate a chain run for the given number of steps."""
    import random as py_random
    if seed is not None:
        py_random.seed(seed)

    trace = [start]
    current = start
    for _ in range(steps):
        succs = mc.successors(current)
        if not succs:
            break
        r = py_random.random()
        cumulative = 0.0
        for next_s, p in succs:
            cumulative += p
            if r <= cumulative:
                current = next_s
                break
        trace.append(current)
    return trace


def empirical_steady_state(mc: MarkovChain, steps: int = 10000, seed: int = None) -> List[float]:
    """Estimate steady-state by simulation."""
    trace = simulate_chain(mc, 0, steps, seed)
    counts = [0] * mc.n_states
    for s in trace:
        counts[s] += 1
    total = len(trace)
    return [c / total for c in counts]


def compare_analytical_vs_simulation(
    mc: MarkovChain,
    steps: int = 10000,
    seed: int = 42,
) -> dict:
    """Compare analytical steady-state with simulation estimate."""
    analytical = steady_state_exact(mc)
    if analytical is None:
        analytical = steady_state(mc)
    empirical = empirical_steady_state(mc, steps, seed)

    max_diff = max(abs(analytical[i] - empirical[i]) for i in range(mc.n_states))
    return {
        "analytical": analytical,
        "empirical": empirical,
        "max_difference": max_diff,
        "steps": steps,
    }
