"""V068: Interval MDP Analysis

Interval Markov Decision Processes (IMDPs) with robust verification.
Transition probabilities are intervals [lo, hi] instead of exact values.
A property holds iff it holds for ALL valid probability distributions
within the intervals (robust/pessimistic) or for SOME distribution (optimistic).

Composes V065 (Markov chain analysis) + V067 (PCTL model checking).

Features:
- IntervalMDP construction with interval transition matrices
- Feasibility checking (does a valid distribution exist in the intervals?)
- Robust reachability: min/max probability of reaching target states
- Interval PCTL model checking (pessimistic/optimistic)
- Nondeterministic actions (MDP with interval uncertainty)
- Value iteration with interval convergence
- Comparison: point MC vs interval bounds
"""

from __future__ import annotations
import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Dict, Tuple, Union
from fractions import Fraction

# Import V065 and V067
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))

from markov_chain import MarkovChain, make_chain, analyze_chain, StateType
from pctl_model_check import (
    PCTL, FormulaKind, LabeledMC, PCTLChecker, make_labeled_mc,
    tt, ff, atom, pnot, pand, por, prob_geq, prob_leq, prob_gt, prob_lt,
    next_f, until, bounded_until, eventually, always, bounded_eventually,
    check_pctl, check_pctl_state, parse_pctl, _solve_linear,
)


# ============================================================
# Core Data Structures
# ============================================================

@dataclass
class ProbInterval:
    """An interval [lo, hi] for a transition probability."""
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi + 1e-12:
            raise ValueError(f"Invalid interval: [{self.lo}, {self.hi}]")
        self.lo = max(0.0, self.lo)
        self.hi = min(1.0, self.hi)

    def contains(self, p: float) -> bool:
        return self.lo - 1e-10 <= p <= self.hi + 1e-10

    def width(self) -> float:
        return self.hi - self.lo

    def midpoint(self) -> float:
        return (self.lo + self.hi) / 2.0

    def __repr__(self):
        return f"[{self.lo:.4f}, {self.hi:.4f}]"


@dataclass
class IntervalMDP:
    """Interval Markov Decision Process.

    For each state and action, transition probabilities are intervals.
    actions[s] = list of action names available at state s.
    transition[s][a_idx][t] = ProbInterval for going from s to t under action a.

    For a pure Interval MC (no nondeterminism), each state has exactly one action.
    """
    n_states: int
    actions: List[List[str]]  # actions[s] = list of action names
    transition: List[List[List[ProbInterval]]]  # [state][action_idx][target]
    state_labels: Optional[List[str]] = None
    ap_labels: Optional[Dict[int, Set[str]]] = None  # atomic propositions

    def __post_init__(self):
        if self.state_labels is None:
            self.state_labels = [f"s{i}" for i in range(self.n_states)]
        if self.ap_labels is None:
            self.ap_labels = {i: set() for i in range(self.n_states)}

    def is_interval_mc(self) -> bool:
        """True if no nondeterminism (each state has exactly one action)."""
        return all(len(acts) == 1 for acts in self.actions)

    def validate(self) -> List[str]:
        """Check structural validity. Does NOT check feasibility."""
        errors = []
        if len(self.actions) != self.n_states:
            errors.append(f"actions length {len(self.actions)} != n_states {self.n_states}")
        if len(self.transition) != self.n_states:
            errors.append(f"transition length {len(self.transition)} != n_states {self.n_states}")
        for s in range(self.n_states):
            if len(self.transition[s]) != len(self.actions[s]):
                errors.append(f"State {s}: {len(self.transition[s])} transition rows != {len(self.actions[s])} actions")
            for a_idx, row in enumerate(self.transition[s]):
                if len(row) != self.n_states:
                    errors.append(f"State {s} action {a_idx}: {len(row)} entries != {self.n_states}")
                for t, iv in enumerate(row):
                    if iv.lo < -1e-10:
                        errors.append(f"P[{s}][{a_idx}][{t}].lo = {iv.lo} < 0")
                    if iv.hi > 1.0 + 1e-10:
                        errors.append(f"P[{s}][{a_idx}][{t}].hi = {iv.hi} > 1")
        return errors


def make_interval_mc(intervals: List[List[Tuple[float, float]]],
                     state_labels: Optional[List[str]] = None,
                     ap_labels: Optional[Dict[int, Set[str]]] = None) -> IntervalMDP:
    """Create an Interval MC (no nondeterminism) from interval matrix.

    intervals[s][t] = (lo, hi) for P(s -> t).
    """
    n = len(intervals)
    actions = [["tau"] for _ in range(n)]
    transition = []
    for s in range(n):
        row = [ProbInterval(lo=lo, hi=hi) for (lo, hi) in intervals[s]]
        transition.append([row])
    return IntervalMDP(
        n_states=n, actions=actions, transition=transition,
        state_labels=state_labels, ap_labels=ap_labels,
    )


def make_interval_mdp(n_states: int,
                      action_transitions: Dict[int, Dict[str, List[Tuple[float, float]]]],
                      state_labels: Optional[List[str]] = None,
                      ap_labels: Optional[Dict[int, Set[str]]] = None) -> IntervalMDP:
    """Create an Interval MDP with multiple actions.

    action_transitions[s][action_name] = [(lo, hi) for each target state]
    """
    actions = []
    transition = []
    for s in range(n_states):
        state_acts = sorted(action_transitions.get(s, {"tau": [(0.0, 0.0)] * n_states}).keys())
        actions.append(state_acts)
        state_trans = []
        for a in state_acts:
            ivs = action_transitions[s][a]
            row = [ProbInterval(lo=lo, hi=hi) for (lo, hi) in ivs]
            state_trans.append(row)
        transition.append(state_trans)
    return IntervalMDP(
        n_states=n_states, actions=actions, transition=transition,
        state_labels=state_labels, ap_labels=ap_labels,
    )


# ============================================================
# Feasibility Checking
# ============================================================

def check_feasibility(imdp: IntervalMDP, state: int, action_idx: int = 0) -> Tuple[bool, Optional[List[float]]]:
    """Check if a valid probability distribution exists within the intervals.

    A distribution p is feasible if:
    - lo[t] <= p[t] <= hi[t] for all t
    - sum(p) = 1

    Uses a greedy algorithm:
    1. Start with all probabilities at their lower bounds
    2. Distribute remaining mass (1 - sum_lo) among components with slack

    Returns (feasible, distribution_or_None).
    """
    row = imdp.transition[state][action_idx]
    n = imdp.n_states

    sum_lo = sum(iv.lo for iv in row)
    sum_hi = sum(iv.hi for iv in row)

    # Quick infeasibility checks
    if sum_lo > 1.0 + 1e-10:
        return False, None
    if sum_hi < 1.0 - 1e-10:
        return False, None

    # Greedy: start at lower bounds, distribute remaining mass
    dist = [iv.lo for iv in row]
    remaining = 1.0 - sum_lo

    if remaining < -1e-10:
        return False, None

    # Distribute remaining to those with slack, proportionally
    for t in range(n):
        if remaining <= 1e-15:
            break
        slack = row[t].hi - dist[t]
        if slack > 1e-15:
            add = min(slack, remaining)
            dist[t] += add
            remaining -= add

    if abs(remaining) > 1e-8:
        return False, None

    return True, dist


def check_all_feasible(imdp: IntervalMDP) -> Tuple[bool, List[str]]:
    """Check feasibility for all state-action pairs."""
    issues = []
    for s in range(imdp.n_states):
        for a_idx in range(len(imdp.actions[s])):
            feasible, _ = check_feasibility(imdp, s, a_idx)
            if not feasible:
                issues.append(f"State {s} action {imdp.actions[s][a_idx]}: infeasible")
    return len(issues) == 0, issues


# ============================================================
# Robust Reachability (min/max probabilities)
# ============================================================

class OptimizationDirection(Enum):
    MIN = "min"
    MAX = "max"


def _optimal_distribution(row: List[ProbInterval], values: List[float],
                          direction: OptimizationDirection) -> List[float]:
    """Find the feasible distribution that optimizes sum(p[t] * values[t]).

    For MIN: assign mass to lowest-value targets first.
    For MAX: assign mass to highest-value targets first.

    Greedy algorithm: sort targets by value, assign lo first,
    then distribute remaining mass in optimal order.
    """
    n = len(row)

    # Start at lower bounds
    dist = [iv.lo for iv in row]
    remaining = 1.0 - sum(dist)

    if remaining < -1e-10:
        # Infeasible -- return lower bounds
        return dist

    # Sort targets by value (ascending for MIN, descending for MAX)
    indices = list(range(n))
    if direction == OptimizationDirection.MAX:
        indices.sort(key=lambda t: values[t], reverse=True)
    else:
        indices.sort(key=lambda t: values[t])

    # Distribute remaining mass in order
    for t in indices:
        if remaining <= 1e-15:
            break
        slack = row[t].hi - dist[t]
        if slack > 1e-15:
            add = min(slack, remaining)
            dist[t] += add
            remaining -= add

    return dist


def robust_reachability(imdp: IntervalMDP, targets: Set[int],
                        direction: OptimizationDirection = OptimizationDirection.MIN,
                        max_iter: int = 1000, tol: float = 1e-10) -> List[float]:
    """Compute min/max probability of eventually reaching target states.

    For Interval MC (no nondeterminism):
    - MIN: adversary chooses worst-case distributions to minimize reachability
    - MAX: adversary chooses best-case distributions to maximize reachability

    For Interval MDP:
    - MIN: minimizing scheduler + adversarial nature
    - MAX: maximizing scheduler + cooperative nature

    Uses value iteration:
    x_s = 1 if s in targets
    x_s = opt_{action} opt_{distribution} sum_t p[t] * x_t
    """
    n = imdp.n_states

    # Initialize
    values = [0.0] * n
    for s in targets:
        values[s] = 1.0

    for iteration in range(max_iter):
        new_values = [0.0] * n
        for s in range(n):
            if s in targets:
                new_values[s] = 1.0
                continue

            # Try all actions, pick optimal
            action_values = []
            for a_idx in range(len(imdp.actions[s])):
                row = imdp.transition[s][a_idx]
                # Find optimal distribution for this action
                dist = _optimal_distribution(row, values, direction)
                val = sum(dist[t] * values[t] for t in range(n))
                action_values.append(val)

            if action_values:
                if direction == OptimizationDirection.MAX:
                    new_values[s] = max(action_values)
                else:
                    new_values[s] = min(action_values)

        # Check convergence
        max_diff = max(abs(new_values[s] - values[s]) for s in range(n))
        values = new_values
        if max_diff < tol:
            break

    # Clamp to [0, 1]
    return [max(0.0, min(1.0, v)) for v in values]


def robust_safety(imdp: IntervalMDP, safe_states: Set[int],
                  steps: int,
                  direction: OptimizationDirection = OptimizationDirection.MIN) -> List[float]:
    """Compute min/max probability of staying in safe_states for `steps` steps.

    Bounded safety via backward induction.
    """
    n = imdp.n_states
    unsafe = set(range(n)) - safe_states

    # prob[s] = probability of staying safe for remaining steps
    prob = [1.0 if s in safe_states else 0.0 for s in range(n)]

    for step in range(steps):
        new_prob = [0.0] * n
        for s in range(n):
            if s in unsafe:
                new_prob[s] = 0.0
                continue

            action_values = []
            for a_idx in range(len(imdp.actions[s])):
                row = imdp.transition[s][a_idx]
                dist = _optimal_distribution(row, prob, direction)
                val = sum(dist[t] * prob[t] for t in range(n))
                action_values.append(val)

            if action_values:
                if direction == OptimizationDirection.MAX:
                    new_prob[s] = max(action_values)
                else:
                    new_prob[s] = min(action_values)

        prob = new_prob

    return [max(0.0, min(1.0, p)) for p in prob]


# ============================================================
# Interval PCTL Model Checking
# ============================================================

@dataclass
class IntervalPCTLResult:
    """Result of interval PCTL model checking."""
    formula: PCTL
    sat_pessimistic: Set[int]   # states satisfying under worst-case
    sat_optimistic: Set[int]    # states satisfying under best-case
    prob_min: List[float]       # min probability per state (for outermost prob)
    prob_max: List[float]       # max probability per state
    n_states: int

    def definitely_holds(self, state: int) -> bool:
        """Property holds in ALL valid resolutions."""
        return state in self.sat_pessimistic

    def possibly_holds(self, state: int) -> bool:
        """Property holds in SOME valid resolution."""
        return state in self.sat_optimistic

    def uncertain(self) -> Set[int]:
        """States where we can't determine (optimistic but not pessimistic)."""
        return self.sat_optimistic - self.sat_pessimistic

    def summary(self) -> str:
        lines = [f"Interval PCTL: {self.formula}"]
        lines.append(f"Pessimistic (all resolutions): {sorted(self.sat_pessimistic)}")
        lines.append(f"Optimistic (some resolution): {sorted(self.sat_optimistic)}")
        unc = self.uncertain()
        if unc:
            lines.append(f"Uncertain: {sorted(unc)}")
        return "\n".join(lines)


class IntervalPCTLChecker:
    """PCTL model checker for Interval MDPs.

    For each probabilistic operator P~p[path], computes both:
    - Pessimistic (min/max as needed to make check hardest)
    - Optimistic (min/max as needed to make check easiest)
    """

    def __init__(self, imdp: IntervalMDP, tol: float = 1e-10):
        self.imdp = imdp
        self.n = imdp.n_states
        self.tol = tol

    def check(self, formula: PCTL) -> IntervalPCTLResult:
        """Full interval PCTL check returning pessimistic + optimistic results."""
        sat_pess = self._check_direction(formula, pessimistic=True)
        sat_opt = self._check_direction(formula, pessimistic=False)

        # Compute probability bounds for the outermost path formula
        prob_min = [0.0] * self.n
        prob_max = [0.0] * self.n
        if formula.kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_LEQ,
                            FormulaKind.PROB_GT, FormulaKind.PROB_LT):
            prob_min = self._path_probs(formula.path, OptimizationDirection.MIN,
                                        pessimistic=True)
            prob_max = self._path_probs(formula.path, OptimizationDirection.MAX,
                                        pessimistic=True)

        return IntervalPCTLResult(
            formula=formula,
            sat_pessimistic=sat_pess,
            sat_optimistic=sat_opt,
            prob_min=prob_min,
            prob_max=prob_max,
            n_states=self.n,
        )

    def _check_direction(self, formula: PCTL, pessimistic: bool) -> Set[int]:
        """Check which states satisfy formula under pessimistic/optimistic semantics."""
        kind = formula.kind

        if kind == FormulaKind.TRUE:
            return set(range(self.n))
        elif kind == FormulaKind.FALSE:
            return set()
        elif kind == FormulaKind.ATOM:
            return {s for s, labs in self.imdp.ap_labels.items() if formula.label in labs}
        elif kind == FormulaKind.NOT:
            sub_sat = self._check_direction(formula.sub, pessimistic=not pessimistic)
            return set(range(self.n)) - sub_sat
        elif kind == FormulaKind.AND:
            return (self._check_direction(formula.left, pessimistic) &
                    self._check_direction(formula.right, pessimistic))
        elif kind == FormulaKind.OR:
            return (self._check_direction(formula.left, pessimistic) |
                    self._check_direction(formula.right, pessimistic))
        elif kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_LEQ,
                      FormulaKind.PROB_GT, FormulaKind.PROB_LT):
            return self._check_prob_direction(formula, pessimistic)
        else:
            raise ValueError(f"Unexpected formula kind: {kind}")

    def _check_prob_direction(self, formula: PCTL, pessimistic: bool) -> Set[int]:
        """Handle P~p[path] with direction-aware probability computation.

        For P>=p (pessimistic): we need to show prob >= p in worst case -> use MIN probs
        For P>=p (optimistic): we need prob >= p in some case -> use MAX probs
        For P<=p (pessimistic): we need prob <= p in worst case -> use MAX probs
        For P<=p (optimistic): we need prob <= p in some case -> use MIN probs
        """
        kind = formula.kind
        threshold = formula.threshold

        # Determine which direction to optimize probabilities
        if kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_GT):
            # Need prob >= threshold: pessimistic uses MIN, optimistic uses MAX
            if pessimistic:
                direction = OptimizationDirection.MIN
            else:
                direction = OptimizationDirection.MAX
        else:
            # P<= or P<: pessimistic uses MAX, optimistic uses MIN
            if pessimistic:
                direction = OptimizationDirection.MAX
            else:
                direction = OptimizationDirection.MIN

        probs = self._path_probs(formula.path, direction, pessimistic)

        result = set()
        for s in range(self.n):
            p = probs[s]
            if kind == FormulaKind.PROB_GEQ:
                if p >= threshold - self.tol:
                    result.add(s)
            elif kind == FormulaKind.PROB_LEQ:
                if p <= threshold + self.tol:
                    result.add(s)
            elif kind == FormulaKind.PROB_GT:
                if p > threshold + self.tol:
                    result.add(s)
            elif kind == FormulaKind.PROB_LT:
                if p < threshold - self.tol:
                    result.add(s)

        return result

    def _path_probs(self, path: PCTL, direction: OptimizationDirection,
                    pessimistic: bool) -> List[float]:
        """Compute probability bounds for path formula."""
        kind = path.kind

        if kind == FormulaKind.NEXT:
            return self._next_probs(path.sub, direction, pessimistic)
        elif kind == FormulaKind.UNTIL:
            phi_sat = self._check_direction(path.left, pessimistic)
            psi_sat = self._check_direction(path.right, pessimistic)
            return self._until_probs(phi_sat, psi_sat, direction)
        elif kind == FormulaKind.BOUNDED_UNTIL:
            phi_sat = self._check_direction(path.left, pessimistic)
            psi_sat = self._check_direction(path.right, pessimistic)
            return self._bounded_until_probs(phi_sat, psi_sat, path.bound, direction)
        else:
            raise ValueError(f"Unexpected path formula kind: {kind}")

    def _next_probs(self, phi: PCTL, direction: OptimizationDirection,
                    pessimistic: bool) -> List[float]:
        """P(X phi | s): sum of transition probs to states satisfying phi."""
        phi_sat = self._check_direction(phi, pessimistic)

        # Create value vector: 1 for phi_sat, 0 otherwise
        values = [1.0 if t in phi_sat else 0.0 for t in range(self.n)]

        probs = []
        for s in range(self.n):
            action_values = []
            for a_idx in range(len(self.imdp.actions[s])):
                row = self.imdp.transition[s][a_idx]
                dist = _optimal_distribution(row, values, direction)
                val = sum(dist[t] * values[t] for t in range(self.n))
                action_values.append(val)

            if direction == OptimizationDirection.MAX:
                probs.append(max(action_values) if action_values else 0.0)
            else:
                probs.append(min(action_values) if action_values else 0.0)

        return probs

    def _until_probs(self, phi_sat: Set[int], psi_sat: Set[int],
                     direction: OptimizationDirection,
                     max_iter: int = 1000) -> List[float]:
        """Compute P(phi U psi | s) via value iteration with interval optimization.

        Classify states: S_yes (prob=1), S_no (prob=0), S_maybe (iterate).
        """
        all_states = set(range(self.n))
        s_yes = set(psi_sat)
        s_no = all_states - phi_sat - psi_sat

        # Refine S_no: states in phi_sat that can't reach psi under MAX transitions
        # (if can't reach under best case, definitely can't reach)
        reachable = set(psi_sat)
        worklist = list(psi_sat)
        while worklist:
            t = worklist.pop()
            for s in range(self.n):
                if s not in reachable and s in phi_sat:
                    # Can s reach t with positive probability (hi > 0)?
                    for a_idx in range(len(self.imdp.actions[s])):
                        if self.imdp.transition[s][a_idx][t].hi > 1e-15:
                            reachable.add(s)
                            worklist.append(s)
                            break

        for s in phi_sat - psi_sat:
            if s not in reachable:
                s_no.add(s)

        s_maybe = all_states - s_yes - s_no

        # Value iteration for s_maybe
        values = [0.0] * self.n
        for s in s_yes:
            values[s] = 1.0

        for iteration in range(max_iter):
            new_values = values[:]
            for s in s_maybe:
                action_values = []
                for a_idx in range(len(self.imdp.actions[s])):
                    row = self.imdp.transition[s][a_idx]
                    dist = _optimal_distribution(row, values, direction)
                    val = sum(dist[t] * values[t] for t in range(self.n))
                    action_values.append(val)

                if direction == OptimizationDirection.MAX:
                    new_values[s] = max(action_values)
                else:
                    new_values[s] = min(action_values)

            max_diff = max(abs(new_values[s] - values[s]) for s in range(self.n))
            values = new_values
            if max_diff < self.tol:
                break

        return [max(0.0, min(1.0, v)) for v in values]

    def _bounded_until_probs(self, phi_sat: Set[int], psi_sat: Set[int],
                              k: int, direction: OptimizationDirection) -> List[float]:
        """Bounded until via backward induction with interval optimization."""
        n = self.n
        prob = [1.0 if s in psi_sat else 0.0 for s in range(n)]

        for step in range(k):
            new_prob = [0.0] * n
            for s in range(n):
                if s in psi_sat:
                    new_prob[s] = 1.0
                elif s in phi_sat:
                    action_values = []
                    for a_idx in range(len(self.imdp.actions[s])):
                        row = self.imdp.transition[s][a_idx]
                        dist = _optimal_distribution(row, prob, direction)
                        val = sum(dist[t] * prob[t] for t in range(n))
                        action_values.append(val)

                    if direction == OptimizationDirection.MAX:
                        new_prob[s] = max(action_values) if action_values else 0.0
                    else:
                        new_prob[s] = min(action_values) if action_values else 0.0
            prob = new_prob

        return [max(0.0, min(1.0, p)) for p in prob]


# ============================================================
# Expected Reward / Cost with Intervals
# ============================================================

def robust_expected_reward(imdp: IntervalMDP, rewards: List[float],
                           targets: Set[int],
                           direction: OptimizationDirection = OptimizationDirection.MIN,
                           max_iter: int = 1000, tol: float = 1e-8) -> List[float]:
    """Compute min/max expected cumulative reward until reaching targets.

    reward[s] is earned each step in state s. Target states earn 0 reward.

    For MIN direction: adversary minimizes total reward (nature picks worst distribution).
    For MAX direction: scheduler maximizes total reward.

    Returns expected reward from each state. Infinity represented as float('inf').
    """
    n = imdp.n_states
    values = [0.0] * n  # targets have 0 reward-to-go

    for iteration in range(max_iter):
        new_values = [0.0] * n
        for s in range(n):
            if s in targets:
                new_values[s] = 0.0
                continue

            action_values = []
            for a_idx in range(len(imdp.actions[s])):
                row = imdp.transition[s][a_idx]
                dist = _optimal_distribution(row, values, direction)
                val = rewards[s] + sum(dist[t] * values[t] for t in range(n))
                action_values.append(val)

            if action_values:
                if direction == OptimizationDirection.MAX:
                    new_values[s] = max(action_values)
                else:
                    new_values[s] = min(action_values)

        max_diff = max(abs(new_values[s] - values[s]) for s in range(n))
        values = new_values
        if max_diff < tol:
            break

    return values


# ============================================================
# Point Resolution: extract a concrete MC from intervals
# ============================================================

def resolve_to_mc(imdp: IntervalMDP, strategy: str = "midpoint") -> MarkovChain:
    """Resolve interval MC to a concrete MC.

    strategy: "midpoint" uses interval midpoints, "lower" uses lower bounds (adjusted),
    "upper" uses upper bounds (adjusted).
    """
    if not imdp.is_interval_mc():
        raise ValueError("Can only resolve Interval MC (no nondeterminism)")

    n = imdp.n_states
    matrix = []
    for s in range(n):
        row_ivs = imdp.transition[s][0]
        if strategy == "midpoint":
            raw = [iv.midpoint() for iv in row_ivs]
        elif strategy == "lower":
            raw = [iv.lo for iv in row_ivs]
        elif strategy == "upper":
            raw = [iv.hi for iv in row_ivs]
        else:
            raw = [iv.midpoint() for iv in row_ivs]

        # Normalize to sum to 1
        total = sum(raw)
        if total > 1e-15:
            row = [p / total for p in raw]
        else:
            # Uniform fallback
            row = [1.0 / n] * n
        matrix.append(row)

    return make_chain(matrix, imdp.state_labels)


def resolve_feasible(imdp: IntervalMDP, state: int, action_idx: int = 0) -> Optional[List[float]]:
    """Get a feasible distribution for a state-action pair."""
    feasible, dist = check_feasibility(imdp, state, action_idx)
    return dist if feasible else None


# ============================================================
# Comparison: Point MC vs Interval Bounds
# ============================================================

def compare_point_vs_interval(imdp: IntervalMDP, targets: Set[int],
                               formula: Optional[PCTL] = None) -> dict:
    """Compare point MC analysis with interval bounds.

    Shows the gap between precise (point) and robust (interval) verification.
    """
    result = {}

    # Interval bounds
    prob_min = robust_reachability(imdp, targets, OptimizationDirection.MIN)
    prob_max = robust_reachability(imdp, targets, OptimizationDirection.MAX)

    # Point MC (midpoint resolution)
    if imdp.is_interval_mc():
        mc = resolve_to_mc(imdp, "midpoint")
        lmc = LabeledMC(mc=mc, labels=imdp.ap_labels or {})
        # Use V067 for point probabilities
        checker = PCTLChecker(lmc)
        point_probs = [0.0] * imdp.n_states
        for s in range(imdp.n_states):
            p = sum(mc.transition[s][t] for t in targets if t < imdp.n_states)
            point_probs[s] = 0.0  # This is just next-step; use reachability instead

        # Reachability for point MC via PCTL
        from pctl_model_check import check_pctl_quantitative
        # Create an atom for target
        # Actually, let's just compute reachability directly
        point_reach = _point_reachability(mc, targets)
        result["point_reachability"] = point_reach
    else:
        result["point_reachability"] = None

    result["min_reachability"] = prob_min
    result["max_reachability"] = prob_max
    result["gap"] = [prob_max[s] - prob_min[s] for s in range(imdp.n_states)]
    result["max_gap"] = max(prob_max[s] - prob_min[s] for s in range(imdp.n_states))

    if formula and imdp.is_interval_mc():
        checker_iv = IntervalPCTLChecker(imdp)
        iv_result = checker_iv.check(formula)
        result["interval_pctl"] = iv_result

    return result


def _point_reachability(mc: MarkovChain, targets: Set[int],
                        max_iter: int = 1000, tol: float = 1e-10) -> List[float]:
    """Compute reachability probabilities for a point MC via value iteration."""
    n = mc.n_states
    values = [0.0] * n
    for s in targets:
        values[s] = 1.0

    for _ in range(max_iter):
        new_values = [0.0] * n
        for s in range(n):
            if s in targets:
                new_values[s] = 1.0
            else:
                new_values[s] = sum(mc.transition[s][t] * values[t] for t in range(n))
        max_diff = max(abs(new_values[s] - values[s]) for s in range(n))
        values = new_values
        if max_diff < tol:
            break

    return values


# ============================================================
# Interval Width Analysis
# ============================================================

def interval_width_analysis(imdp: IntervalMDP) -> dict:
    """Analyze the uncertainty in the interval MDP."""
    total_width = 0.0
    max_width = 0.0
    n_intervals = 0
    tight_count = 0  # intervals with width < 1e-10 (essentially exact)

    for s in range(imdp.n_states):
        for a_idx in range(len(imdp.actions[s])):
            for t in range(imdp.n_states):
                iv = imdp.transition[s][a_idx][t]
                w = iv.width()
                total_width += w
                max_width = max(max_width, w)
                n_intervals += 1
                if w < 1e-10:
                    tight_count += 1

    return {
        "total_intervals": n_intervals,
        "tight_intervals": tight_count,
        "uncertain_intervals": n_intervals - tight_count,
        "mean_width": total_width / max(1, n_intervals),
        "max_width": max_width,
        "uncertainty_ratio": (n_intervals - tight_count) / max(1, n_intervals),
    }


# ============================================================
# Sensitivity Analysis
# ============================================================

def sensitivity_analysis(imdp: IntervalMDP, targets: Set[int],
                         perturbation: float = 0.01) -> Dict[Tuple[int, int, int], float]:
    """Measure sensitivity of min reachability to each interval.

    For each transition interval, tighten it by `perturbation` and measure
    the change in min reachability probability. Higher sensitivity means
    that interval matters more for verification precision.

    Returns dict mapping (state, action_idx, target) -> sensitivity score.
    """
    base_probs = robust_reachability(imdp, targets, OptimizationDirection.MIN)
    sensitivities = {}

    for s in range(imdp.n_states):
        for a_idx in range(len(imdp.actions[s])):
            for t in range(imdp.n_states):
                iv = imdp.transition[s][a_idx][t]
                if iv.width() < perturbation * 2:
                    continue  # Can't tighten enough

                # Tighten interval from both sides
                new_lo = iv.lo + perturbation
                new_hi = iv.hi - perturbation
                if new_lo > new_hi:
                    continue

                # Create modified IMDP
                orig_iv = imdp.transition[s][a_idx][t]
                imdp.transition[s][a_idx][t] = ProbInterval(new_lo, new_hi)

                new_probs = robust_reachability(imdp, targets, OptimizationDirection.MIN)
                change = sum(abs(new_probs[i] - base_probs[i]) for i in range(imdp.n_states))
                sensitivities[(s, a_idx, t)] = change

                # Restore
                imdp.transition[s][a_idx][t] = orig_iv

    return sensitivities


# ============================================================
# High-level API
# ============================================================

def check_interval_pctl(imdp: IntervalMDP, formula: PCTL) -> IntervalPCTLResult:
    """Main API: check PCTL formula on Interval MDP."""
    checker = IntervalPCTLChecker(imdp)
    return checker.check(formula)


def check_interval_pctl_state(imdp: IntervalMDP, state: int,
                               formula: PCTL) -> Dict[str, bool]:
    """Check if a specific state satisfies formula (pessimistic + optimistic)."""
    result = check_interval_pctl(imdp, formula)
    return {
        "definitely": result.definitely_holds(state),
        "possibly": result.possibly_holds(state),
        "uncertain": state in result.uncertain(),
    }


def verify_robust_property(imdp: IntervalMDP, targets: Set[int],
                            min_prob: float) -> dict:
    """Verify that P(reach targets) >= min_prob holds robustly (all resolutions)."""
    prob_min = robust_reachability(imdp, targets, OptimizationDirection.MIN)
    prob_max = robust_reachability(imdp, targets, OptimizationDirection.MAX)

    verified_states = {s for s in range(imdp.n_states) if prob_min[s] >= min_prob - 1e-10}
    possible_states = {s for s in range(imdp.n_states) if prob_max[s] >= min_prob - 1e-10}

    return {
        "min_probs": prob_min,
        "max_probs": prob_max,
        "verified_states": verified_states,
        "possible_states": possible_states,
        "uncertain_states": possible_states - verified_states,
        "verdict": "ROBUST" if len(verified_states) == imdp.n_states else
                   "VIOLATED" if len(possible_states) == 0 else "UNCERTAIN",
    }


def batch_interval_check(imdp: IntervalMDP,
                          formulas: List[PCTL]) -> List[IntervalPCTLResult]:
    """Check multiple PCTL formulas on the same interval MDP."""
    checker = IntervalPCTLChecker(imdp)
    return [checker.check(f) for f in formulas]
