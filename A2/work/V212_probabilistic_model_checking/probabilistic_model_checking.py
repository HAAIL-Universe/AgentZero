"""V212: Probabilistic Model Checking (PRISM-style).

Verification of probabilistic systems using:
1. DTMC (Discrete-Time Markov Chains) -- stochastic state transitions
2. CTMC (Continuous-Time Markov Chains) -- continuous-time transitions via rates
3. PCTL (Probabilistic Computation Tree Logic) -- property specification
4. CSL (Continuous Stochastic Logic) -- CTMC property specification
5. Numerical model checking -- iterative/matrix-based probability computation
6. Steady-state analysis -- long-run behavior

Composes V209 (Factor operations for reward models).

Key algorithms:
- PCTL P>=p [phi U<=k psi]: bounded until probability via matrix power
- PCTL P>=p [phi U psi]: unbounded until via linear system solving
- CSL time-bounded reachability via uniformization (Jensen's method)
- Steady-state distribution via power iteration
- Expected reward computation (cumulative and reachability)
"""

from __future__ import annotations
import math
from collections import defaultdict
from enum import Enum, auto
from typing import Callable


# ---------------------------------------------------------------------------
# DTMC: Discrete-Time Markov Chain
# ---------------------------------------------------------------------------

class DTMC:
    """Discrete-Time Markov Chain.

    States are strings. Transitions are probability distributions.
    """

    def __init__(self):
        self.states: list[str] = []
        self._state_set: set[str] = set()
        self.transitions: dict[str, dict[str, float]] = {}  # s -> {s' -> prob}
        self.labels: dict[str, set[str]] = {}  # state -> set of atomic props
        self.initial: str | None = None
        self.rewards: dict[str, dict[str, float]] = {}  # reward_name -> {state -> value}

    def add_state(self, name: str, labels: set[str] | None = None):
        """Add a state with optional atomic proposition labels."""
        if name not in self._state_set:
            self.states.append(name)
            self._state_set.add(name)
            self.transitions[name] = {}
            self.labels[name] = labels or set()

    def add_transition(self, src: str, dst: str, prob: float):
        """Add a probabilistic transition."""
        self.transitions[src][dst] = prob

    def set_initial(self, state: str):
        """Set the initial state."""
        self.initial = state

    def add_reward(self, name: str, state: str, value: float):
        """Add a state reward."""
        if name not in self.rewards:
            self.rewards[name] = {}
        self.rewards[name][state] = value

    def get_successors(self, state: str) -> dict[str, float]:
        """Get successor distribution for a state."""
        return self.transitions.get(state, {})

    def is_absorbing(self, state: str) -> bool:
        """Check if state is absorbing (only self-loop or no transitions)."""
        succs = self.transitions.get(state, {})
        if not succs:
            return True
        return len(succs) == 1 and state in succs and abs(succs[state] - 1.0) < 1e-12

    def validate(self) -> list[str]:
        """Validate that transition probabilities sum to 1 for each state."""
        errors = []
        for s in self.states:
            total = sum(self.transitions[s].values())
            if self.transitions[s] and abs(total - 1.0) > 1e-9:
                errors.append(f"State {s}: probabilities sum to {total}")
        return errors

    def state_index(self, state: str) -> int:
        """Get index of state in state list."""
        return self.states.index(state)

    def sat(self, label: str) -> set[str]:
        """Return set of states satisfying an atomic proposition."""
        return {s for s in self.states if label in self.labels[s]}


# ---------------------------------------------------------------------------
# CTMC: Continuous-Time Markov Chain
# ---------------------------------------------------------------------------

class CTMC:
    """Continuous-Time Markov Chain.

    Transitions have rates (not probabilities).
    Rate r means expected time 1/r before transition fires.
    """

    def __init__(self):
        self.states: list[str] = []
        self._state_set: set[str] = set()
        self.rates: dict[str, dict[str, float]] = {}  # s -> {s' -> rate}
        self.labels: dict[str, set[str]] = {}
        self.initial: str | None = None
        self.rewards: dict[str, dict[str, float]] = {}

    def add_state(self, name: str, labels: set[str] | None = None):
        if name not in self._state_set:
            self.states.append(name)
            self._state_set.add(name)
            self.rates[name] = {}
            self.labels[name] = labels or set()

    def add_rate(self, src: str, dst: str, rate: float):
        """Add a transition rate."""
        self.rates[src][dst] = rate

    def set_initial(self, state: str):
        self.initial = state

    def add_reward(self, name: str, state: str, value: float):
        if name not in self.rewards:
            self.rewards[name] = {}
        self.rewards[name][state] = value

    def exit_rate(self, state: str) -> float:
        """Total exit rate from a state."""
        return sum(self.rates.get(state, {}).values())

    def embedded_dtmc(self) -> DTMC:
        """Extract the embedded DTMC (normalize rates to probabilities)."""
        dtmc = DTMC()
        for s in self.states:
            dtmc.add_state(s, self.labels[s].copy())
        dtmc.initial = self.initial

        for s in self.states:
            E = self.exit_rate(s)
            if E > 0:
                for s2, r in self.rates[s].items():
                    dtmc.add_transition(s, s2, r / E)
            else:
                # Absorbing state: self-loop with prob 1
                dtmc.add_transition(s, s, 1.0)

        for rname, rvals in self.rewards.items():
            for s, v in rvals.items():
                dtmc.add_reward(rname, s, v)

        return dtmc

    def sat(self, label: str) -> set[str]:
        return {s for s in self.states if label in self.labels[s]}

    def uniformization_rate(self) -> float:
        """Maximum exit rate (for uniformization)."""
        if not self.states:
            return 1.0
        return max(self.exit_rate(s) for s in self.states)


# ---------------------------------------------------------------------------
# PCTL: Probabilistic Computation Tree Logic
# ---------------------------------------------------------------------------

class PCTLOp(Enum):
    TRUE = auto()
    ATOM = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    PROB_BOUND = auto()   # P>=p [path_formula]
    PROB_QUERY = auto()   # P=? [path_formula]
    STEADY = auto()       # S>=p [phi]
    STEADY_QUERY = auto() # S=? [phi]
    REWARD_BOUND = auto() # R>=r [...]
    REWARD_QUERY = auto() # R=? [...]


class PathOp(Enum):
    NEXT = auto()         # X phi
    UNTIL = auto()        # phi U phi
    BOUNDED_UNTIL = auto() # phi U<=k phi
    EVENTUALLY = auto()   # F phi (sugar for true U phi)
    ALWAYS = auto()       # G phi


class PCTLFormula:
    """PCTL state or path formula."""

    def __init__(self, op: PCTLOp, **kwargs):
        self.op = op
        self.label = kwargs.get('label', '')
        self.sub = kwargs.get('sub', None)      # single sub-formula
        self.left = kwargs.get('left', None)     # binary left
        self.right = kwargs.get('right', None)   # binary right
        self.bound = kwargs.get('bound', 0.0)    # probability/reward bound
        self.comp = kwargs.get('comp', '>=')     # comparison: >=, >, <=, <
        self.path_op = kwargs.get('path_op', None)
        self.path_left = kwargs.get('path_left', None)   # left of Until
        self.path_right = kwargs.get('path_right', None)  # right of Until
        self.steps = kwargs.get('steps', None)   # step bound for bounded until
        self.reward_name = kwargs.get('reward_name', '')
        self.cumulative = kwargs.get('cumulative', False)

    def __repr__(self):
        if self.op == PCTLOp.ATOM:
            return f'Atom({self.label})'
        if self.op == PCTLOp.TRUE:
            return 'True'
        if self.op == PCTLOp.NOT:
            return f'!{self.sub}'
        if self.op == PCTLOp.AND:
            return f'({self.left} & {self.right})'
        if self.op == PCTLOp.OR:
            return f'({self.left} | {self.right})'
        if self.op == PCTLOp.PROB_BOUND:
            return f'P{self.comp}{self.bound}[{self.path_op}]'
        if self.op == PCTLOp.PROB_QUERY:
            return f'P=?[{self.path_op}]'
        return f'PCTL({self.op})'


# Convenience constructors
def tt() -> PCTLFormula:
    return PCTLFormula(PCTLOp.TRUE)

def atom(label: str) -> PCTLFormula:
    return PCTLFormula(PCTLOp.ATOM, label=label)

def neg(f: PCTLFormula) -> PCTLFormula:
    return PCTLFormula(PCTLOp.NOT, sub=f)

def conj(a: PCTLFormula, b: PCTLFormula) -> PCTLFormula:
    return PCTLFormula(PCTLOp.AND, left=a, right=b)

def disj(a: PCTLFormula, b: PCTLFormula) -> PCTLFormula:
    return PCTLFormula(PCTLOp.OR, left=a, right=b)

def prob_bound(comp: str, bound: float, path_op: PathOp,
               path_left: PCTLFormula | None = None,
               path_right: PCTLFormula | None = None,
               steps: int | None = None) -> PCTLFormula:
    """P>=p [path_formula]."""
    return PCTLFormula(PCTLOp.PROB_BOUND, comp=comp, bound=bound,
                       path_op=path_op, path_left=path_left,
                       path_right=path_right, steps=steps)

def prob_query(path_op: PathOp,
               path_left: PCTLFormula | None = None,
               path_right: PCTLFormula | None = None,
               steps: int | None = None) -> PCTLFormula:
    """P=? [path_formula]."""
    return PCTLFormula(PCTLOp.PROB_QUERY, path_op=path_op,
                       path_left=path_left, path_right=path_right,
                       steps=steps)

def steady_bound(comp: str, bound: float, sub: PCTLFormula) -> PCTLFormula:
    """S>=p [phi]."""
    return PCTLFormula(PCTLOp.STEADY, comp=comp, bound=bound, sub=sub)

def steady_query(sub: PCTLFormula) -> PCTLFormula:
    """S=? [phi]."""
    return PCTLFormula(PCTLOp.STEADY_QUERY, sub=sub)

def reward_bound(comp: str, bound: float, reward_name: str,
                 path_op: PathOp, path_right: PCTLFormula | None = None,
                 steps: int | None = None, cumulative: bool = False) -> PCTLFormula:
    """R>=r [F phi] or R>=r [C<=k]."""
    return PCTLFormula(PCTLOp.REWARD_BOUND, comp=comp, bound=bound,
                       reward_name=reward_name, path_op=path_op,
                       path_right=path_right, steps=steps, cumulative=cumulative)

def reward_query(reward_name: str, path_op: PathOp,
                 path_right: PCTLFormula | None = None,
                 steps: int | None = None, cumulative: bool = False) -> PCTLFormula:
    """R=? [F phi] or R=? [C<=k]."""
    return PCTLFormula(PCTLOp.REWARD_QUERY, reward_name=reward_name,
                       path_op=path_op, path_right=path_right,
                       steps=steps, cumulative=cumulative)


# ---------------------------------------------------------------------------
# PCTL Model Checker for DTMCs
# ---------------------------------------------------------------------------

class DTMCModelChecker:
    """PCTL model checking over DTMCs.

    Implements:
    - Qualitative: which states satisfy P>=p [path_formula]
    - Quantitative: compute exact probabilities P=? [path_formula]
    - Steady-state: long-run probabilities
    - Expected rewards: reachability and cumulative
    """

    def __init__(self, dtmc: DTMC, epsilon: float = 1e-10, max_iter: int = 10000):
        self.dtmc = dtmc
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n = len(dtmc.states)

    def check(self, formula: PCTLFormula) -> set[str] | dict[str, float]:
        """Check a PCTL formula. Returns sat set or probability/reward map."""
        if formula.op == PCTLOp.TRUE:
            return set(self.dtmc.states)
        elif formula.op == PCTLOp.ATOM:
            return self.dtmc.sat(formula.label)
        elif formula.op == PCTLOp.NOT:
            sat_sub = self.check(formula.sub)
            return set(self.dtmc.states) - sat_sub
        elif formula.op == PCTLOp.AND:
            return self.check(formula.left) & self.check(formula.right)
        elif formula.op == PCTLOp.OR:
            return self.check(formula.left) | self.check(formula.right)
        elif formula.op == PCTLOp.PROB_BOUND:
            return self._check_prob_bound(formula)
        elif formula.op == PCTLOp.PROB_QUERY:
            return self._check_prob_query(formula)
        elif formula.op == PCTLOp.STEADY:
            return self._check_steady_bound(formula)
        elif formula.op == PCTLOp.STEADY_QUERY:
            return self._check_steady_query(formula)
        elif formula.op == PCTLOp.REWARD_BOUND:
            return self._check_reward_bound(formula)
        elif formula.op == PCTLOp.REWARD_QUERY:
            return self._check_reward_query(formula)
        else:
            raise ValueError(f"Unknown formula op: {formula.op}")

    def _check_prob_bound(self, formula: PCTLFormula) -> set[str]:
        """Check P>=p [path]. Returns states satisfying the bound."""
        probs = self._compute_path_probs(formula)
        return self._filter_by_bound(probs, formula.comp, formula.bound)

    def _check_prob_query(self, formula: PCTLFormula) -> dict[str, float]:
        """Compute P=? [path]. Returns probability map."""
        return self._compute_path_probs(formula)

    def _compute_path_probs(self, formula: PCTLFormula) -> dict[str, float]:
        """Compute path probabilities for each state."""
        path_op = formula.path_op

        if path_op == PathOp.NEXT:
            return self._prob_next(formula.path_right)
        elif path_op == PathOp.BOUNDED_UNTIL:
            return self._prob_bounded_until(
                formula.path_left, formula.path_right, formula.steps)
        elif path_op == PathOp.UNTIL:
            return self._prob_until(formula.path_left, formula.path_right)
        elif path_op == PathOp.EVENTUALLY:
            return self._prob_until(tt(), formula.path_right)
        elif path_op == PathOp.ALWAYS:
            # G phi = 1 - P[true U !phi]
            neg_probs = self._prob_until(tt(), neg(formula.path_right))
            return {s: 1.0 - p for s, p in neg_probs.items()}
        else:
            raise ValueError(f"Unknown path op: {path_op}")

    def _prob_next(self, phi: PCTLFormula) -> dict[str, float]:
        """Compute P(X phi) for each state."""
        sat_phi = self.check(phi)
        result = {}
        for s in self.dtmc.states:
            p = sum(prob for s2, prob in self.dtmc.transitions[s].items()
                    if s2 in sat_phi)
            result[s] = p
        return result

    def _prob_bounded_until(self, phi: PCTLFormula, psi: PCTLFormula,
                             k: int) -> dict[str, float]:
        """Compute P(phi U<=k psi) via iterative matrix-vector multiplication."""
        sat_phi = self.check(phi)
        sat_psi = self.check(psi)

        # prob0: states that can never reach psi (satisfy neither phi nor psi)
        # prob1: states already satisfying psi
        probs = {s: 0.0 for s in self.dtmc.states}
        for s in sat_psi:
            probs[s] = 1.0

        # Iterative computation: k steps
        for step in range(k):
            new_probs = {}
            for s in self.dtmc.states:
                if s in sat_psi:
                    new_probs[s] = 1.0
                elif s not in sat_phi:
                    new_probs[s] = 0.0
                else:
                    # Sum over successors
                    p = sum(prob * probs[s2]
                            for s2, prob in self.dtmc.transitions[s].items())
                    new_probs[s] = p
            probs = new_probs

        return probs

    def _prob_until(self, phi: PCTLFormula, psi: PCTLFormula) -> dict[str, float]:
        """Compute P(phi U psi) via value iteration.

        Partition states into:
        - prob1: states in sat(psi) -> probability 1
        - prob0: states not in sat(phi) and not in sat(psi) -> probability 0
        - unknown: remaining states -> solve iteratively
        """
        sat_phi = self.check(phi)
        sat_psi = self.check(psi)

        # States definitely reaching psi with prob 1: in sat_psi
        # States definitely reaching psi with prob 0: can't reach sat_psi via sat_phi
        prob1 = set(sat_psi)

        # Compute prob0: states from which psi is unreachable via phi-paths
        # BFS backward from sat_psi through phi-states
        can_reach_psi = set(sat_psi)
        worklist = list(sat_psi)
        while worklist:
            s = worklist.pop()
            for s2 in self.dtmc.states:
                if s2 not in can_reach_psi and s2 in sat_phi:
                    if s in self.dtmc.transitions.get(s2, {}):
                        can_reach_psi.add(s2)
                        worklist.append(s2)

        prob0 = set(self.dtmc.states) - can_reach_psi

        # Unknown states: those that can reach psi but haven't yet
        unknown = set(self.dtmc.states) - prob1 - prob0

        # Value iteration on unknown states
        probs = {s: 0.0 for s in self.dtmc.states}
        for s in prob1:
            probs[s] = 1.0

        for _ in range(self.max_iter):
            max_diff = 0.0
            for s in unknown:
                if s not in sat_phi:
                    new_p = 0.0
                else:
                    new_p = sum(prob * probs[s2]
                                for s2, prob in self.dtmc.transitions[s].items())
                diff = abs(new_p - probs[s])
                if diff > max_diff:
                    max_diff = diff
                probs[s] = new_p
            if max_diff < self.epsilon:
                break

        return probs

    def steady_state(self) -> dict[str, float]:
        """Compute steady-state distribution via power iteration.

        For ergodic chains, this converges to the unique stationary distribution.
        """
        n = self.n
        if n == 0:
            return {}

        # Start with uniform distribution
        dist = {s: 1.0 / n for s in self.dtmc.states}

        for _ in range(self.max_iter):
            new_dist = {s: 0.0 for s in self.dtmc.states}
            for s in self.dtmc.states:
                for s2, prob in self.dtmc.transitions[s].items():
                    new_dist[s2] += dist[s] * prob

            max_diff = max(abs(new_dist[s] - dist[s]) for s in self.dtmc.states)
            dist = new_dist
            if max_diff < self.epsilon:
                break

        return dist

    def _check_steady_bound(self, formula: PCTLFormula) -> set[str]:
        """S>=p [phi]: check if steady-state probability of phi satisfies bound."""
        sat_phi = self.check(formula.sub)
        ss = self.steady_state()
        total_prob = sum(ss[s] for s in sat_phi)
        if self._compare(total_prob, formula.comp, formula.bound):
            return set(self.dtmc.states)  # All states satisfy (property is global)
        return set()

    def _check_steady_query(self, formula: PCTLFormula) -> dict[str, float]:
        """S=? [phi]: return steady-state probability of phi."""
        sat_phi = self.check(formula.sub)
        ss = self.steady_state()
        total_prob = sum(ss[s] for s in sat_phi)
        # Steady-state is a global property, same for all states
        return {s: total_prob for s in self.dtmc.states}

    def _check_reward_bound(self, formula: PCTLFormula) -> set[str]:
        """R>=r [F phi] or R>=r [C<=k]."""
        rewards = self._compute_rewards(formula)
        return self._filter_by_bound(rewards, formula.comp, formula.bound)

    def _check_reward_query(self, formula: PCTLFormula) -> dict[str, float]:
        """R=? [F phi] or R=? [C<=k]."""
        return self._compute_rewards(formula)

    def _compute_rewards(self, formula: PCTLFormula) -> dict[str, float]:
        """Compute expected rewards."""
        rname = formula.reward_name
        rew = self.dtmc.rewards.get(rname, {})

        if formula.cumulative and formula.steps is not None:
            return self._cumulative_reward(rew, formula.steps)
        elif formula.path_op == PathOp.EVENTUALLY:
            return self._reachability_reward(rew, formula.path_right)
        elif formula.path_op == PathOp.BOUNDED_UNTIL and formula.steps is not None:
            return self._cumulative_reward(rew, formula.steps)
        else:
            return self._reachability_reward(rew, formula.path_right)

    def _cumulative_reward(self, rew: dict[str, float], k: int) -> dict[str, float]:
        """Expected cumulative reward over k steps."""
        rewards = {s: 0.0 for s in self.dtmc.states}

        for step in range(k):
            new_rewards = {}
            for s in self.dtmc.states:
                r = rew.get(s, 0.0)
                future = sum(prob * rewards[s2]
                             for s2, prob in self.dtmc.transitions[s].items())
                new_rewards[s] = r + future
            rewards = new_rewards

        return rewards

    def _reachability_reward(self, rew: dict[str, float],
                              target_formula: PCTLFormula | None) -> dict[str, float]:
        """Expected reward to reach target states (R[F phi]).

        Value iteration: R(s) = rew(s) + sum_s' P(s,s') * R(s')
        for non-target states. Target states have R = 0.
        """
        if target_formula is None:
            target = set()
        else:
            target = self.check(target_formula)

        rewards = {s: 0.0 for s in self.dtmc.states}

        for _ in range(self.max_iter):
            max_diff = 0.0
            for s in self.dtmc.states:
                if s in target:
                    continue  # Target states: reward = 0
                if self.dtmc.is_absorbing(s):
                    rewards[s] = float('inf')  # Can't reach target
                    continue
                r = rew.get(s, 0.0)
                future = sum(prob * rewards[s2]
                             for s2, prob in self.dtmc.transitions[s].items())
                new_r = r + future
                if not math.isinf(new_r):
                    diff = abs(new_r - rewards[s])
                    if diff > max_diff:
                        max_diff = diff
                rewards[s] = new_r
            if max_diff < self.epsilon:
                break

        return rewards

    def _filter_by_bound(self, values: dict[str, float], comp: str,
                          bound: float) -> set[str]:
        """Filter states satisfying the comparison."""
        result = set()
        for s, v in values.items():
            if self._compare(v, comp, bound):
                result.add(s)
        return result

    @staticmethod
    def _compare(value: float, comp: str, bound: float) -> bool:
        if comp == '>=':
            return value >= bound - 1e-9
        elif comp == '>':
            return value > bound + 1e-9
        elif comp == '<=':
            return value <= bound + 1e-9
        elif comp == '<':
            return value < bound - 1e-9
        return False


# ---------------------------------------------------------------------------
# CTMC Model Checker (CSL)
# ---------------------------------------------------------------------------

class CTMCModelChecker:
    """CSL model checking over CTMCs.

    Uses uniformization (Jensen's method) for time-bounded properties.
    """

    def __init__(self, ctmc: CTMC, epsilon: float = 1e-10, max_iter: int = 10000):
        self.ctmc = ctmc
        self.epsilon = epsilon
        self.max_iter = max_iter

    def check(self, formula: PCTLFormula) -> set[str] | dict[str, float]:
        """Check a CSL formula (reuses PCTL formula structure)."""
        if formula.op == PCTLOp.TRUE:
            return set(self.ctmc.states)
        elif formula.op == PCTLOp.ATOM:
            return self.ctmc.sat(formula.label)
        elif formula.op == PCTLOp.NOT:
            sat_sub = self.check(formula.sub)
            return set(self.ctmc.states) - sat_sub
        elif formula.op == PCTLOp.AND:
            return self.check(formula.left) & self.check(formula.right)
        elif formula.op == PCTLOp.OR:
            return self.check(formula.left) | self.check(formula.right)
        elif formula.op == PCTLOp.PROB_BOUND:
            probs = self._compute_probs(formula)
            return DTMCModelChecker._filter_by_bound(None, probs, formula.comp, formula.bound)
        elif formula.op == PCTLOp.PROB_QUERY:
            return self._compute_probs(formula)
        elif formula.op == PCTLOp.STEADY:
            return self._check_steady_bound(formula)
        elif formula.op == PCTLOp.STEADY_QUERY:
            return self._check_steady_query(formula)
        else:
            raise ValueError(f"Unknown formula op: {formula.op}")

    def _compute_probs(self, formula: PCTLFormula) -> dict[str, float]:
        """Compute path probabilities for CTMC."""
        path_op = formula.path_op

        if path_op == PathOp.NEXT:
            # For CTMC, next is defined via embedded DTMC
            edtmc = self.ctmc.embedded_dtmc()
            checker = DTMCModelChecker(edtmc, self.epsilon, self.max_iter)
            return checker._prob_next(formula.path_right)
        elif path_op == PathOp.BOUNDED_UNTIL:
            return self._time_bounded_until(
                formula.path_left, formula.path_right, formula.steps)
        elif path_op == PathOp.UNTIL:
            # Unbounded until: use embedded DTMC
            edtmc = self.ctmc.embedded_dtmc()
            checker = DTMCModelChecker(edtmc, self.epsilon, self.max_iter)
            return checker._prob_until(formula.path_left, formula.path_right)
        elif path_op == PathOp.EVENTUALLY:
            edtmc = self.ctmc.embedded_dtmc()
            checker = DTMCModelChecker(edtmc, self.epsilon, self.max_iter)
            return checker._prob_until(tt(), formula.path_right)
        else:
            raise ValueError(f"Unknown path op for CTMC: {path_op}")

    def _time_bounded_until(self, phi: PCTLFormula, psi: PCTLFormula,
                             time_bound: float) -> dict[str, float]:
        """CSL time-bounded until via uniformization (Jensen's method).

        Computes P(phi U<=t psi) for each state.

        Uniformization transforms CTMC into a DTMC with uniform exit rate q,
        where q >= max exit rate. The number of transitions in time t follows
        a Poisson distribution with parameter q*t.
        """
        sat_phi = self.check(phi)
        sat_psi = self.check(psi)
        states = self.ctmc.states
        n = len(states)

        if time_bound <= 0:
            return {s: (1.0 if s in sat_psi else 0.0) for s in states}

        # Uniformization rate
        q = self.ctmc.uniformization_rate()
        if q == 0:
            return {s: (1.0 if s in sat_psi else 0.0) for s in states}

        # Build uniformized DTMC transition matrix (as dict)
        # P_unif(s, s') = rate(s, s')/q   for s != s'
        # P_unif(s, s)  = 1 - E(s)/q      (self-loop)
        unif_trans = {}
        for s in states:
            unif_trans[s] = {}
            E = self.ctmc.exit_rate(s)
            for s2, r in self.ctmc.rates[s].items():
                unif_trans[s][s2] = r / q
            unif_trans[s][s] = unif_trans[s].get(s, 0.0) + (1.0 - E / q)

        # Compute Poisson probabilities for qt
        qt = q * time_bound
        # Truncation point: find K such that remaining tail < epsilon
        K = self._poisson_truncation(qt)

        # Backward iteration: compute gamma_k vectors
        # gamma_K(s) = 1 if s in sat_psi, 0 otherwise
        # gamma_k(s) = 1 if s in sat_psi
        #            = 0 if s not in sat_phi
        #            = sum_s' P_unif(s,s') * gamma_{k+1}(s')  otherwise
        gamma = {s: (1.0 if s in sat_psi else 0.0) for s in states}

        # Poisson weights
        poisson = self._poisson_weights(qt, K)

        # Accumulate: result(s) = sum_k poisson(k) * gamma_k(s)
        result = {s: poisson[K] * gamma[s] for s in states}

        for k in range(K - 1, -1, -1):
            new_gamma = {}
            for s in states:
                if s in sat_psi:
                    new_gamma[s] = 1.0
                elif s not in sat_phi:
                    new_gamma[s] = 0.0
                else:
                    new_gamma[s] = sum(
                        unif_trans[s].get(s2, 0.0) * gamma[s2]
                        for s2 in states
                    )
            gamma = new_gamma
            for s in states:
                result[s] += poisson[k] * gamma[s]

        return result

    def steady_state(self) -> dict[str, float]:
        """CTMC steady-state distribution.

        For an ergodic CTMC, pi * Q = 0, sum(pi) = 1.
        Uses embedded DTMC power iteration adjusted by exit rates.
        """
        edtmc = self.ctmc.embedded_dtmc()
        checker = DTMCModelChecker(edtmc, self.epsilon, self.max_iter)
        embedded_ss = checker.steady_state()

        # CTMC steady state: pi(s) proportional to embedded_ss(s) / exit_rate(s)
        raw = {}
        for s in self.ctmc.states:
            E = self.ctmc.exit_rate(s)
            if E > 0:
                raw[s] = embedded_ss[s] / E
            else:
                raw[s] = embedded_ss[s]  # Absorbing state

        total = sum(raw.values())
        if total > 0:
            return {s: v / total for s, v in raw.items()}
        return raw

    def _check_steady_bound(self, formula: PCTLFormula) -> set[str]:
        sat_phi = self.check(formula.sub)
        ss = self.steady_state()
        total = sum(ss[s] for s in sat_phi)
        if DTMCModelChecker._compare(total, formula.comp, formula.bound):
            return set(self.ctmc.states)
        return set()

    def _check_steady_query(self, formula: PCTLFormula) -> dict[str, float]:
        sat_phi = self.check(formula.sub)
        ss = self.steady_state()
        total = sum(ss[s] for s in sat_phi)
        return {s: total for s in self.ctmc.states}

    def _poisson_truncation(self, qt: float) -> int:
        """Find truncation point K for Poisson(qt) such that tail < epsilon."""
        if qt == 0:
            return 0
        # Use right-tail bound: start from mode and extend
        K = max(int(qt + 4 * math.sqrt(qt) + 10), 10)
        # Verify tail is small
        while K < 10000:
            tail = self._poisson_tail(qt, K)
            if tail < self.epsilon:
                return K
            K += 10
        return K

    def _poisson_tail(self, qt: float, K: int) -> float:
        """Compute P(X > K) for X ~ Poisson(qt) using upper incomplete gamma."""
        # Use summation from 0 to K
        total = 0.0
        log_prob = -qt  # log(e^{-qt} * qt^0 / 0!)
        for k in range(K + 1):
            total += math.exp(log_prob)
            log_prob += math.log(qt) - math.log(k + 1)
            if total > 1.0 - self.epsilon:
                return 0.0
        return 1.0 - total

    def _poisson_weights(self, qt: float, K: int) -> list[float]:
        """Compute Poisson(qt) PMF for k = 0, 1, ..., K."""
        if qt == 0:
            weights = [0.0] * (K + 1)
            weights[0] = 1.0
            return weights

        weights = [0.0] * (K + 1)
        log_prob = -qt
        for k in range(K + 1):
            weights[k] = math.exp(log_prob)
            if k < K:
                log_prob += math.log(qt) - math.log(k + 1)
        return weights


# ---------------------------------------------------------------------------
# High-level verification API
# ---------------------------------------------------------------------------

def verify_dtmc(dtmc: DTMC, formula: PCTLFormula,
                epsilon: float = 1e-10) -> set[str] | dict[str, float]:
    """Verify a PCTL formula on a DTMC."""
    checker = DTMCModelChecker(dtmc, epsilon)
    return checker.check(formula)

def verify_ctmc(ctmc: CTMC, formula: PCTLFormula,
                epsilon: float = 1e-10) -> set[str] | dict[str, float]:
    """Verify a CSL formula on a CTMC."""
    checker = CTMCModelChecker(ctmc, epsilon)
    return checker.check(formula)

def transient_analysis(dtmc: DTMC, initial_dist: dict[str, float],
                        steps: int) -> dict[str, float]:
    """Compute transient distribution after k steps.

    pi_{k+1} = pi_k * P
    """
    dist = dict(initial_dist)
    for s in dtmc.states:
        if s not in dist:
            dist[s] = 0.0

    for _ in range(steps):
        new_dist = {s: 0.0 for s in dtmc.states}
        for s in dtmc.states:
            for s2, prob in dtmc.transitions[s].items():
                new_dist[s2] += dist[s] * prob
        dist = new_dist

    return dist

def ctmc_transient(ctmc: CTMC, initial_dist: dict[str, float],
                    time: float, epsilon: float = 1e-10) -> dict[str, float]:
    """Compute transient distribution at time t via uniformization.

    pi(t) = sum_k Poisson(qt, k) * pi_0 * P_unif^k
    """
    states = ctmc.states
    dist = {s: initial_dist.get(s, 0.0) for s in states}

    if time <= 0:
        return dist

    q = ctmc.uniformization_rate()
    if q == 0:
        return dist

    # Build uniformized transitions
    unif_trans = {}
    for s in states:
        unif_trans[s] = {}
        E = ctmc.exit_rate(s)
        for s2, r in ctmc.rates[s].items():
            unif_trans[s][s2] = r / q
        unif_trans[s][s] = unif_trans[s].get(s, 0.0) + (1.0 - E / q)

    qt = q * time
    checker = CTMCModelChecker(ctmc, epsilon)
    K = checker._poisson_truncation(qt)
    poisson = checker._poisson_weights(qt, K)

    # Forward: compute pi_k = pi_0 * P_unif^k
    result = {s: poisson[0] * dist[s] for s in states}
    current = dict(dist)

    for k in range(1, K + 1):
        new_current = {s: 0.0 for s in states}
        for s in states:
            for s2, prob in unif_trans[s].items():
                new_current[s2] += current[s] * prob
        current = new_current
        for s in states:
            result[s] += poisson[k] * current[s]

    return result


# ---------------------------------------------------------------------------
# Model builders (convenience)
# ---------------------------------------------------------------------------

def build_dtmc_from_matrix(states: list[str], matrix: list[list[float]],
                            labels: dict[str, set[str]] | None = None,
                            initial: str | None = None) -> DTMC:
    """Build a DTMC from a transition probability matrix."""
    dtmc = DTMC()
    label_map = labels or {}
    for s in states:
        dtmc.add_state(s, label_map.get(s, set()))
    for i, s in enumerate(states):
        for j, s2 in enumerate(states):
            if matrix[i][j] > 0:
                dtmc.add_transition(s, s2, matrix[i][j])
    if initial:
        dtmc.set_initial(initial)
    return dtmc

def build_ctmc_from_matrix(states: list[str], matrix: list[list[float]],
                            labels: dict[str, set[str]] | None = None,
                            initial: str | None = None) -> CTMC:
    """Build a CTMC from a rate matrix (off-diagonal entries are rates)."""
    ctmc = CTMC()
    label_map = labels or {}
    for s in states:
        ctmc.add_state(s, label_map.get(s, set()))
    for i, s in enumerate(states):
        for j, s2 in enumerate(states):
            if i != j and matrix[i][j] > 0:
                ctmc.add_rate(s, s2, matrix[i][j])
    if initial:
        ctmc.set_initial(initial)
    return ctmc


# ---------------------------------------------------------------------------
# BSCC (Bottom Strongly Connected Components) analysis
# ---------------------------------------------------------------------------

def find_bsccs(dtmc: DTMC) -> list[set[str]]:
    """Find all bottom SCCs (absorbing components) of a DTMC.

    A BSCC is an SCC with no outgoing transitions.
    """
    # Tarjan's SCC algorithm
    index_counter = [0]
    stack = []
    on_stack = set()
    index = {}
    lowlink = {}
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in dtmc.transitions.get(v, {}):
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

    for v in dtmc.states:
        if v not in index:
            strongconnect(v)

    # Filter to bottom SCCs: no transition leaves the SCC
    bsccs = []
    for scc in sccs:
        is_bottom = True
        for s in scc:
            for s2 in dtmc.transitions.get(s, {}):
                if s2 not in scc:
                    is_bottom = False
                    break
            if not is_bottom:
                break
        if is_bottom:
            bsccs.append(scc)

    return bsccs


def bscc_steady_state(dtmc: DTMC) -> dict[str, float]:
    """Compute steady-state by analyzing BSCCs.

    More precise than power iteration for multi-chain DTMCs.
    Each BSCC has its own stationary distribution.
    The global steady-state depends on the probability of reaching each BSCC.
    """
    bsccs = find_bsccs(dtmc)
    if not bsccs:
        return {s: 0.0 for s in dtmc.states}

    # For single-BSCC chains, just do power iteration within the BSCC
    if len(bsccs) == 1 and len(bsccs[0]) == len(dtmc.states):
        checker = DTMCModelChecker(dtmc)
        return checker.steady_state()

    # Compute reaching probabilities for each BSCC from initial state
    # and local steady-state within each BSCC
    result = {s: 0.0 for s in dtmc.states}

    for bscc in bsccs:
        # Build sub-DTMC for BSCC
        sub = DTMC()
        for s in bscc:
            sub.add_state(s, dtmc.labels[s].copy())
        for s in bscc:
            for s2, p in dtmc.transitions[s].items():
                if s2 in bscc:
                    sub.add_transition(s, s2, p)

        # Normalize (BSCC transitions sum to 1 within BSCC)
        for s in sub.states:
            total = sum(sub.transitions[s].values())
            if total > 0 and abs(total - 1.0) > 1e-12:
                for s2 in sub.transitions[s]:
                    sub.transitions[s][s2] /= total

        checker = DTMCModelChecker(sub)
        local_ss = checker.steady_state()

        # Reaching probability from each state
        reach_prob = _bscc_reach_prob(dtmc, bscc)

        for s in bscc:
            for s0 in dtmc.states:
                result[s] += reach_prob.get(s0, 0.0) * local_ss.get(s, 0.0) / len(dtmc.states)

    # If we have an initial state, use that
    if dtmc.initial:
        result = {s: 0.0 for s in dtmc.states}
        for bscc in bsccs:
            sub = DTMC()
            for s in bscc:
                sub.add_state(s, dtmc.labels[s].copy())
            for s in bscc:
                for s2, p in dtmc.transitions[s].items():
                    if s2 in bscc:
                        sub.add_transition(s, s2, p)
            for s in sub.states:
                total = sum(sub.transitions[s].values())
                if total > 0 and abs(total - 1.0) > 1e-12:
                    for s2 in sub.transitions[s]:
                        sub.transitions[s][s2] /= total

            checker = DTMCModelChecker(sub)
            local_ss = checker.steady_state()
            reach = _bscc_reach_prob(dtmc, bscc)
            rp = reach.get(dtmc.initial, 0.0)
            for s in bscc:
                result[s] += rp * local_ss.get(s, 0.0)

    return result


def _bscc_reach_prob(dtmc: DTMC, bscc: set[str]) -> dict[str, float]:
    """Probability of reaching a BSCC from each state."""
    # States in the BSCC reach it with probability 1
    # Value iteration for others
    probs = {s: 0.0 for s in dtmc.states}
    for s in bscc:
        probs[s] = 1.0

    non_bscc = [s for s in dtmc.states if s not in bscc]
    for _ in range(10000):
        max_diff = 0.0
        for s in non_bscc:
            new_p = sum(p * probs[s2] for s2, p in dtmc.transitions[s].items())
            diff = abs(new_p - probs[s])
            if diff > max_diff:
                max_diff = diff
            probs[s] = new_p
        if max_diff < 1e-10:
            break

    return probs
