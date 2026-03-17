"""
V182: Probabilistic Model Checking
PRISM-style verification of DTMCs and MDPs.

Supports:
- DTMC: Discrete-Time Markov Chains (purely probabilistic)
- MDP: Markov Decision Processes (nondeterministic + probabilistic)
- Reachability probability computation (P=? [F target])
- Expected reward computation (R=? [F target])
- Min/max optimization for MDPs
- Steady-state (long-run) probability analysis for DTMCs
- PCTL model checking (probabilistic CTL)
- Bisimulation quotient for state space reduction
"""

from fractions import Fraction
from collections import defaultdict, deque


# ============================================================
# Core data structures
# ============================================================

class State:
    """A state in a probabilistic model."""
    __slots__ = ('name', 'labels', 'reward')

    def __init__(self, name, labels=None, reward=0):
        self.name = name
        self.labels = set(labels) if labels else set()
        self.reward = Fraction(reward)

    def __repr__(self):
        return f"State({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, State) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Distribution:
    """A probability distribution over successor states.
    Maps state_name -> probability (as Fraction for exactness).
    """
    __slots__ = ('_probs', 'action')

    def __init__(self, probs, action=None):
        self._probs = {}
        for s, p in probs.items():
            p = Fraction(p)
            if p < 0:
                raise ValueError(f"Negative probability: {p}")
            if p > 0:
                self._probs[s] = p
        total = sum(self._probs.values())
        if abs(total - 1) > Fraction(1, 10000):
            raise ValueError(f"Probabilities sum to {total}, not 1")
        self.action = action

    def support(self):
        return set(self._probs.keys())

    def prob(self, state_name):
        return self._probs.get(state_name, Fraction(0))

    def items(self):
        return self._probs.items()

    def __repr__(self):
        act = f", action={self.action!r}" if self.action else ""
        return f"Distribution({dict(self._probs)}{act})"


# ============================================================
# DTMC
# ============================================================

class DTMC:
    """Discrete-Time Markov Chain.
    Each state has exactly one probability distribution over successors.
    """

    def __init__(self):
        self.states = {}          # name -> State
        self.transitions = {}     # name -> Distribution
        self.initial = None       # name of initial state

    def add_state(self, name, labels=None, reward=0):
        s = State(name, labels, reward)
        self.states[name] = s
        return s

    def set_initial(self, name):
        if name not in self.states:
            raise ValueError(f"Unknown state: {name}")
        self.initial = name

    def add_transition(self, src, probs):
        """Add transition from src with given probability distribution.
        probs: dict mapping state_name -> probability
        """
        if src not in self.states:
            raise ValueError(f"Unknown state: {src}")
        for dst in probs:
            if dst not in self.states:
                raise ValueError(f"Unknown state: {dst}")
        self.transitions[src] = Distribution(probs)

    def successors(self, name):
        """Return set of successor state names."""
        if name in self.transitions:
            return self.transitions[name].support()
        return set()

    def predecessors(self, name):
        """Return set of predecessor state names."""
        preds = set()
        for src, dist in self.transitions.items():
            if name in dist.support():
                preds.add(src)
        return preds

    def states_with_label(self, label):
        """Return set of state names that have the given label."""
        return {n for n, s in self.states.items() if label in s.labels}

    def is_absorbing(self, name):
        """Check if state is absorbing (self-loop with prob 1)."""
        if name not in self.transitions:
            return True
        dist = self.transitions[name]
        return dist.support() == {name} and dist.prob(name) == 1


# ============================================================
# MDP
# ============================================================

class MDP:
    """Markov Decision Process.
    Each state has one or more actions, each leading to a distribution.
    """

    def __init__(self):
        self.states = {}          # name -> State
        self.transitions = {}     # name -> list of Distribution (each with .action)
        self.initial = None

    def add_state(self, name, labels=None, reward=0):
        s = State(name, labels, reward)
        self.states[name] = s
        if name not in self.transitions:
            self.transitions[name] = []
        return s

    def set_initial(self, name):
        if name not in self.states:
            raise ValueError(f"Unknown state: {name}")
        self.initial = name

    def add_transition(self, src, probs, action=None):
        """Add a nondeterministic choice from src with given distribution."""
        if src not in self.states:
            raise ValueError(f"Unknown state: {src}")
        for dst in probs:
            if dst not in self.states:
                raise ValueError(f"Unknown state: {dst}")
        self.transitions[src].append(Distribution(probs, action=action))

    def actions(self, name):
        """Return list of available distributions at a state."""
        return self.transitions.get(name, [])

    def states_with_label(self, label):
        return {n for n, s in self.states.items() if label in s.labels}

    def is_absorbing(self, name):
        acts = self.actions(name)
        if not acts:
            return True
        return all(d.support() == {name} and d.prob(name) == 1 for d in acts)


# ============================================================
# Analysis: Reachability (BFS/DFS utilities)
# ============================================================

def _reachable_from(model, start_set, forward=True):
    """BFS reachability from start_set."""
    visited = set()
    queue = deque(start_set)
    for s in start_set:
        visited.add(s)
    while queue:
        s = queue.popleft()
        if forward:
            if isinstance(model, DTMC):
                nexts = model.successors(s)
            else:
                nexts = set()
                for d in model.actions(s):
                    nexts |= d.support()
        else:
            # backward
            if isinstance(model, DTMC):
                nexts = model.predecessors(s)
            else:
                nexts = set()
                for src in model.states:
                    for d in model.actions(src):
                        if s in d.support():
                            nexts.add(src)
        for n in nexts:
            if n not in visited:
                visited.add(n)
                queue.append(n)
    return visited


def _prob0(model, target_set):
    """Compute states with probability 0 of reaching target.
    These are states from which target is unreachable.
    """
    # States that CAN reach target (backward reachability)
    can_reach = _reachable_from(model, target_set, forward=False)
    # prob0 = all states NOT in can_reach
    return set(model.states.keys()) - can_reach


def _prob1_dtmc(dtmc, target_set):
    """Compute states with probability 1 of reaching target in a DTMC.
    A state has prob 1 if it's not prob 0 and ALL paths lead to target.
    Equivalently: remove prob0 states, then backward reachability from target.
    """
    prob0 = _prob0(dtmc, target_set)
    # In the sub-DTMC excluding prob0 states, find states that reach target
    # A state has prob1 iff it can reach target without going through prob0
    # Iterative: prob1 = greatest fixpoint of: s in prob1 if s in target OR
    #   all successors of s are in prob1
    remaining = set(dtmc.states.keys()) - prob0
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for s in remaining:
            if s in target_set:
                continue
            succs = dtmc.successors(s)
            if not succs:
                # Dead end, not in target -> prob0 actually
                to_remove.add(s)
                continue
            if not succs.issubset(remaining):
                to_remove.add(s)
        if to_remove:
            remaining -= to_remove
            changed = True
    return remaining


def _prob1_mdp_max(mdp, target_set):
    """Prob1 for maximizing MDP: states where there EXISTS a scheduler
    achieving probability 1.
    Iterative: s in Prob1 if s in target OR exists action where all
    successors are in Prob1.
    """
    prob0 = _prob0(mdp, target_set)
    remaining = set(mdp.states.keys()) - prob0
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for s in remaining:
            if s in target_set:
                continue
            acts = mdp.actions(s)
            if not acts:
                to_remove.add(s)
                continue
            # Need at least one action with all successors in remaining
            ok = False
            for d in acts:
                if d.support().issubset(remaining):
                    ok = True
                    break
            if not ok:
                to_remove.add(s)
        if to_remove:
            remaining -= to_remove
            changed = True
    return remaining


def _prob1_mdp_min(mdp, target_set):
    """Prob1 for minimizing MDP: states where ALL schedulers achieve prob 1.
    s in Prob1 if s in target OR for ALL actions, all successors in Prob1.
    """
    prob0 = _prob0(mdp, target_set)
    remaining = set(mdp.states.keys()) - prob0
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for s in remaining:
            if s in target_set:
                continue
            acts = mdp.actions(s)
            if not acts:
                to_remove.add(s)
                continue
            # ALL actions must have all successors in remaining
            ok = True
            for d in acts:
                if not d.support().issubset(remaining):
                    ok = False
                    break
            if not ok:
                to_remove.add(s)
        if to_remove:
            remaining -= to_remove
            changed = True
    return remaining


# ============================================================
# Gaussian elimination solver (exact Fraction arithmetic)
# ============================================================

def _solve_linear_system(dtmc, unknown_list, target_set, prob0_set, mode='reach'):
    """Solve a linear system for unknown states using Gaussian elimination.

    For mode='reach': x[s] = sum_t P(s,t)*x[t], where x[target]=1, x[prob0]=0
    For mode='reward': x[s] = reward(s) + sum_t P(s,t)*x[t], where x[target]=0, x[prob0]=inf

    Returns dict: state_name -> Fraction value for unknown states.
    """
    n = len(unknown_list)
    if n == 0:
        return {}

    idx = {s: i for i, s in enumerate(unknown_list)}

    # Build system: for each unknown s:
    # x[s] - sum_{t in unknown} P(s,t)*x[t] = sum_{t in target} P(s,t)*val(t) + reward(s)
    # => coefficient matrix A, right-hand side b
    # A[i][j] = (1 if i==j else 0) - P(s_i, s_j) for s_j in unknown
    # b[i] = sum_{t in target} P(s_i, t) * 1  (for reach) or reward(s_i) (for reward)

    A = [[Fraction(0)] * n for _ in range(n)]
    b = [Fraction(0)] * n

    for i, s in enumerate(unknown_list):
        A[i][i] = Fraction(1)
        dist = dtmc.transitions.get(s)
        if dist is None:
            continue
        for t, p in dist.items():
            if t in idx:
                A[i][idx[t]] -= p
            elif mode == 'reach' and t in target_set:
                b[i] += p  # * 1
            elif mode == 'reward' and t in target_set:
                pass  # target has reward 0
            # prob0 states contribute 0 (for reach) or inf (for reward, but we exclude those)
        if mode == 'reward':
            b[i] += dtmc.states[s].reward

    # Gaussian elimination with partial pivoting
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(col, n):
            if A[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            b[col], b[pivot] = b[pivot], b[col]

        # Eliminate
        for row in range(n):
            if row == col or A[row][col] == 0:
                continue
            factor = A[row][col] / A[col][col]
            for j in range(col, n):
                A[row][j] -= factor * A[col][j]
            b[row] -= factor * b[col]

    # Back-substitute
    result = {}
    for i in range(n):
        if A[i][i] != 0:
            result[unknown_list[i]] = b[i] / A[i][i]
        else:
            result[unknown_list[i]] = Fraction(0)

    return result


# ============================================================
# DTMC: Reachability probability
# ============================================================

def dtmc_reachability_probability(dtmc, target_label):
    """Compute probability of reaching states labeled with target_label
    from the initial state.

    Solves the linear system:
      x[s] = 1                          if s in target
      x[s] = 0                          if s in prob0
      x[s] = 1                          if s in prob1
      x[s] = sum_{t} P(s,t) * x[t]     otherwise (Gaussian elimination)

    Returns dict: state_name -> probability (as Fraction).
    """
    target_set = dtmc.states_with_label(target_label)
    prob0 = _prob0(dtmc, target_set)
    prob1 = _prob1_dtmc(dtmc, target_set)

    result = {}
    for s in dtmc.states:
        if s in target_set:
            result[s] = Fraction(1)
        elif s in prob0:
            result[s] = Fraction(0)
        elif s in prob1:
            result[s] = Fraction(1)
        else:
            result[s] = Fraction(0)  # will be solved

    # Solve for unknown states using Gaussian elimination
    unknown = sorted(set(dtmc.states.keys()) - target_set - prob0 - prob1)
    if unknown:
        solved = _solve_linear_system(dtmc, unknown, target_set | prob1, prob0, mode='reach')
        result.update(solved)

    return result


# ============================================================
# DTMC: Expected reward
# ============================================================

def dtmc_expected_reward(dtmc, target_label):
    """Compute expected cumulative reward to reach target.

    Solves:
      r[s] = 0                               if s in target
      r[s] = infinity                         if s in prob0 (unreachable)
      r[s] = reward(s) + sum_t P(s,t)*r[t]   otherwise

    Uses Gaussian elimination for exact results.
    Returns dict: state_name -> expected reward (Fraction or 'inf').
    """
    target_set = dtmc.states_with_label(target_label)
    prob0 = _prob0(dtmc, target_set)
    prob1 = _prob1_dtmc(dtmc, target_set)

    result = {}
    for s in dtmc.states:
        if s in target_set:
            result[s] = Fraction(0)
        elif s not in prob1:
            result[s] = float('inf')

    # Solve for prob1 - target states using Gaussian elimination
    unknown = sorted(prob1 - target_set)
    if unknown:
        solved = _solve_linear_system(dtmc, unknown, target_set, prob0, mode='reward')
        result.update(solved)

    return result


# ============================================================
# DTMC: Steady-state (stationary) distribution
# ============================================================

def dtmc_steady_state(dtmc):
    """Compute steady-state distribution for an ergodic DTMC.

    Solves pi * P = pi, sum(pi) = 1 via power iteration.
    Returns dict: state_name -> long-run probability.
    """
    names = sorted(dtmc.states.keys())
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}

    # Initialize uniform
    pi = [Fraction(1, n)] * n

    for _ in range(2000):
        new_pi = [Fraction(0)] * n
        for s in names:
            i = idx[s]
            dist = dtmc.transitions.get(s)
            if dist is None:
                # Absorbing: stays in s
                new_pi[i] += pi[i]
            else:
                for t, p in dist.items():
                    j = idx[t]
                    new_pi[j] += pi[i] * p

        max_diff = max(abs(new_pi[i] - pi[i]) for i in range(n))
        pi = new_pi
        if max_diff < Fraction(1, 10**12):
            break

    return {names[i]: pi[i] for i in range(n)}


# ============================================================
# MDP: Reachability probability (min/max)
# ============================================================

def mdp_reachability_probability(mdp, target_label, minimize=False):
    """Compute min/max reachability probability for an MDP.

    For maximize: at each state, pick action maximizing probability.
    For minimize: at each state, pick action minimizing probability.

    Uses value iteration.
    Returns (values, scheduler) where:
      values: state_name -> probability
      scheduler: state_name -> chosen Distribution
    """
    target_set = mdp.states_with_label(target_label)
    prob0 = _prob0(mdp, target_set)

    if minimize:
        prob1 = _prob1_mdp_min(mdp, target_set)
    else:
        prob1 = _prob1_mdp_max(mdp, target_set)

    result = {}
    for s in mdp.states:
        if s in target_set:
            result[s] = Fraction(1)
        elif s in prob0:
            result[s] = Fraction(0)
        elif s in prob1:
            result[s] = Fraction(1)
        else:
            result[s] = Fraction(1, 2)

    scheduler = {}
    unknown = set(mdp.states.keys()) - target_set - prob0 - prob1

    for _ in range(1000):
        max_diff = Fraction(0)
        for s in unknown:
            acts = mdp.actions(s)
            if not acts:
                result[s] = Fraction(0)
                continue
            best_val = None
            best_act = None
            for d in acts:
                val = Fraction(0)
                for t, p in d.items():
                    val += p * result[t]
                if best_val is None:
                    best_val = val
                    best_act = d
                elif minimize and val < best_val:
                    best_val = val
                    best_act = d
                elif not minimize and val > best_val:
                    best_val = val
                    best_act = d
            diff = abs(best_val - result[s])
            if diff > max_diff:
                max_diff = diff
            result[s] = best_val
            scheduler[s] = best_act
        if max_diff < Fraction(1, 10**12):
            break

    # Also record scheduler for prob1 states
    for s in prob1 - target_set:
        acts = mdp.actions(s)
        if acts:
            scheduler[s] = acts[0]

    return result, scheduler


# ============================================================
# MDP: Expected reward (min/max)
# ============================================================

def mdp_expected_reward(mdp, target_label, minimize=True):
    """Compute min/max expected reward to reach target in an MDP.

    For states that can't reach target under any/all schedulers, reward is inf.
    Uses strategy iteration: pick best action, solve linear system, repeat.

    Returns (values, scheduler).
    """
    target_set = mdp.states_with_label(target_label)
    prob0 = _prob0(mdp, target_set)

    # For expected reward, use prob1_max: states where SOME scheduler achieves prob 1.
    # Even a minimizer needs to reach the target -- it picks the cheapest way.
    prob1 = _prob1_mdp_max(mdp, target_set)

    result = {}
    for s in mdp.states:
        if s in target_set:
            result[s] = Fraction(0)
        elif s not in prob1:
            result[s] = float('inf')
        else:
            result[s] = Fraction(0)

    scheduler = {}
    unknown = sorted(prob1 - target_set)

    if not unknown:
        return result, scheduler

    # Strategy iteration: pick best actions, solve, repeat
    # Initialize scheduler: pick first action whose successors are in prob1 | target
    valid_targets = prob1 | target_set
    for s in unknown:
        acts = mdp.actions(s)
        for d in acts:
            if d.support().issubset(valid_targets):
                scheduler[s] = d
                break
        else:
            if acts:
                scheduler[s] = acts[0]

    for _ in range(100):
        # Solve linear system given current scheduler
        solved = _solve_mdp_linear(mdp, unknown, target_set, prob0 | (set(mdp.states.keys()) - prob1), scheduler)
        result.update(solved)

        # Improve scheduler
        changed = False
        for s in unknown:
            acts = mdp.actions(s)
            if len(acts) <= 1:
                continue
            best_val = None
            best_act = None
            for d in acts:
                val = mdp.states[s].reward
                all_finite = True
                for t, p in d.items():
                    r_t = result.get(t, Fraction(0))
                    if r_t == float('inf'):
                        all_finite = False
                        break
                    val += p * r_t
                if not all_finite:
                    if not minimize:
                        val = float('inf')
                    else:
                        continue
                if best_val is None:
                    best_val = val
                    best_act = d
                elif minimize and val < best_val:
                    best_val = val
                    best_act = d
                elif not minimize and val > best_val:
                    best_val = val
                    best_act = d
            if best_act is not None and best_act is not scheduler.get(s):
                scheduler[s] = best_act
                changed = True

        if not changed:
            break

    return result, scheduler


def _solve_mdp_linear(mdp, unknown_list, target_set, inf_set, scheduler):
    """Solve the linear system for MDP given a fixed scheduler."""
    n = len(unknown_list)
    if n == 0:
        return {}
    idx = {s: i for i, s in enumerate(unknown_list)}

    A = [[Fraction(0)] * n for _ in range(n)]
    b = [Fraction(0)] * n

    for i, s in enumerate(unknown_list):
        A[i][i] = Fraction(1)
        d = scheduler.get(s)
        if d is None:
            continue
        for t, p in d.items():
            if t in idx:
                A[i][idx[t]] -= p
            # target contributes 0 to reward
        b[i] = mdp.states[s].reward

    # Gaussian elimination
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if A[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            b[col], b[pivot] = b[pivot], b[col]
        for row in range(n):
            if row == col or A[row][col] == 0:
                continue
            factor = A[row][col] / A[col][col]
            for j in range(col, n):
                A[row][j] -= factor * A[col][j]
            b[row] -= factor * b[col]

    result = {}
    for i in range(n):
        if A[i][i] != 0:
            result[unknown_list[i]] = b[i] / A[i][i]
        else:
            result[unknown_list[i]] = Fraction(0)
    return result


# ============================================================
# PCTL Model Checking
# ============================================================

class PCTLFormula:
    """Base class for PCTL formulas."""
    pass

class Atomic(PCTLFormula):
    """Atomic proposition (label)."""
    def __init__(self, label):
        self.label = label
    def __repr__(self):
        return f"Atomic({self.label!r})"

class Not(PCTLFormula):
    def __init__(self, sub):
        self.sub = sub
    def __repr__(self):
        return f"Not({self.sub})"

class And(PCTLFormula):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"And({self.left}, {self.right})"

class Or(PCTLFormula):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Or({self.left}, {self.right})"

class ProbOp(PCTLFormula):
    """P~p [path_formula].
    comp: '<', '<=', '>', '>='
    bound: Fraction
    path_formula: Finally or Until or Bounded
    """
    def __init__(self, comp, bound, path_formula):
        self.comp = comp
        self.bound = Fraction(bound)
        self.path_formula = path_formula
    def __repr__(self):
        return f"P{self.comp}{self.bound}[{self.path_formula}]"

class Finally(PCTLFormula):
    """F phi (eventually phi)."""
    def __init__(self, sub):
        self.sub = sub
    def __repr__(self):
        return f"F({self.sub})"

class Until(PCTLFormula):
    """phi1 U phi2."""
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Until({self.left}, {self.right})"

class BoundedFinally(PCTLFormula):
    """F<=k phi (eventually within k steps)."""
    def __init__(self, sub, bound):
        self.sub = sub
        self.bound = bound
    def __repr__(self):
        return f"F<={self.bound}({self.sub})"

class ExpRewardOp(PCTLFormula):
    """R~r [F phi] -- expected reward to reach phi."""
    def __init__(self, comp, bound, target):
        self.comp = comp
        self.bound = Fraction(bound)
        self.target = target
    def __repr__(self):
        return f"R{self.comp}{self.bound}[F {self.target}]"


def _compare(val, comp, bound):
    """Evaluate val comp bound."""
    if val == float('inf'):
        return comp in ('>', '>=')
    val = Fraction(val)
    bound = Fraction(bound)
    if comp == '<':
        return val < bound
    elif comp == '<=':
        return val <= bound
    elif comp == '>':
        return val > bound
    elif comp == '>=':
        return val >= bound
    return False


def pctl_check_dtmc(dtmc, formula):
    """Check PCTL formula on a DTMC.
    Returns the set of state names satisfying the formula.
    """
    if isinstance(formula, Atomic):
        return dtmc.states_with_label(formula.label)

    if isinstance(formula, Not):
        sub_sat = pctl_check_dtmc(dtmc, formula.sub)
        return set(dtmc.states.keys()) - sub_sat

    if isinstance(formula, And):
        left_sat = pctl_check_dtmc(dtmc, formula.left)
        right_sat = pctl_check_dtmc(dtmc, formula.right)
        return left_sat & right_sat

    if isinstance(formula, Or):
        left_sat = pctl_check_dtmc(dtmc, formula.left)
        right_sat = pctl_check_dtmc(dtmc, formula.right)
        return left_sat | right_sat

    if isinstance(formula, ProbOp):
        pf = formula.path_formula
        if isinstance(pf, Finally):
            # P~p [F phi]
            target_sat = pctl_check_dtmc(dtmc, pf.sub)
            # Temporarily label target states
            probs = _dtmc_reach_prob_set(dtmc, target_sat)
            return {s for s in dtmc.states if _compare(probs[s], formula.comp, formula.bound)}

        if isinstance(pf, BoundedFinally):
            # P~p [F<=k phi]
            target_sat = pctl_check_dtmc(dtmc, pf.sub)
            probs = _dtmc_bounded_reach(dtmc, target_sat, pf.bound)
            return {s for s in dtmc.states if _compare(probs[s], formula.comp, formula.bound)}

        if isinstance(pf, Until):
            # P~p [phi1 U phi2]
            phi1_sat = pctl_check_dtmc(dtmc, pf.left)
            phi2_sat = pctl_check_dtmc(dtmc, pf.right)
            probs = _dtmc_until_prob(dtmc, phi1_sat, phi2_sat)
            return {s for s in dtmc.states if _compare(probs[s], formula.comp, formula.bound)}

    if isinstance(formula, ExpRewardOp):
        target_sat = pctl_check_dtmc(dtmc, formula.target)
        rewards = _dtmc_exp_reward_set(dtmc, target_sat)
        return {s for s in dtmc.states if _compare(rewards[s], formula.comp, formula.bound)}

    raise ValueError(f"Unsupported formula: {formula}")


def _dtmc_reach_prob_set(dtmc, target_set):
    """Compute reachability probability given explicit target set (Gaussian elimination)."""
    prob0 = set(dtmc.states.keys()) - _reachable_from(dtmc, target_set, forward=False)
    prob1 = _prob1_dtmc(dtmc, target_set)

    result = {}
    for s in dtmc.states:
        if s in target_set:
            result[s] = Fraction(1)
        elif s in prob0:
            result[s] = Fraction(0)
        elif s in prob1:
            result[s] = Fraction(1)
        else:
            result[s] = Fraction(0)

    unknown = sorted(set(dtmc.states.keys()) - target_set - prob0 - prob1)
    if unknown:
        solved = _solve_linear_system(dtmc, unknown, target_set | prob1, prob0, mode='reach')
        result.update(solved)
    return result


def _dtmc_exp_reward_set(dtmc, target_set):
    """Expected reward given explicit target set (Gaussian elimination)."""
    prob0 = set(dtmc.states.keys()) - _reachable_from(dtmc, target_set, forward=False)
    prob1 = _prob1_dtmc(dtmc, target_set)

    result = {}
    for s in dtmc.states:
        if s in target_set:
            result[s] = Fraction(0)
        elif s not in prob1:
            result[s] = float('inf')
        else:
            result[s] = Fraction(0)

    unknown = sorted(prob1 - target_set)
    if unknown:
        solved = _solve_linear_system(dtmc, unknown, target_set, prob0 | (set(dtmc.states.keys()) - prob1), mode='reward')
        result.update(solved)
    return result


def _dtmc_bounded_reach(dtmc, target_set, steps):
    """Bounded reachability: probability of reaching target within k steps."""
    names = list(dtmc.states.keys())

    # prev[s] = prob of reaching target from s in <= i steps
    prev = {}
    for s in names:
        prev[s] = Fraction(1) if s in target_set else Fraction(0)

    for _ in range(steps):
        curr = {}
        for s in names:
            if s in target_set:
                curr[s] = Fraction(1)
                continue
            dist = dtmc.transitions.get(s)
            if dist is None:
                curr[s] = Fraction(0)
                continue
            val = Fraction(0)
            for t, p in dist.items():
                val += p * prev[t]
            curr[s] = val
        prev = curr

    return prev


def _dtmc_until_prob(dtmc, phi1_set, phi2_set):
    """Compute P[phi1 U phi2] for a DTMC using Gaussian elimination.
    Probability of reaching phi2 while staying in phi1.
    """
    all_states = set(dtmc.states.keys())
    valid = phi1_set | phi2_set

    # Backward reach from phi2 within valid
    can_reach = set(phi2_set)
    queue = deque(phi2_set)
    while queue:
        s = queue.popleft()
        for src in dtmc.states:
            if src in valid and src not in can_reach:
                dist = dtmc.transitions.get(src)
                if dist and s in dist.support():
                    can_reach.add(src)
                    queue.append(src)

    result = {}
    for s in all_states:
        if s in phi2_set:
            result[s] = Fraction(1)
        elif s not in can_reach or s not in valid:
            result[s] = Fraction(0)

    # Unknown: states in phi1, can reach phi2, not in phi2
    unknown = sorted(s for s in all_states if s in can_reach and s in phi1_set and s not in phi2_set)
    if unknown:
        # Treat phi2 as target (value 1), everything outside valid+can_reach as prob0 (value 0)
        target_equiv = phi2_set
        prob0_equiv = all_states - can_reach - phi2_set
        solved = _solve_linear_system(dtmc, unknown, target_equiv, prob0_equiv, mode='reach')
        result.update(solved)

    return result


# ============================================================
# Probabilistic Bisimulation (state space reduction)
# ============================================================

def dtmc_bisimulation_quotient(dtmc):
    """Compute probabilistic bisimulation quotient of a DTMC.

    Two states are bisimilar if:
    1. They have the same labels
    2. For every equivalence class C, they have the same total transition
       probability to C.

    Returns (quotient_dtmc, partition) where partition maps state -> block_id.
    """
    # Initial partition by label set
    label_to_block = {}
    partition = {}
    for name, state in dtmc.states.items():
        key = frozenset(state.labels)
        if key not in label_to_block:
            label_to_block[key] = len(label_to_block)
        partition[name] = label_to_block[key]

    # Refine partition
    for _ in range(len(dtmc.states)):
        new_partition = {}
        sig_to_block = {}
        block_id = 0

        for name in dtmc.states:
            # Compute signature: (current_block, {target_block: total_prob})
            dist = dtmc.transitions.get(name)
            if dist is None:
                block_probs = {}
            else:
                block_probs = defaultdict(Fraction)
                for t, p in dist.items():
                    block_probs[partition[t]] += p
                block_probs = dict(block_probs)

            sig = (partition[name], tuple(sorted(block_probs.items())))
            if sig not in sig_to_block:
                sig_to_block[sig] = block_id
                block_id += 1
            new_partition[name] = sig_to_block[sig]

        if new_partition == partition:
            break
        partition = new_partition

    # Build quotient DTMC
    quotient = DTMC()
    block_states = defaultdict(list)
    for name, block in partition.items():
        block_states[block].append(name)

    for block, members in block_states.items():
        rep = members[0]
        labels = dtmc.states[rep].labels
        reward = dtmc.states[rep].reward
        quotient.add_state(f"B{block}", labels=labels, reward=reward)

    if dtmc.initial is not None:
        quotient.set_initial(f"B{partition[dtmc.initial]}")

    for block, members in block_states.items():
        rep = members[0]
        dist = dtmc.transitions.get(rep)
        if dist is not None:
            block_probs = defaultdict(Fraction)
            for t, p in dist.items():
                block_probs[f"B{partition[t]}"] += p
            quotient.add_transition(f"B{block}", dict(block_probs))

    return quotient, partition


# ============================================================
# Transient analysis (step-bounded)
# ============================================================

def dtmc_transient_probs(dtmc, steps, start=None):
    """Compute probability distribution after k steps.
    Returns dict: state_name -> probability of being in that state.
    """
    if start is None:
        start = dtmc.initial
    if start is None:
        raise ValueError("No initial state set")

    dist = {s: Fraction(0) for s in dtmc.states}
    dist[start] = Fraction(1)

    for _ in range(steps):
        new_dist = {s: Fraction(0) for s in dtmc.states}
        for s in dtmc.states:
            if dist[s] == 0:
                continue
            trans = dtmc.transitions.get(s)
            if trans is None:
                new_dist[s] += dist[s]
            else:
                for t, p in trans.items():
                    new_dist[t] += dist[s] * p
        dist = new_dist

    return dist


# ============================================================
# Convenience builders
# ============================================================

def build_dtmc(states, transitions, initial, labels=None, rewards=None):
    """Quick builder for DTMCs.

    states: list of state names
    transitions: dict {src: {dst: prob, ...}, ...}
    initial: name of initial state
    labels: dict {state_name: [label, ...]}
    rewards: dict {state_name: reward}
    """
    labels = labels or {}
    rewards = rewards or {}
    dtmc = DTMC()
    for s in states:
        dtmc.add_state(s, labels=labels.get(s, []), reward=rewards.get(s, 0))
    dtmc.set_initial(initial)
    for src, probs in transitions.items():
        dtmc.add_transition(src, probs)
    return dtmc


def build_mdp(states, transitions, initial, labels=None, rewards=None):
    """Quick builder for MDPs.

    states: list of state names
    transitions: dict {src: [{dst: prob, ...}, ...] or [({dst: prob}, action), ...]}
    initial: name of initial state
    labels: dict {state_name: [label, ...]}
    rewards: dict {state_name: reward}
    """
    labels = labels or {}
    rewards = rewards or {}
    mdp = MDP()
    for s in states:
        mdp.add_state(s, labels=labels.get(s, []), reward=rewards.get(s, 0))
    mdp.set_initial(initial)
    for src, choices in transitions.items():
        for choice in choices:
            if isinstance(choice, tuple) and len(choice) == 2:
                probs, action = choice
                mdp.add_transition(src, probs, action=action)
            elif isinstance(choice, tuple) and len(choice) == 1:
                mdp.add_transition(src, choice[0])
            else:
                mdp.add_transition(src, choice)
    return mdp


# ============================================================
# Path simulation (Monte Carlo)
# ============================================================

def dtmc_simulate_path(dtmc, max_steps=1000, rng=None):
    """Simulate a single path through a DTMC.
    Returns list of state names visited.
    Uses deterministic Fraction-based selection if no rng provided.
    """
    import random
    if rng is None:
        rng = random.Random()

    path = [dtmc.initial]
    current = dtmc.initial
    for _ in range(max_steps):
        if dtmc.is_absorbing(current):
            break
        dist = dtmc.transitions.get(current)
        if dist is None:
            break
        # Sample from distribution
        r = Fraction(rng.randint(0, 10**9), 10**9)
        cumulative = Fraction(0)
        chosen = None
        for t, p in dist.items():
            cumulative += p
            if r < cumulative:
                chosen = t
                break
        if chosen is None:
            chosen = t  # last one (rounding)
        path.append(chosen)
        current = chosen

    return path


def dtmc_estimate_reachability(dtmc, target_label, num_samples=10000, max_steps=1000, rng=None):
    """Monte Carlo estimation of reachability probability."""
    import random
    if rng is None:
        rng = random.Random(42)

    target_set = dtmc.states_with_label(target_label)
    hits = 0
    for _ in range(num_samples):
        path = dtmc_simulate_path(dtmc, max_steps=max_steps, rng=rng)
        if any(s in target_set for s in path):
            hits += 1
    return Fraction(hits, num_samples)
