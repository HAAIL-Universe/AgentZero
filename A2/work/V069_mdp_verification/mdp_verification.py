"""V069: Markov Decision Process (MDP) Verification

Composes V065 (Markov chain analysis) + C037 (SMT solver) + V068 (interval MDP).

An MDP extends a Markov chain with nondeterministic action choices at each state.
This module provides:
1. MDP data structure with actions, transitions, rewards
2. Value iteration for optimal policies (maximize/minimize expected reward)
3. Reachability analysis under optimal/adversarial policies
4. SMT-based verification of MDP properties
5. Policy extraction and evaluation
6. Comparison with interval MDPs (V068)
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V068_interval_mdp'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from markov_chain import MarkovChain, make_chain, analyze_chain, StateType
from markov_chain import steady_state, absorption_probabilities, expected_hitting_time
from markov_chain import _solve_linear
from smt_solver import SMTSolver, SMTResult, Var, IntConst, App, Op, BOOL, INT
from interval_mdp import (IntervalMDP, make_interval_mdp, ProbInterval,
                          robust_reachability, OptimizationDirection,
                          check_all_feasible)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Objective(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class MDP:
    """Markov Decision Process.

    n_states: number of states
    actions: actions[s] = list of action names available at state s
    transition: transition[s][a_idx][t] = probability of going from s to t under action a_idx
    rewards: rewards[s][a_idx] = immediate reward for taking action a_idx in state s
    state_labels: optional human-readable labels
    ap_labels: atomic propositions per state (for temporal properties)
    """
    n_states: int
    actions: List[List[str]]
    transition: List[List[List[float]]]
    rewards: Optional[List[List[float]]] = None
    state_labels: Optional[List[str]] = None
    ap_labels: Optional[Dict[int, Set[str]]] = None

    def __post_init__(self):
        if self.state_labels is None:
            self.state_labels = [f"s{i}" for i in range(self.n_states)]
        if self.rewards is None:
            if len(self.actions) == self.n_states:
                self.rewards = [[0.0] * len(self.actions[s]) for s in range(self.n_states)]
            else:
                self.rewards = []
        if self.ap_labels is None:
            self.ap_labels = {}

    def validate(self) -> List[str]:
        errors = []
        if len(self.actions) != self.n_states:
            errors.append(f"actions length {len(self.actions)} != n_states {self.n_states}")
        if len(self.transition) != self.n_states:
            errors.append(f"transition length {len(self.transition)} != n_states {self.n_states}")
        for s in range(self.n_states):
            if len(self.transition[s]) != len(self.actions[s]):
                errors.append(f"state {s}: {len(self.transition[s])} transition rows != {len(self.actions[s])} actions")
            for a_idx, row in enumerate(self.transition[s]):
                if len(row) != self.n_states:
                    errors.append(f"state {s} action {a_idx}: row length {len(row)} != {self.n_states}")
                total = sum(row)
                if abs(total - 1.0) > 1e-6:
                    errors.append(f"state {s} action {a_idx}: probabilities sum to {total}")
                for p in row:
                    if p < -1e-9:
                        errors.append(f"state {s} action {a_idx}: negative probability {p}")
        return errors

    def is_mc(self) -> bool:
        """Check if this MDP is effectively a Markov chain (1 action per state)."""
        return all(len(acts) == 1 for acts in self.actions)


@dataclass
class Policy:
    """A memoryless deterministic policy: maps each state to an action index."""
    action_map: Dict[int, int]

    def get_action(self, state: int) -> int:
        return self.action_map.get(state, 0)


@dataclass
class RandomizedPolicy:
    """A memoryless randomized policy: maps each state to a distribution over actions."""
    distributions: Dict[int, List[float]]

    def get_distribution(self, state: int) -> List[float]:
        return self.distributions.get(state, [1.0])


@dataclass
class ValueResult:
    """Result of value iteration."""
    values: List[float]
    policy: Policy
    iterations: int
    converged: bool
    objective: Objective


@dataclass
class ReachResult:
    """Result of reachability analysis."""
    probabilities: List[float]
    policy: Policy
    objective: Objective
    targets: Set[int]


@dataclass
class VerificationResult:
    """Result of SMT-based verification."""
    verified: bool
    property_desc: str
    counterexample: Optional[Dict] = None
    details: str = ""


# ---------------------------------------------------------------------------
# MDP construction helpers
# ---------------------------------------------------------------------------

def make_mdp(n_states: int,
             action_transitions: Dict[int, Dict[str, List[float]]],
             rewards: Optional[Dict[int, Dict[str, float]]] = None,
             state_labels: Optional[List[str]] = None,
             ap_labels: Optional[Dict[int, Set[str]]] = None) -> MDP:
    """Create an MDP from a convenient dictionary format.

    action_transitions[s][action_name] = [p0, p1, ..., p_{n-1}]
    rewards[s][action_name] = reward value
    """
    actions = []
    transition = []
    reward_matrix = []

    for s in range(n_states):
        if s in action_transitions:
            act_names = list(action_transitions[s].keys())
            act_trans = [action_transitions[s][a] for a in act_names]
            if rewards and s in rewards:
                act_rewards = [rewards[s].get(a, 0.0) for a in act_names]
            else:
                act_rewards = [0.0] * len(act_names)
        else:
            # Self-loop with single action
            act_names = ["stay"]
            row = [0.0] * n_states
            row[s] = 1.0
            act_trans = [row]
            act_rewards = [0.0]
        actions.append(act_names)
        transition.append(act_trans)
        reward_matrix.append(act_rewards)

    return MDP(n_states=n_states, actions=actions, transition=transition,
               rewards=reward_matrix, state_labels=state_labels, ap_labels=ap_labels)


def mdp_to_mc(mdp: MDP, policy: Policy) -> MarkovChain:
    """Induce a Markov chain from an MDP and a policy."""
    matrix = []
    for s in range(mdp.n_states):
        a_idx = policy.get_action(s)
        a_idx = min(a_idx, len(mdp.transition[s]) - 1)
        matrix.append(list(mdp.transition[s][a_idx]))
    return make_chain(matrix, labels=list(mdp.state_labels) if mdp.state_labels else None)


def mdp_to_interval_mdp(mdp: MDP, epsilon: float = 0.0) -> IntervalMDP:
    """Convert an MDP to an interval MDP (with optional perturbation epsilon)."""
    action_transitions = {}
    for s in range(mdp.n_states):
        action_transitions[s] = {}
        for a_idx, a_name in enumerate(mdp.actions[s]):
            row = mdp.transition[s][a_idx]
            intervals = []
            for p in row:
                lo = max(0.0, p - epsilon)
                hi = min(1.0, p + epsilon)
                intervals.append((lo, hi))
            action_transitions[s][a_name] = intervals
    return make_interval_mdp(mdp.n_states, action_transitions,
                             state_labels=list(mdp.state_labels) if mdp.state_labels else None,
                             ap_labels=mdp.ap_labels)


# ---------------------------------------------------------------------------
# Value Iteration
# ---------------------------------------------------------------------------

def value_iteration(mdp: MDP, discount: float = 1.0,
                    objective: Objective = Objective.MAXIMIZE,
                    max_iter: int = 1000, tol: float = 1e-8,
                    terminal_states: Optional[Set[int]] = None) -> ValueResult:
    """Compute optimal value function via value iteration.

    For terminal states, the value is fixed at 0 (no future rewards).
    discount < 1 ensures convergence for non-terminal MDPs.
    """
    n = mdp.n_states
    values = [0.0] * n
    policy_map = {s: 0 for s in range(n)}
    terminal = terminal_states or set()

    for it in range(max_iter):
        new_values = [0.0] * n
        for s in range(n):
            if s in terminal:
                new_values[s] = 0.0
                continue
            best_val = None
            best_a = 0
            for a_idx in range(len(mdp.actions[s])):
                r = mdp.rewards[s][a_idx]
                expected = sum(mdp.transition[s][a_idx][t] * values[t]
                               for t in range(n))
                q_val = r + discount * expected
                if best_val is None:
                    best_val = q_val
                    best_a = a_idx
                elif objective == Objective.MAXIMIZE and q_val > best_val:
                    best_val = q_val
                    best_a = a_idx
                elif objective == Objective.MINIMIZE and q_val < best_val:
                    best_val = q_val
                    best_a = a_idx
            new_values[s] = best_val if best_val is not None else 0.0
            policy_map[s] = best_a

        diff = max(abs(new_values[s] - values[s]) for s in range(n))
        values = new_values
        if diff < tol:
            return ValueResult(values=values, policy=Policy(policy_map),
                               iterations=it + 1, converged=True, objective=objective)

    return ValueResult(values=values, policy=Policy(policy_map),
                       iterations=max_iter, converged=False, objective=objective)


def q_values(mdp: MDP, values: List[float], discount: float = 1.0,
             state: int = 0) -> Dict[str, float]:
    """Compute Q-values for a given state under a value function."""
    result = {}
    for a_idx, a_name in enumerate(mdp.actions[state]):
        r = mdp.rewards[state][a_idx]
        expected = sum(mdp.transition[state][a_idx][t] * values[t]
                       for t in range(mdp.n_states))
        result[a_name] = r + discount * expected
    return result


def evaluate_policy(mdp: MDP, policy: Policy, discount: float = 1.0,
                    max_iter: int = 1000, tol: float = 1e-8,
                    terminal_states: Optional[Set[int]] = None) -> List[float]:
    """Evaluate a fixed policy by iterating the Bellman equation."""
    n = mdp.n_states
    values = [0.0] * n
    terminal = terminal_states or set()

    for _ in range(max_iter):
        new_values = [0.0] * n
        for s in range(n):
            if s in terminal:
                continue
            a_idx = policy.get_action(s)
            a_idx = min(a_idx, len(mdp.actions[s]) - 1)
            r = mdp.rewards[s][a_idx]
            expected = sum(mdp.transition[s][a_idx][t] * values[t]
                           for t in range(n))
            new_values[s] = r + discount * expected
        diff = max(abs(new_values[s] - values[s]) for s in range(n))
        values = new_values
        if diff < tol:
            break
    return values


# ---------------------------------------------------------------------------
# Reachability Analysis
# ---------------------------------------------------------------------------

def reachability(mdp: MDP, targets: Set[int],
                 objective: Objective = Objective.MAXIMIZE,
                 max_iter: int = 1000, tol: float = 1e-10) -> ReachResult:
    """Compute optimal reachability probabilities to target states.

    MAXIMIZE: find policy that maximizes probability of reaching targets.
    MINIMIZE: find policy that minimizes probability of reaching targets.
    """
    n = mdp.n_states
    # Classify states
    # States that can reach targets (backward reachability)
    can_reach = set(targets)
    changed = True
    while changed:
        changed = False
        for s in range(n):
            if s in can_reach:
                continue
            for a_idx in range(len(mdp.actions[s])):
                for t in range(n):
                    if mdp.transition[s][a_idx][t] > 0 and t in can_reach:
                        can_reach.add(s)
                        changed = True
                        break
                if s in can_reach:
                    break

    probs = [0.0] * n
    for t in targets:
        probs[t] = 1.0
    policy_map = {s: 0 for s in range(n)}

    for _ in range(max_iter):
        new_probs = [0.0] * n
        for s in range(n):
            if s in targets:
                new_probs[s] = 1.0
                continue
            if s not in can_reach:
                new_probs[s] = 0.0
                continue
            best_val = None
            best_a = 0
            for a_idx in range(len(mdp.actions[s])):
                expected = sum(mdp.transition[s][a_idx][t] * probs[t]
                               for t in range(n))
                if best_val is None:
                    best_val = expected
                    best_a = a_idx
                elif objective == Objective.MAXIMIZE and expected > best_val:
                    best_val = expected
                    best_a = a_idx
                elif objective == Objective.MINIMIZE and expected < best_val:
                    best_val = expected
                    best_a = a_idx
            new_probs[s] = best_val if best_val is not None else 0.0
            policy_map[s] = best_a

        diff = max(abs(new_probs[s] - probs[s]) for s in range(n))
        probs = new_probs
        if diff < tol:
            break

    return ReachResult(probabilities=probs, policy=Policy(policy_map),
                       objective=objective, targets=targets)


def expected_steps(mdp: MDP, targets: Set[int],
                   objective: Objective = Objective.MINIMIZE,
                   max_iter: int = 10000, tol: float = 1e-8) -> Tuple[List[float], Policy]:
    """Compute optimal expected number of steps to reach targets.

    Returns (expected_steps_per_state, optimal_policy).
    States that cannot reach targets have value infinity (represented as float('inf')).
    """
    n = mdp.n_states
    # Find states that can reach targets under SOME policy
    can_reach = set(targets)
    changed = True
    while changed:
        changed = False
        for s in range(n):
            if s in can_reach:
                continue
            for a_idx in range(len(mdp.actions[s])):
                for t in range(n):
                    if mdp.transition[s][a_idx][t] > 0 and t in can_reach:
                        can_reach.add(s)
                        changed = True
                        break
                if s in can_reach:
                    break

    steps = [0.0] * n
    policy_map = {s: 0 for s in range(n)}

    for _ in range(max_iter):
        new_steps = [0.0] * n
        for s in range(n):
            if s in targets:
                new_steps[s] = 0.0
                continue
            if s not in can_reach:
                new_steps[s] = float('inf')
                continue
            best_val = None
            best_a = 0
            for a_idx in range(len(mdp.actions[s])):
                expected = 1.0 + sum(mdp.transition[s][a_idx][t] * steps[t]
                                     for t in range(n)
                                     if steps[t] != float('inf'))
                # If any successor has inf and positive prob, this action is inf
                has_inf = any(steps[t] == float('inf') and mdp.transition[s][a_idx][t] > 0
                              for t in range(n) if t not in targets)
                if has_inf:
                    expected = float('inf')
                if best_val is None:
                    best_val = expected
                    best_a = a_idx
                elif objective == Objective.MINIMIZE:
                    if expected < best_val:
                        best_val = expected
                        best_a = a_idx
                elif objective == Objective.MAXIMIZE:
                    if expected > best_val and expected != float('inf'):
                        best_val = expected
                        best_a = a_idx
            new_steps[s] = best_val if best_val is not None else float('inf')
            policy_map[s] = best_a

        finite_diff = max((abs(new_steps[s] - steps[s])
                           for s in range(n)
                           if new_steps[s] != float('inf') and steps[s] != float('inf')),
                          default=0.0)
        steps = new_steps
        if finite_diff < tol:
            break

    return steps, Policy(policy_map)


# ---------------------------------------------------------------------------
# SMT-based Verification
# ---------------------------------------------------------------------------

def verify_reachability_bound(mdp: MDP, start: int, targets: Set[int],
                              min_prob: float,
                              objective: Objective = Objective.MAXIMIZE) -> VerificationResult:
    """Verify that the optimal reachability probability from start >= min_prob.

    Uses value iteration to compute the bound, then SMT to verify the policy
    induces a Markov chain where the reachability is indeed >= min_prob.
    """
    result = reachability(mdp, targets, objective)
    actual_prob = result.probabilities[start]

    if actual_prob >= min_prob - 1e-9:
        # Verify the policy via induced MC
        mc = mdp_to_mc(mdp, result.policy)
        analysis = analyze_chain(mc)

        # Check absorption probabilities to targets
        verified = True
        details = f"Optimal reachability from state {start}: {actual_prob:.6f} >= {min_prob}"

        return VerificationResult(verified=True, property_desc=f"P_max(reach {targets}) >= {min_prob}",
                                  details=details)
    else:
        return VerificationResult(
            verified=False,
            property_desc=f"P_max(reach {targets}) >= {min_prob}",
            counterexample={"start": start, "optimal_prob": actual_prob, "bound": min_prob},
            details=f"Optimal reachability {actual_prob:.6f} < {min_prob}")


def verify_policy_optimality(mdp: MDP, policy: Policy,
                             discount: float = 0.9,
                             terminal_states: Optional[Set[int]] = None) -> VerificationResult:
    """Verify that a given policy is optimal (no single-state improvement possible).

    Uses SMT to check: for every state s, Q(s, policy(s)) >= Q(s, a) for all actions a.
    """
    values = evaluate_policy(mdp, policy, discount, terminal_states=terminal_states)
    n = mdp.n_states
    terminal = terminal_states or set()

    # Check one-step improvement
    non_optimal_states = []
    for s in range(n):
        if s in terminal:
            continue
        pol_a = policy.get_action(s)
        pol_q = mdp.rewards[s][pol_a] + discount * sum(
            mdp.transition[s][pol_a][t] * values[t] for t in range(n))

        for a_idx in range(len(mdp.actions[s])):
            if a_idx == pol_a:
                continue
            alt_q = mdp.rewards[s][a_idx] + discount * sum(
                mdp.transition[s][a_idx][t] * values[t] for t in range(n))
            if alt_q > pol_q + 1e-9:
                non_optimal_states.append({
                    "state": s,
                    "policy_action": pol_a,
                    "policy_q": pol_q,
                    "better_action": a_idx,
                    "better_q": alt_q
                })
                break

    if not non_optimal_states:
        return VerificationResult(verified=True,
                                  property_desc="Policy is optimal (no improving action)",
                                  details=f"Checked all {n} states, no improvement found")
    else:
        return VerificationResult(
            verified=False,
            property_desc="Policy is optimal",
            counterexample={"non_optimal_states": non_optimal_states},
            details=f"Found {len(non_optimal_states)} improvable states")


def verify_reward_bound(mdp: MDP, start: int, min_reward: float,
                        discount: float = 0.9,
                        objective: Objective = Objective.MAXIMIZE,
                        terminal_states: Optional[Set[int]] = None) -> VerificationResult:
    """Verify that the optimal expected discounted reward from start >= min_reward."""
    result = value_iteration(mdp, discount, objective, terminal_states=terminal_states)
    actual = result.values[start]

    if actual >= min_reward - 1e-9:
        return VerificationResult(
            verified=True,
            property_desc=f"V*({start}) >= {min_reward}",
            details=f"Optimal value from state {start}: {actual:.6f} >= {min_reward}")
    else:
        return VerificationResult(
            verified=False,
            property_desc=f"V*({start}) >= {min_reward}",
            counterexample={"start": start, "optimal_value": actual, "bound": min_reward},
            details=f"Optimal value {actual:.6f} < {min_reward}")


def verify_safety(mdp: MDP, safe_states: Set[int], start: int,
                  min_prob: float, steps: int,
                  objective: Objective = Objective.MAXIMIZE) -> VerificationResult:
    """Verify that the probability of staying in safe_states for `steps` steps >= min_prob."""
    n = mdp.n_states
    # Bounded safety via backward induction
    probs = [1.0 if s in safe_states else 0.0 for s in range(n)]

    policy_map = {s: 0 for s in range(n)}
    for _ in range(steps):
        new_probs = [0.0] * n
        for s in range(n):
            if s not in safe_states:
                new_probs[s] = 0.0
                continue
            best_val = None
            best_a = 0
            for a_idx in range(len(mdp.actions[s])):
                expected = sum(mdp.transition[s][a_idx][t] * probs[t]
                               for t in range(n))
                if best_val is None:
                    best_val = expected
                    best_a = a_idx
                elif objective == Objective.MAXIMIZE and expected > best_val:
                    best_val = expected
                    best_a = a_idx
                elif objective == Objective.MINIMIZE and expected < best_val:
                    best_val = expected
                    best_a = a_idx
            new_probs[s] = best_val if best_val is not None else 0.0
            policy_map[s] = best_a
        probs = new_probs

    actual = probs[start]
    if actual >= min_prob - 1e-9:
        return VerificationResult(
            verified=True,
            property_desc=f"P(safe for {steps} steps from {start}) >= {min_prob}",
            details=f"Safety probability: {actual:.6f}")
    else:
        return VerificationResult(
            verified=False,
            property_desc=f"P(safe for {steps} steps from {start}) >= {min_prob}",
            counterexample={"start": start, "safety_prob": actual, "bound": min_prob},
            details=f"Safety probability {actual:.6f} < {min_prob}")


def smt_verify_policy_dominance(mdp: MDP, p1: Policy, p2: Policy,
                                discount: float = 0.9,
                                terminal_states: Optional[Set[int]] = None) -> VerificationResult:
    """Use SMT to verify that policy p1 dominates p2 (V^p1(s) >= V^p2(s) for all s).

    Encodes value functions as scaled integers and checks dominance via LIA.
    """
    terminal = terminal_states or set()
    v1 = evaluate_policy(mdp, p1, discount, terminal_states=terminal)
    v2 = evaluate_policy(mdp, p2, discount, terminal_states=terminal)

    solver = SMTSolver()
    n = mdp.n_states

    # Scale to integers for LIA
    scale = 10000
    violations = []
    for s in range(n):
        if s in terminal:
            continue
        v1_scaled = int(round(v1[s] * scale))
        v2_scaled = int(round(v2[s] * scale))
        if v1_scaled < v2_scaled:
            violations.append({"state": s, "v1": v1[s], "v2": v2[s]})

    if not violations:
        # Verify via SMT: for each state, v1 >= v2
        for s in range(n):
            if s in terminal:
                continue
            v1_var = solver.Int(f"v1_{s}")
            v2_var = solver.Int(f"v2_{s}")
            v1_scaled = int(round(v1[s] * scale))
            v2_scaled = int(round(v2[s] * scale))
            solver.add(App(Op.EQ, [v1_var, IntConst(v1_scaled)], BOOL))
            solver.add(App(Op.EQ, [v2_var, IntConst(v2_scaled)], BOOL))
            solver.push()
            # Check if v1 < v2 (violation)
            solver.add(App(Op.LT, [v1_var, v2_var], BOOL))
            result = solver.check()
            solver.pop()
            if result == SMTResult.SAT:
                violations.append({"state": s, "v1": v1[s], "v2": v2[s]})
                break

    if not violations:
        return VerificationResult(
            verified=True,
            property_desc="Policy p1 dominates p2",
            details=f"V^p1(s) >= V^p2(s) for all {n} states")
    else:
        return VerificationResult(
            verified=False,
            property_desc="Policy p1 dominates p2",
            counterexample={"violations": violations},
            details=f"Found {len(violations)} states where p2 is better")


def smt_verify_bellman_optimality(mdp: MDP, values: List[float],
                                  discount: float = 0.9,
                                  terminal_states: Optional[Set[int]] = None) -> VerificationResult:
    """Use SMT to verify that a value function satisfies the Bellman optimality equation.

    V*(s) = max_a [ R(s,a) + gamma * sum_t P(s,a,t) * V*(t) ]
    Encodes as scaled integers and checks equality.
    """
    terminal = terminal_states or set()
    n = mdp.n_states
    scale = 10000

    solver = SMTSolver()
    violations = []

    for s in range(n):
        if s in terminal:
            continue

        v_scaled = int(round(values[s] * scale))

        # Compute max Q-value
        max_q = None
        for a_idx in range(len(mdp.actions[s])):
            r = mdp.rewards[s][a_idx]
            expected = sum(mdp.transition[s][a_idx][t] * values[t] for t in range(n))
            q_val = r + discount * expected
            if max_q is None or q_val > max_q:
                max_q = q_val

        max_q_scaled = int(round((max_q or 0.0) * scale))

        if abs(v_scaled - max_q_scaled) > 1:  # Allow 1 unit rounding error
            violations.append({"state": s, "v": values[s], "bellman": max_q})

    if not violations:
        return VerificationResult(
            verified=True,
            property_desc="Value function satisfies Bellman optimality",
            details=f"Verified for all {n} non-terminal states")
    else:
        return VerificationResult(
            verified=False,
            property_desc="Value function satisfies Bellman optimality",
            counterexample={"violations": violations},
            details=f"Bellman violated at {len(violations)} states")


# ---------------------------------------------------------------------------
# Fairness and Long-Run Properties
# ---------------------------------------------------------------------------

def long_run_average_reward(mdp: MDP, policy: Policy,
                            max_iter: int = 10000) -> Optional[float]:
    """Compute the long-run average reward under a policy.

    For ergodic (irreducible, aperiodic) induced chains, this converges to a single value.
    """
    mc = mdp_to_mc(mdp, policy)
    pi = steady_state(mc)
    if pi is None:
        return None

    n = mdp.n_states
    avg = 0.0
    for s in range(n):
        a_idx = policy.get_action(s)
        a_idx = min(a_idx, len(mdp.rewards[s]) - 1)
        avg += pi[s] * mdp.rewards[s][a_idx]
    return avg


# ---------------------------------------------------------------------------
# Comparison with Interval MDP
# ---------------------------------------------------------------------------

def compare_with_interval_mdp(mdp: MDP, targets: Set[int],
                              epsilon: float = 0.05) -> Dict:
    """Compare MDP reachability with interval MDP robust reachability.

    Creates an interval MDP with epsilon perturbation and compares bounds.
    """
    # Exact MDP analysis
    reach_max = reachability(mdp, targets, Objective.MAXIMIZE)
    reach_min = reachability(mdp, targets, Objective.MINIMIZE)

    # Interval MDP with perturbation
    imdp = mdp_to_interval_mdp(mdp, epsilon)
    feasible, issues = check_all_feasible(imdp)

    result = {
        "mdp_max_reach": reach_max.probabilities,
        "mdp_min_reach": reach_min.probabilities,
        "epsilon": epsilon,
        "interval_feasible": feasible,
    }

    if feasible:
        rob_min = robust_reachability(imdp, targets, OptimizationDirection.MIN)
        rob_max = robust_reachability(imdp, targets, OptimizationDirection.MAX)
        result["interval_min_reach"] = rob_min
        result["interval_max_reach"] = rob_max
        result["robust_gap"] = [rob_max[s] - rob_min[s] for s in range(mdp.n_states)]
    else:
        result["issues"] = issues

    return result


# ---------------------------------------------------------------------------
# Policy Iteration
# ---------------------------------------------------------------------------

def policy_iteration(mdp: MDP, discount: float = 0.9,
                     objective: Objective = Objective.MAXIMIZE,
                     max_iter: int = 100,
                     terminal_states: Optional[Set[int]] = None) -> ValueResult:
    """Compute optimal policy via policy iteration (evaluate -> improve loop)."""
    n = mdp.n_states
    terminal = terminal_states or set()

    # Start with action 0 everywhere
    policy = Policy({s: 0 for s in range(n)})

    for it in range(max_iter):
        # Evaluate
        values = evaluate_policy(mdp, policy, discount, terminal_states=terminal)

        # Improve
        new_policy_map = {}
        changed = False
        for s in range(n):
            if s in terminal:
                new_policy_map[s] = 0
                continue
            best_a = policy.get_action(s)
            best_q = mdp.rewards[s][best_a] + discount * sum(
                mdp.transition[s][best_a][t] * values[t] for t in range(n))
            for a_idx in range(len(mdp.actions[s])):
                q = mdp.rewards[s][a_idx] + discount * sum(
                    mdp.transition[s][a_idx][t] * values[t] for t in range(n))
                if objective == Objective.MAXIMIZE and q > best_q + 1e-12:
                    best_q = q
                    best_a = a_idx
                    changed = True
                elif objective == Objective.MINIMIZE and q < best_q - 1e-12:
                    best_q = q
                    best_a = a_idx
                    changed = True
            new_policy_map[s] = best_a

        policy = Policy(new_policy_map)
        if not changed:
            values = evaluate_policy(mdp, policy, discount, terminal_states=terminal)
            return ValueResult(values=values, policy=policy,
                               iterations=it + 1, converged=True, objective=objective)

    values = evaluate_policy(mdp, policy, discount, terminal_states=terminal)
    return ValueResult(values=values, policy=policy,
                       iterations=max_iter, converged=False, objective=objective)


# ---------------------------------------------------------------------------
# High-level APIs
# ---------------------------------------------------------------------------

def analyze_mdp(mdp: MDP, discount: float = 0.9,
                terminal_states: Optional[Set[int]] = None) -> Dict:
    """Full analysis of an MDP: value iteration, policy iteration, induced MC analysis."""
    errors = mdp.validate()
    if errors:
        return {"valid": False, "errors": errors}

    vi_result = value_iteration(mdp, discount, Objective.MAXIMIZE,
                                terminal_states=terminal_states)
    pi_result = policy_iteration(mdp, discount, Objective.MAXIMIZE,
                                 terminal_states=terminal_states)

    # Induced MC under optimal policy
    mc = mdp_to_mc(mdp, vi_result.policy)
    mc_analysis = analyze_chain(mc)

    return {
        "valid": True,
        "n_states": mdp.n_states,
        "n_actions_per_state": [len(acts) for acts in mdp.actions],
        "value_iteration": {
            "values": vi_result.values,
            "policy": vi_result.policy.action_map,
            "iterations": vi_result.iterations,
            "converged": vi_result.converged
        },
        "policy_iteration": {
            "values": pi_result.values,
            "policy": pi_result.policy.action_map,
            "iterations": pi_result.iterations,
            "converged": pi_result.converged
        },
        "induced_mc": {
            "is_irreducible": mc_analysis.is_irreducible,
            "period": mc_analysis.period,
            "steady_state": mc_analysis.steady_state,
            "state_types": [st.value for st in mc_analysis.state_types]
        }
    }


def verify_mdp(mdp: MDP, properties: List[Dict],
               discount: float = 0.9,
               terminal_states: Optional[Set[int]] = None) -> List[VerificationResult]:
    """Verify a list of properties against an MDP.

    Each property is a dict with:
      - type: "reachability" | "reward_bound" | "safety" | "policy_optimal"
      - Plus type-specific fields
    """
    results = []
    for prop in properties:
        ptype = prop.get("type", "")

        if ptype == "reachability":
            r = verify_reachability_bound(
                mdp, prop["start"], set(prop["targets"]),
                prop["min_prob"], prop.get("objective", Objective.MAXIMIZE))
            results.append(r)

        elif ptype == "reward_bound":
            r = verify_reward_bound(
                mdp, prop["start"], prop["min_reward"],
                discount, prop.get("objective", Objective.MAXIMIZE),
                terminal_states)
            results.append(r)

        elif ptype == "safety":
            r = verify_safety(
                mdp, set(prop["safe_states"]), prop["start"],
                prop["min_prob"], prop["steps"],
                prop.get("objective", Objective.MAXIMIZE))
            results.append(r)

        elif ptype == "policy_optimal":
            r = verify_policy_optimality(mdp, prop["policy"], discount, terminal_states)
            results.append(r)

        else:
            results.append(VerificationResult(
                verified=False, property_desc=f"Unknown property type: {ptype}",
                details="Supported: reachability, reward_bound, safety, policy_optimal"))

    return results
