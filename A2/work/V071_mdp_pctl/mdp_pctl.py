"""V071: MDP Model Checking (PCTL for MDPs)

Extends V067 PCTL model checking to handle MDP nondeterminism.
Composes V067 (PCTL AST/parser) + V069 (MDP data structures).

In a Markov chain, transition probabilities are fixed. In an MDP, each state
has a nondeterministic choice of actions. PCTL model checking for MDPs computes:
  - Pmax(phi): maximum probability over all policies (optimistic)
  - Pmin(phi): minimum probability over all policies (pessimistic)

Satisfaction semantics (standard universal interpretation):
  - P>=p[phi] holds iff Pmin(phi) >= p (under ALL policies, prob >= p)
  - P<=p[phi] holds iff Pmax(phi) <= p (under ALL policies, prob <= p)

Also supports existential interpretation:
  - P>=p[phi] holds iff Pmax(phi) >= p (EXISTS a policy with prob >= p)
  - P<=p[phi] holds iff Pmin(phi) <= p (EXISTS a policy with prob <= p)

Algorithms:
  - Next: max/min over actions of sum of transition probs to satisfying states
  - Until: value iteration with max/min over actions at each step
  - Bounded Until: backward induction with max/min over actions
  - Expected reward until target with max/min over actions
"""

from __future__ import annotations
import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Dict, Tuple

# Import V067 PCTL AST and parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
from pctl_model_check import (
    PCTL, FormulaKind, PCTLResult,
    tt, ff, atom, pnot, pand, por,
    prob_geq, prob_leq, prob_gt, prob_lt,
    next_f, until, bounded_until,
    eventually, always, bounded_eventually,
    parse_pctl,
)

# Import V069 MDP
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V069_mdp_verification'))
from mdp_verification import (
    MDP, Policy, make_mdp, mdp_to_mc, Objective,
)

# Import V065 for MC analysis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
from markov_chain import MarkovChain, make_chain, analyze_chain


# ---------------------------------------------------------------------------
# Labeled MDP
# ---------------------------------------------------------------------------

@dataclass
class LabeledMDP:
    """MDP with state labeling for PCTL model checking."""
    mdp: MDP
    labels: Dict[int, Set[str]]

    def states_with(self, label: str) -> Set[int]:
        return {s for s, labs in self.labels.items() if label in labs}

    def states_without(self, label: str) -> Set[int]:
        return set(range(self.mdp.n_states)) - self.states_with(label)


def make_labeled_mdp(n_states: int,
                     action_transitions: Dict[int, Dict[str, List[float]]],
                     labels: Dict[int, Set[str]],
                     rewards: Optional[Dict[int, Dict[str, float]]] = None,
                     state_labels: Optional[List[str]] = None) -> LabeledMDP:
    """Create a labeled MDP from convenient dict format."""
    mdp = make_mdp(n_states, action_transitions, rewards, state_labels,
                   ap_labels=None)
    full_labels = {}
    for s in range(n_states):
        full_labels[s] = set(labels.get(s, set()))
    return LabeledMDP(mdp=mdp, labels=full_labels)


# ---------------------------------------------------------------------------
# Quantification mode
# ---------------------------------------------------------------------------

class Quantification(Enum):
    """How to interpret P~p operators over MDP nondeterminism."""
    UNIVERSAL = "universal"    # For ALL policies
    EXISTENTIAL = "existential"  # EXISTS a policy


# ---------------------------------------------------------------------------
# MDP PCTL Checker
# ---------------------------------------------------------------------------

class MDPPCTLChecker:
    """PCTL model checker for MDPs.

    Computes min and max probabilities over all deterministic memoryless
    policies via value iteration, then checks PCTL formulas.
    """

    def __init__(self, lmdp: LabeledMDP, tol: float = 1e-10,
                 max_iter: int = 10000,
                 quantification: Quantification = Quantification.UNIVERSAL):
        self.lmdp = lmdp
        self.mdp = lmdp.mdp
        self.n = lmdp.mdp.n_states
        self.tol = tol
        self.max_iter = max_iter
        self.quantification = quantification

    def check(self, formula: PCTL) -> Set[int]:
        """Return the set of states satisfying the PCTL formula."""
        kind = formula.kind

        if kind == FormulaKind.TRUE:
            return set(range(self.n))
        elif kind == FormulaKind.FALSE:
            return set()
        elif kind == FormulaKind.ATOM:
            return self.lmdp.states_with(formula.label)
        elif kind == FormulaKind.NOT:
            return set(range(self.n)) - self.check(formula.sub)
        elif kind == FormulaKind.AND:
            return self.check(formula.left) & self.check(formula.right)
        elif kind == FormulaKind.OR:
            return self.check(formula.left) | self.check(formula.right)
        elif kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_LEQ,
                      FormulaKind.PROB_GT, FormulaKind.PROB_LT):
            return self._check_prob(formula)
        else:
            raise ValueError(f"Unexpected formula kind: {kind}")

    def _check_prob(self, formula: PCTL) -> Set[int]:
        """Handle P~p[path_formula] for MDPs.

        Under universal quantification:
          P>=p: need Pmin >= p (all policies achieve >= p)
          P<=p: need Pmax <= p (all policies achieve <= p)
          P>p:  need Pmin > p
          P<p:  need Pmax < p

        Under existential quantification:
          P>=p: need Pmax >= p (some policy achieves >= p)
          P<=p: need Pmin <= p (some policy achieves <= p)
          P>p:  need Pmax > p
          P<p:  need Pmin < p
        """
        path = formula.path
        threshold = formula.threshold

        # Determine which probability (min or max) to use
        if formula.kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_GT):
            if self.quantification == Quantification.UNIVERSAL:
                probs = self._path_probs_min(path)
            else:
                probs = self._path_probs_max(path)
        else:  # PROB_LEQ, PROB_LT
            if self.quantification == Quantification.UNIVERSAL:
                probs = self._path_probs_max(path)
            else:
                probs = self._path_probs_min(path)

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

    # -------------------------------------------------------------------
    # Max probability computation
    # -------------------------------------------------------------------

    def _path_probs_max(self, path: PCTL) -> List[float]:
        """Compute MAXIMUM probability of path formula over all policies."""
        kind = path.kind
        if kind == FormulaKind.NEXT:
            return self._next_probs(path.sub, maximize=True)
        elif kind == FormulaKind.UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._until_probs(phi_sat, psi_sat, maximize=True)
        elif kind == FormulaKind.BOUNDED_UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._bounded_until_probs(phi_sat, psi_sat, path.bound,
                                              maximize=True)
        else:
            raise ValueError(f"Unexpected path formula kind: {kind}")

    def _path_probs_min(self, path: PCTL) -> List[float]:
        """Compute MINIMUM probability of path formula over all policies."""
        kind = path.kind
        if kind == FormulaKind.NEXT:
            return self._next_probs(path.sub, maximize=False)
        elif kind == FormulaKind.UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._until_probs(phi_sat, psi_sat, maximize=False)
        elif kind == FormulaKind.BOUNDED_UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._bounded_until_probs(phi_sat, psi_sat, path.bound,
                                              maximize=False)
        else:
            raise ValueError(f"Unexpected path formula kind: {kind}")

    # -------------------------------------------------------------------
    # Next-state probabilities
    # -------------------------------------------------------------------

    def _next_probs(self, phi: PCTL, maximize: bool) -> List[float]:
        """P_max/min(X phi | s) = max/min_a sum_{t in Sat(phi)} P(s,a,t)."""
        phi_sat = self.check(phi)
        probs = []
        for s in range(self.n):
            vals = []
            for a_idx in range(len(self.mdp.actions[s])):
                p = sum(self.mdp.transition[s][a_idx][t] for t in phi_sat)
                vals.append(p)
            if maximize:
                probs.append(max(vals))
            else:
                probs.append(min(vals))
        return probs

    # -------------------------------------------------------------------
    # Unbounded Until (value iteration)
    # -------------------------------------------------------------------

    def _until_probs(self, phi_sat: Set[int], psi_sat: Set[int],
                     maximize: bool) -> List[float]:
        """Compute P_max/min(phi U psi) via value iteration.

        State classification:
        - S_yes (prob=1): psi holds
        - S_no (prob=0): can't reach psi through phi states
          For Pmax: no action sequence can reach psi through phi
          For Pmin: same classification (states that are S_no under ANY policy)
        - S_maybe: solve via value iteration
        """
        all_states = set(range(self.n))

        # S_yes: psi already holds
        s_yes = set(psi_sat)

        # S_no: states not in phi and not in psi
        s_no = all_states - phi_sat - psi_sat

        # For max: states in phi-psi that can reach psi through phi under
        # SOME action sequence. States that can't reach -> S_no.
        # For min: same classification. States truly unreachable are S_no
        # regardless of policy.
        # Backward BFS from psi through phi, considering all actions.
        reachable = set(psi_sat)
        worklist = list(psi_sat)
        while worklist:
            t = worklist.pop()
            for s in range(self.n):
                if s in reachable or s not in phi_sat or s in psi_sat:
                    continue
                # Check if any action can transition to t
                for a_idx in range(len(self.mdp.actions[s])):
                    if self.mdp.transition[s][a_idx][t] > 0:
                        reachable.add(s)
                        worklist.append(s)
                        break

        # States in phi-psi that can't reach psi under any action -> S_no
        for s in phi_sat - psi_sat:
            if s not in reachable:
                s_no.add(s)

        # For Pmin, additional refinement: states where the minimizer can
        # FORCE prob=0 by choosing actions that avoid psi-reachable states.
        # A state is S_no for Pmin if the minimizer can choose to stay in
        # a cycle that never reaches psi.
        if not maximize:
            # Compute states from which psi is reachable under ALL actions
            # (adversary can't avoid psi). Use backward attractor computation.
            # A state s is in the "attractor" if:
            #   - s is in psi_sat, OR
            #   - s is in phi_sat and ALL actions from s lead to the attractor
            #     (i.e., for each action, some successor is in attractor)
            # States NOT in this attractor under min can be forced to prob 0.

            # Actually for Pmin, the correct classification uses:
            # S_no_min = states from which the minimizer can avoid psi forever.
            # This is the complement of the "almost-sure reachability" set.
            # A state has Pmin > 0 iff EVERY policy eventually reaches psi.
            # But value iteration handles this correctly -- states where the
            # minimizer can delay indefinitely will converge to 0.
            # So we just let value iteration handle it.
            pass

        s_maybe = all_states - s_yes - s_no

        probs = [0.0] * self.n
        for s in s_yes:
            probs[s] = 1.0

        if not s_maybe:
            return probs

        # Value iteration for S_maybe states
        for _ in range(self.max_iter):
            new_probs = list(probs)
            for s in s_maybe:
                vals = []
                for a_idx in range(len(self.mdp.actions[s])):
                    expected = sum(self.mdp.transition[s][a_idx][t] * probs[t]
                                   for t in range(self.n))
                    vals.append(expected)
                if maximize:
                    new_probs[s] = max(vals)
                else:
                    new_probs[s] = min(vals)

            diff = max(abs(new_probs[s] - probs[s]) for s in s_maybe)
            probs = new_probs
            if diff < self.tol:
                break

        # Clamp to [0, 1]
        for s in range(self.n):
            probs[s] = max(0.0, min(1.0, probs[s]))

        return probs

    # -------------------------------------------------------------------
    # Bounded Until (backward induction)
    # -------------------------------------------------------------------

    def _bounded_until_probs(self, phi_sat: Set[int], psi_sat: Set[int],
                              k: int, maximize: bool) -> List[float]:
        """Compute P_max/min(phi U<=k psi) via backward induction."""
        prob = [0.0] * self.n
        for s in psi_sat:
            prob[s] = 1.0

        for step in range(k):
            new_prob = [0.0] * self.n
            for s in range(self.n):
                if s in psi_sat:
                    new_prob[s] = 1.0
                elif s in phi_sat:
                    vals = []
                    for a_idx in range(len(self.mdp.actions[s])):
                        expected = sum(self.mdp.transition[s][a_idx][t] * prob[t]
                                       for t in range(self.n))
                        vals.append(expected)
                    if maximize:
                        new_prob[s] = max(vals)
                    else:
                        new_prob[s] = min(vals)
                # else: not in phi or psi -> 0
            prob = new_prob

        return prob

    # -------------------------------------------------------------------
    # Quantitative API (raw probability vectors)
    # -------------------------------------------------------------------

    def check_quantitative_max(self, path_formula: PCTL) -> List[float]:
        """Compute max probability of path formula from each state."""
        return self._path_probs_max(path_formula)

    def check_quantitative_min(self, path_formula: PCTL) -> List[float]:
        """Compute min probability of path formula from each state."""
        return self._path_probs_min(path_formula)

    def check_state(self, state: int, formula: PCTL) -> bool:
        return state in self.check(formula)

    def check_all(self, formula: PCTL) -> bool:
        return self.check(formula) == set(range(self.n))

    # -------------------------------------------------------------------
    # Policy extraction
    # -------------------------------------------------------------------

    def extract_policy(self, path_formula: PCTL,
                       maximize: bool = True) -> Policy:
        """Extract the optimal policy for a path formula.

        Returns the policy that maximizes (or minimizes) the probability
        of the path formula.
        """
        kind = path_formula.kind
        if kind == FormulaKind.NEXT:
            return self._next_policy(path_formula.sub, maximize)
        elif kind == FormulaKind.UNTIL:
            phi_sat = self.check(path_formula.left)
            psi_sat = self.check(path_formula.right)
            return self._until_policy(phi_sat, psi_sat, maximize)
        elif kind == FormulaKind.BOUNDED_UNTIL:
            phi_sat = self.check(path_formula.left)
            psi_sat = self.check(path_formula.right)
            return self._bounded_until_policy(phi_sat, psi_sat,
                                               path_formula.bound, maximize)
        else:
            raise ValueError(f"Unexpected path formula kind: {kind}")

    def _next_policy(self, phi: PCTL, maximize: bool) -> Policy:
        phi_sat = self.check(phi)
        policy_map = {}
        for s in range(self.n):
            best_a = 0
            best_val = None
            for a_idx in range(len(self.mdp.actions[s])):
                p = sum(self.mdp.transition[s][a_idx][t] for t in phi_sat)
                if best_val is None:
                    best_val = p
                    best_a = a_idx
                elif maximize and p > best_val:
                    best_val = p
                    best_a = a_idx
                elif not maximize and p < best_val:
                    best_val = p
                    best_a = a_idx
            policy_map[s] = best_a
        return Policy(action_map=policy_map)

    def _until_policy(self, phi_sat: Set[int], psi_sat: Set[int],
                      maximize: bool) -> Policy:
        """Extract policy from the converged value iteration."""
        if maximize:
            probs = self._until_probs(phi_sat, psi_sat, maximize=True)
        else:
            probs = self._until_probs(phi_sat, psi_sat, maximize=False)

        policy_map = {}
        for s in range(self.n):
            best_a = 0
            best_val = None
            for a_idx in range(len(self.mdp.actions[s])):
                expected = sum(self.mdp.transition[s][a_idx][t] * probs[t]
                               for t in range(self.n))
                if best_val is None:
                    best_val = expected
                    best_a = a_idx
                elif maximize and expected > best_val:
                    best_val = expected
                    best_a = a_idx
                elif not maximize and expected < best_val:
                    best_val = expected
                    best_a = a_idx
            policy_map[s] = best_a
        return Policy(action_map=policy_map)

    def _bounded_until_policy(self, phi_sat: Set[int], psi_sat: Set[int],
                               k: int, maximize: bool) -> Policy:
        """Extract greedy policy from final step of backward induction."""
        if maximize:
            probs = self._bounded_until_probs(phi_sat, psi_sat, k,
                                               maximize=True)
        else:
            probs = self._bounded_until_probs(phi_sat, psi_sat, k,
                                               maximize=False)

        policy_map = {}
        for s in range(self.n):
            best_a = 0
            best_val = None
            for a_idx in range(len(self.mdp.actions[s])):
                expected = sum(self.mdp.transition[s][a_idx][t] * probs[t]
                               for t in range(self.n))
                if best_val is None:
                    best_val = expected
                    best_a = a_idx
                elif maximize and expected > best_val:
                    best_val = expected
                    best_a = a_idx
                elif not maximize and expected < best_val:
                    best_val = expected
                    best_a = a_idx
            policy_map[s] = best_a
        return Policy(action_map=policy_map)


# ---------------------------------------------------------------------------
# Expected reward until target (MDP version)
# ---------------------------------------------------------------------------

def mdp_expected_reward(lmdp: LabeledMDP, rewards: List[float],
                        target: PCTL, maximize: bool = True,
                        max_iter: int = 10000, tol: float = 1e-10
                        ) -> Tuple[List[float], Policy]:
    """Compute expected accumulated reward until target, optimizing over policies.

    rewards[s] = per-step reward in state s.
    maximize=True: find policy that maximizes expected reward.
    maximize=False: find policy that minimizes expected reward.

    Returns (expected_rewards_per_state, optimal_policy).
    """
    checker = MDPPCTLChecker(lmdp, tol=tol, max_iter=max_iter)
    target_sat = checker.check(target)

    n = lmdp.mdp.n_states
    mdp = lmdp.mdp

    values = [0.0] * n
    policy_map = {s: 0 for s in range(n)}

    for _ in range(max_iter):
        new_values = [0.0] * n
        for s in range(n):
            if s in target_sat:
                new_values[s] = 0.0
                continue
            best_val = None
            best_a = 0
            for a_idx in range(len(mdp.actions[s])):
                expected = rewards[s] + sum(
                    mdp.transition[s][a_idx][t] * values[t]
                    for t in range(n)
                )
                if best_val is None:
                    best_val = expected
                    best_a = a_idx
                elif maximize and expected > best_val:
                    best_val = expected
                    best_a = a_idx
                elif not maximize and expected < best_val:
                    best_val = expected
                    best_a = a_idx
            new_values[s] = best_val if best_val is not None else 0.0
            policy_map[s] = best_a

        diff = max(abs(new_values[s] - values[s]) for s in range(n))
        values = new_values
        if diff < tol:
            break

    return values, Policy(action_map=policy_map)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

@dataclass
class MDPPCTLResult:
    """Result of PCTL model checking on an MDP."""
    formula: PCTL
    satisfying_states: Set[int]
    all_states: int
    prob_max: Optional[List[float]] = None
    prob_min: Optional[List[float]] = None
    quantification: Quantification = Quantification.UNIVERSAL
    state_labels: Optional[List[str]] = None
    policy_max: Optional[Policy] = None
    policy_min: Optional[Policy] = None

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
            f"Quantification: {self.quantification.value}",
            f"Satisfying states: {sat_names} ({len(self.satisfying_states)}/{self.all_states})",
        ]
        if self.prob_max is not None:
            lines.append("Max probabilities:")
            for i, p in enumerate(self.prob_max):
                lines.append(f"  {labels[i]}: {p:.6f}")
        if self.prob_min is not None:
            lines.append("Min probabilities:")
            for i, p in enumerate(self.prob_min):
                lines.append(f"  {labels[i]}: {p:.6f}")
        return "\n".join(lines)


def check_mdp_pctl(lmdp: LabeledMDP, formula: PCTL,
                   quantification: Quantification = Quantification.UNIVERSAL,
                   tol: float = 1e-10, max_iter: int = 10000
                   ) -> MDPPCTLResult:
    """Check a PCTL formula against a labeled MDP.

    Returns MDPPCTLResult with satisfying states, min/max probabilities,
    and witness policies.
    """
    checker = MDPPCTLChecker(lmdp, tol=tol, max_iter=max_iter,
                             quantification=quantification)
    sat = checker.check(formula)

    prob_max = None
    prob_min = None
    policy_max = None
    policy_min = None

    if formula.kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_LEQ,
                        FormulaKind.PROB_GT, FormulaKind.PROB_LT):
        prob_max = checker.check_quantitative_max(formula.path)
        prob_min = checker.check_quantitative_min(formula.path)
        policy_max = checker.extract_policy(formula.path, maximize=True)
        policy_min = checker.extract_policy(formula.path, maximize=False)

    return MDPPCTLResult(
        formula=formula,
        satisfying_states=sat,
        all_states=lmdp.mdp.n_states,
        prob_max=prob_max,
        prob_min=prob_min,
        quantification=quantification,
        state_labels=lmdp.mdp.state_labels,
        policy_max=policy_max,
        policy_min=policy_min,
    )


def check_mdp_pctl_state(lmdp: LabeledMDP, state: int, formula: PCTL,
                          quantification: Quantification = Quantification.UNIVERSAL
                          ) -> bool:
    """Check if a specific state satisfies a PCTL formula in the MDP."""
    checker = MDPPCTLChecker(lmdp, quantification=quantification)
    return checker.check_state(state, formula)


def mdp_pctl_quantitative(lmdp: LabeledMDP, path_formula: PCTL
                           ) -> Dict[str, List[float]]:
    """Compute min and max probabilities for a path formula.

    Returns dict with 'max' and 'min' probability vectors.
    """
    checker = MDPPCTLChecker(lmdp)
    return {
        'max': checker.check_quantitative_max(path_formula),
        'min': checker.check_quantitative_min(path_formula),
    }


def verify_mdp_property(lmdp: LabeledMDP, formula: PCTL,
                         initial_state: int = 0,
                         quantification: Quantification = Quantification.UNIVERSAL
                         ) -> Dict:
    """Verify a PCTL property at an initial state of an MDP.

    Returns a verification dict.
    """
    result = check_mdp_pctl(lmdp, formula, quantification=quantification)
    holds = initial_state in result.satisfying_states

    out = {
        'holds': holds,
        'initial_state': initial_state,
        'formula': str(formula),
        'quantification': quantification.value,
        'satisfying_states': sorted(result.satisfying_states),
        'total_states': result.all_states,
    }
    if result.prob_max is not None:
        out['prob_max'] = result.prob_max
        out['prob_min'] = result.prob_min
        out['max_at_initial'] = result.prob_max[initial_state]
        out['min_at_initial'] = result.prob_min[initial_state]

    return out


def compare_quantifications(lmdp: LabeledMDP, formula: PCTL) -> Dict:
    """Compare universal vs existential quantification for a formula."""
    uni = check_mdp_pctl(lmdp, formula, Quantification.UNIVERSAL)
    exi = check_mdp_pctl(lmdp, formula, Quantification.EXISTENTIAL)

    return {
        'formula': str(formula),
        'universal_sat': sorted(uni.satisfying_states),
        'existential_sat': sorted(exi.satisfying_states),
        'universal_count': len(uni.satisfying_states),
        'existential_count': len(exi.satisfying_states),
        'prob_max': uni.prob_max,
        'prob_min': uni.prob_min,
        'universal_subset_of_existential': uni.satisfying_states.issubset(exi.satisfying_states),
    }


def batch_check_mdp(lmdp: LabeledMDP, formulas: List[PCTL],
                     quantification: Quantification = Quantification.UNIVERSAL
                     ) -> List[MDPPCTLResult]:
    """Check multiple PCTL formulas against the same MDP."""
    return [check_mdp_pctl(lmdp, f, quantification) for f in formulas]


def induced_mc_comparison(lmdp: LabeledMDP, formula: PCTL) -> Dict:
    """Compare MDP PCTL result with induced MC (under max/min policies).

    Shows that the MDP result bounds the MC result.
    """
    from pctl_model_check import LabeledMC, make_labeled_mc, check_pctl

    result = check_mdp_pctl(lmdp, formula, Quantification.EXISTENTIAL)

    comparison = {
        'formula': str(formula),
        'mdp_sat_existential': sorted(result.satisfying_states),
    }

    if result.policy_max is not None:
        # Induce MC under max policy
        mc_max = mdp_to_mc(lmdp.mdp, result.policy_max)
        lmc_max = LabeledMC(mc=mc_max, labels=dict(lmdp.labels))
        mc_result_max = check_pctl(lmc_max, formula)
        comparison['mc_max_policy_sat'] = sorted(mc_result_max.satisfying_states)
        if mc_result_max.probabilities is not None:
            comparison['mc_max_probs'] = mc_result_max.probabilities

    if result.policy_min is not None:
        mc_min = mdp_to_mc(lmdp.mdp, result.policy_min)
        lmc_min = LabeledMC(mc=mc_min, labels=dict(lmdp.labels))
        mc_result_min = check_pctl(lmc_min, formula)
        comparison['mc_min_policy_sat'] = sorted(mc_result_min.satisfying_states)
        if mc_result_min.probabilities is not None:
            comparison['mc_min_probs'] = mc_result_min.probabilities

    if result.prob_max is not None:
        comparison['mdp_prob_max'] = result.prob_max
        comparison['mdp_prob_min'] = result.prob_min

    return comparison
