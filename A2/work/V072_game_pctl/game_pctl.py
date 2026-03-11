"""
V072: PCTL Model Checking for Stochastic Games

Extends V067 PCTL model checking and V071 MDP PCTL to two-player
stochastic games (V070). In a game, Player 1 (maximizer) and Player 2
(minimizer) alternately choose actions at their owned states. CHANCE
states resolve stochastically with fixed probabilities.

PCTL model checking for games computes:
- Pmax: the probability P1 can guarantee (best P1 strategy, worst P2 response)
- Pmin: the probability P2 can force (best P2 strategy, worst P1 response)

Key algorithms:
- Next: owner-dependent max/min over actions
- Until: two-player value iteration (minimax at each state)
- Bounded Until: two-player backward induction
- Expected reward: two-player value iteration with per-step rewards

Composes:
- V067 (PCTL AST/parser)
- V070 (StochasticGame, Player, StrategyPair)
- V065 (MarkovChain for induced chain comparison)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V070_stochastic_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))

from pctl_model_check import (
    PCTL, FormulaKind, tt, ff, atom, pnot, pand, por,
    prob_geq, prob_leq, prob_gt, prob_lt,
    next_f, until, bounded_until,
    eventually, always, bounded_eventually,
    parse_pctl,
)
from stochastic_games import (
    StochasticGame, Player, StrategyPair, GameValueResult,
    GameReachResult, make_game, game_to_mc,
)
from markov_chain import MarkovChain, make_chain, analyze_chain

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# Labeled Stochastic Game
# ============================================================

@dataclass
class LabeledGame:
    """Stochastic game with state labeling for PCTL atomic propositions."""
    game: StochasticGame
    labels: Dict[int, Set[str]]

    def states_with(self, label: str) -> Set[int]:
        """Return set of states where label holds."""
        return {s for s in range(self.game.n_states) if label in self.labels.get(s, set())}

    def states_without(self, label: str) -> Set[int]:
        """Return set of states where label does not hold."""
        return {s for s in range(self.game.n_states) if label not in self.labels.get(s, set())}


def make_labeled_game(n_states, owners, action_transitions, labels,
                      rewards=None, state_labels=None):
    """Convenience constructor for LabeledGame.

    Args:
        n_states: number of states
        owners: dict mapping state -> Player (P1, P2, or CHANCE)
        action_transitions: dict mapping state -> {action_name: [prob_to_each_state]}
        labels: dict mapping state -> set of atomic proposition strings
        rewards: optional dict mapping state -> {action_name: reward}
        state_labels: optional list of state name strings
    """
    game = make_game(n_states, owners, action_transitions, rewards, state_labels)
    full_labels = {s: labels.get(s, set()) for s in range(n_states)}
    return LabeledGame(game=game, labels=full_labels)


# ============================================================
# Quantification modes for games
# ============================================================

class GameQuantification(Enum):
    """How to interpret P~p in a game context."""
    P1_OPTIMISTIC = "p1_optimistic"     # P1 maximizes, check if Pmax satisfies
    P2_OPTIMISTIC = "p2_optimistic"     # P2 minimizes, check if Pmin satisfies
    COOPERATIVE = "cooperative"          # Both players cooperate (upper bound)
    ADVERSARIAL = "adversarial"          # P1 max, P2 min (standard game value)


# ============================================================
# Game PCTL Result
# ============================================================

@dataclass
class GamePCTLResult:
    """Result of PCTL model checking on a stochastic game."""
    formula: PCTL
    satisfying_states: Set[int]
    all_states: int
    prob_max: Optional[List[float]] = None   # P1 maximizes (game value)
    prob_min: Optional[List[float]] = None   # P2 minimizes (game value)
    quantification: GameQuantification = GameQuantification.ADVERSARIAL
    state_labels: Optional[List[str]] = None
    p1_strategy: Optional[Dict[int, int]] = None  # P1 strategy for max
    p2_strategy: Optional[Dict[int, int]] = None  # P2 strategy for min

    @property
    def all_satisfy(self):
        return len(self.satisfying_states) == self.all_states

    @property
    def none_satisfy(self):
        return len(self.satisfying_states) == 0

    def summary(self):
        lines = [f"Formula: {self.formula}"]
        lines.append(f"Satisfying: {len(self.satisfying_states)}/{self.all_states} states")
        lines.append(f"Quantification: {self.quantification.value}")
        if self.prob_max is not None:
            lines.append(f"Pmax: {[round(p, 6) for p in self.prob_max]}")
        if self.prob_min is not None:
            lines.append(f"Pmin: {[round(p, 6) for p in self.prob_min]}")
        return "\n".join(lines)


# ============================================================
# Game PCTL Checker
# ============================================================

class GamePCTLChecker:
    """PCTL model checker for stochastic games."""

    def __init__(self, lgame, tol=1e-10, max_iter=10000,
                 quantification=GameQuantification.ADVERSARIAL):
        self.lgame = lgame
        self.game = lgame.game
        self.n = lgame.game.n_states
        self.tol = tol
        self.max_iter = max_iter
        self.quantification = quantification

    def check(self, formula):
        """Check which states satisfy a PCTL formula. Returns set of states."""
        kind = formula.kind

        if kind == FormulaKind.TRUE:
            return set(range(self.n))
        elif kind == FormulaKind.FALSE:
            return set()
        elif kind == FormulaKind.ATOM:
            return self.lgame.states_with(formula.label)
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
            raise ValueError(f"Unknown formula kind: {kind}")

    def _check_prob(self, formula):
        """Check a probability operator P~p[path]."""
        threshold = formula.threshold
        path = formula.path
        kind = formula.kind

        # Compute game values (Pmax and Pmin)
        pmax = self._path_probs_game_value(path)  # P1 maximizes, P2 minimizes
        pmin = pmax  # In adversarial mode, pmax IS the game value

        # For cooperative mode, also compute when both cooperate to max/min
        if self.quantification == GameQuantification.COOPERATIVE:
            pmax = self._path_probs_cooperative(path, maximize=True)
            pmin = self._path_probs_cooperative(path, maximize=False)

        # Determine which probability vector to compare against threshold
        if kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_GT):
            if self.quantification == GameQuantification.P1_OPTIMISTIC:
                probs = pmax
            elif self.quantification == GameQuantification.P2_OPTIMISTIC:
                pmin_vec = self._path_probs_p2_optimistic(path)
                probs = pmin_vec
            elif self.quantification == GameQuantification.COOPERATIVE:
                probs = pmax
            else:  # ADVERSARIAL (default)
                probs = pmax
        else:  # LEQ, LT
            if self.quantification == GameQuantification.P1_OPTIMISTIC:
                probs = pmax
            elif self.quantification == GameQuantification.P2_OPTIMISTIC:
                pmin_vec = self._path_probs_p2_optimistic(path)
                probs = pmin_vec
            elif self.quantification == GameQuantification.COOPERATIVE:
                probs = pmin
            else:  # ADVERSARIAL
                probs = pmax

        # Apply threshold comparison
        sat = set()
        for s in range(self.n):
            p = probs[s]
            if kind == FormulaKind.PROB_GEQ and p >= threshold - self.tol:
                sat.add(s)
            elif kind == FormulaKind.PROB_GT and p > threshold + self.tol:
                sat.add(s)
            elif kind == FormulaKind.PROB_LEQ and p <= threshold + self.tol:
                sat.add(s)
            elif kind == FormulaKind.PROB_LT and p < threshold - self.tol:
                sat.add(s)
        return sat

    # --------------------------------------------------------
    # Path probability computation: game value (P1 max, P2 min)
    # --------------------------------------------------------

    def _path_probs_game_value(self, path):
        """Compute game value probabilities: P1 maximizes, P2 minimizes."""
        kind = path.kind
        if kind == FormulaKind.NEXT:
            return self._next_probs_game(path.sub)
        elif kind == FormulaKind.UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._until_probs_game(phi_sat, psi_sat)
        elif kind == FormulaKind.BOUNDED_UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._bounded_until_probs_game(phi_sat, psi_sat, path.bound)
        else:
            raise ValueError(f"Unknown path formula kind: {kind}")

    def _path_probs_p2_optimistic(self, path):
        """Compute P2 optimistic: P2 minimizes, P1 also minimizes (worst case for property)."""
        kind = path.kind
        if kind == FormulaKind.NEXT:
            return self._next_probs_all_min(path.sub)
        elif kind == FormulaKind.UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._until_probs_all_min(phi_sat, psi_sat)
        elif kind == FormulaKind.BOUNDED_UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            return self._bounded_until_probs_all_min(phi_sat, psi_sat, path.bound)
        else:
            raise ValueError(f"Unknown path formula kind: {kind}")

    def _path_probs_cooperative(self, path, maximize=True):
        """Compute cooperative probabilities: both players cooperate."""
        kind = path.kind
        if kind == FormulaKind.NEXT:
            if maximize:
                return self._next_probs_all_max(path.sub)
            else:
                return self._next_probs_all_min(path.sub)
        elif kind == FormulaKind.UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            if maximize:
                return self._until_probs_all_max(phi_sat, psi_sat)
            else:
                return self._until_probs_all_min(phi_sat, psi_sat)
        elif kind == FormulaKind.BOUNDED_UNTIL:
            phi_sat = self.check(path.left)
            psi_sat = self.check(path.right)
            if maximize:
                return self._bounded_until_probs_all_max(phi_sat, psi_sat, path.bound)
            else:
                return self._bounded_until_probs_all_min(phi_sat, psi_sat, path.bound)
        else:
            raise ValueError(f"Unknown path formula kind: {kind}")

    # --------------------------------------------------------
    # Next-state formulas
    # --------------------------------------------------------

    def _next_probs_game(self, phi):
        """P1 max, P2 min for X phi."""
        phi_sat = self.check(phi)
        return self._next_probs_by_owner(phi_sat)

    def _next_probs_by_owner(self, phi_sat):
        """Compute next-state probs respecting state ownership."""
        result = [0.0] * self.n
        for s in range(self.n):
            actions = self.game.actions[s]
            if not actions:
                result[s] = 0.0
                continue
            vals = []
            for a_idx in range(len(actions)):
                p = sum(self.game.transition[s][a_idx][t] for t in phi_sat
                        if t < self.n)
                vals.append(p)
            owner = self.game.owners[s]
            if owner == Player.P1:
                result[s] = max(vals)
            elif owner == Player.P2:
                result[s] = min(vals)
            else:  # CHANCE
                # CHANCE has exactly one "action" -- expected value
                result[s] = vals[0] if vals else 0.0
        return result

    def _next_probs_all_max(self, phi):
        """All players maximize for X phi."""
        phi_sat = self.check(phi)
        result = [0.0] * self.n
        for s in range(self.n):
            actions = self.game.actions[s]
            if not actions:
                continue
            vals = []
            for a_idx in range(len(actions)):
                p = sum(self.game.transition[s][a_idx][t] for t in phi_sat
                        if t < self.n)
                vals.append(p)
            owner = self.game.owners[s]
            if owner == Player.CHANCE:
                result[s] = vals[0] if vals else 0.0
            else:
                result[s] = max(vals) if vals else 0.0
        return result

    def _next_probs_all_min(self, phi):
        """All players minimize for X phi."""
        phi_sat = self.check(phi)
        result = [0.0] * self.n
        for s in range(self.n):
            actions = self.game.actions[s]
            if not actions:
                continue
            vals = []
            for a_idx in range(len(actions)):
                p = sum(self.game.transition[s][a_idx][t] for t in phi_sat
                        if t < self.n)
                vals.append(p)
            owner = self.game.owners[s]
            if owner == Player.CHANCE:
                result[s] = vals[0] if vals else 0.0
            else:
                result[s] = min(vals) if vals else 0.0
        return result

    # --------------------------------------------------------
    # Until formulas: two-player value iteration
    # --------------------------------------------------------

    def _until_probs_game(self, phi_sat, psi_sat):
        """Game value for phi U psi: P1 maximizes, P2 minimizes."""
        return self._until_vi(phi_sat, psi_sat, p1_max=True, p2_max=False)

    def _until_probs_all_max(self, phi_sat, psi_sat):
        """All cooperate to maximize phi U psi."""
        return self._until_vi(phi_sat, psi_sat, p1_max=True, p2_max=True)

    def _until_probs_all_min(self, phi_sat, psi_sat):
        """All cooperate to minimize phi U psi."""
        return self._until_vi(phi_sat, psi_sat, p1_max=False, p2_max=False)

    def _until_vi(self, phi_sat, psi_sat, p1_max, p2_max):
        """Two-player value iteration for unbounded until.

        State classification:
        - S_yes = psi_sat (probability 1)
        - S_no = not in phi and not in psi (probability 0)
        - S_maybe = everything else

        For game value (p1_max=True, p2_max=False):
        P1 tries to maximize reachability, P2 tries to minimize.
        """
        probs = [0.0] * self.n
        for s in psi_sat:
            probs[s] = 1.0

        # States that can never reach psi
        s_no = set()
        for s in range(self.n):
            if s not in phi_sat and s not in psi_sat:
                s_no.add(s)

        # Value iteration
        for iteration in range(self.max_iter):
            new_probs = list(probs)
            max_diff = 0.0
            for s in range(self.n):
                if s in psi_sat:
                    new_probs[s] = 1.0
                    continue
                if s in s_no:
                    new_probs[s] = 0.0
                    continue
                if s not in phi_sat:
                    new_probs[s] = 0.0
                    continue

                actions = self.game.actions[s]
                if not actions:
                    new_probs[s] = 0.0
                    continue

                vals = []
                for a_idx in range(len(actions)):
                    ev = sum(self.game.transition[s][a_idx][t] * probs[t]
                             for t in range(self.n))
                    vals.append(ev)

                owner = self.game.owners[s]
                if owner == Player.P1:
                    new_probs[s] = max(vals) if p1_max else min(vals)
                elif owner == Player.P2:
                    new_probs[s] = max(vals) if p2_max else min(vals)
                else:  # CHANCE
                    new_probs[s] = vals[0] if vals else 0.0

                max_diff = max(max_diff, abs(new_probs[s] - probs[s]))

            probs = new_probs
            if max_diff < self.tol:
                break

        return probs

    # --------------------------------------------------------
    # Bounded until: two-player backward induction
    # --------------------------------------------------------

    def _bounded_until_probs_game(self, phi_sat, psi_sat, k):
        """Game value for phi U<=k psi."""
        return self._bounded_until_bi(phi_sat, psi_sat, k, p1_max=True, p2_max=False)

    def _bounded_until_probs_all_max(self, phi_sat, psi_sat, k):
        return self._bounded_until_bi(phi_sat, psi_sat, k, p1_max=True, p2_max=True)

    def _bounded_until_probs_all_min(self, phi_sat, psi_sat, k):
        return self._bounded_until_bi(phi_sat, psi_sat, k, p1_max=False, p2_max=False)

    def _bounded_until_bi(self, phi_sat, psi_sat, k, p1_max, p2_max):
        """Two-player backward induction for bounded until."""
        probs = [0.0] * self.n
        for s in psi_sat:
            probs[s] = 1.0

        for step in range(k):
            new_probs = [0.0] * self.n
            for s in range(self.n):
                if s in psi_sat:
                    new_probs[s] = 1.0
                    continue
                if s not in phi_sat:
                    new_probs[s] = 0.0
                    continue

                actions = self.game.actions[s]
                if not actions:
                    new_probs[s] = 0.0
                    continue

                vals = []
                for a_idx in range(len(actions)):
                    ev = sum(self.game.transition[s][a_idx][t] * probs[t]
                             for t in range(self.n))
                    vals.append(ev)

                owner = self.game.owners[s]
                if owner == Player.P1:
                    new_probs[s] = max(vals) if p1_max else min(vals)
                elif owner == Player.P2:
                    new_probs[s] = max(vals) if p2_max else min(vals)
                else:  # CHANCE
                    new_probs[s] = vals[0] if vals else 0.0

            probs = new_probs

        return probs

    # --------------------------------------------------------
    # Strategy extraction
    # --------------------------------------------------------

    def _extract_strategies(self, path, probs):
        """Extract P1 and P2 strategies from converged probability vector."""
        p1_strat = {}
        p2_strat = {}
        for s in range(self.n):
            actions = self.game.actions[s]
            if not actions:
                continue
            owner = self.game.owners[s]
            if owner == Player.CHANCE:
                continue

            # Evaluate each action
            vals = []
            for a_idx in range(len(actions)):
                ev = sum(self.game.transition[s][a_idx][t] * probs[t]
                         for t in range(self.n))
                vals.append(ev)

            if owner == Player.P1:
                best_idx = vals.index(max(vals))
                p1_strat[s] = best_idx
            elif owner == Player.P2:
                best_idx = vals.index(min(vals))
                p2_strat[s] = best_idx

        return p1_strat, p2_strat


# ============================================================
# Expected reward for games
# ============================================================

def game_expected_reward_pctl(lgame, rewards, target_formula,
                              maximize_p1=True, max_iter=10000, tol=1e-10):
    """Compute expected accumulated reward until target in a game.

    P1 tries to maximize (if maximize_p1=True) and P2 tries to minimize,
    or vice versa.

    Args:
        lgame: LabeledGame
        rewards: list of per-state rewards
        target_formula: PCTL formula identifying target states
        maximize_p1: if True, P1 maximizes reward; P2 minimizes
        max_iter, tol: convergence parameters

    Returns:
        (values, p1_strategy, p2_strategy) tuple
    """
    game = lgame.game
    n = game.n_states
    checker = GamePCTLChecker(lgame, tol=tol, max_iter=max_iter)
    target_states = checker.check(target_formula)

    values = [0.0] * n

    for iteration in range(max_iter):
        new_values = list(values)
        max_diff = 0.0
        for s in range(n):
            if s in target_states:
                new_values[s] = 0.0
                continue

            actions = game.actions[s]
            if not actions:
                new_values[s] = 0.0
                continue

            vals = []
            for a_idx in range(len(actions)):
                ev = rewards[s] + sum(game.transition[s][a_idx][t] * values[t]
                                      for t in range(n))
                vals.append(ev)

            owner = game.owners[s]
            if owner == Player.P1:
                new_values[s] = max(vals) if maximize_p1 else min(vals)
            elif owner == Player.P2:
                new_values[s] = min(vals) if maximize_p1 else max(vals)
            else:  # CHANCE
                new_values[s] = vals[0] if vals else 0.0

            max_diff = max(max_diff, abs(new_values[s] - values[s]))

        values = new_values
        if max_diff < tol:
            break

    # Extract strategies
    p1_strat = {}
    p2_strat = {}
    for s in range(n):
        if s in target_states:
            continue
        actions = game.actions[s]
        if not actions:
            continue
        owner = game.owners[s]
        if owner == Player.CHANCE:
            continue

        action_vals = []
        for a_idx in range(len(actions)):
            ev = rewards[s] + sum(game.transition[s][a_idx][t] * values[t]
                                  for t in range(n))
            action_vals.append(ev)

        if owner == Player.P1:
            best = action_vals.index(max(action_vals) if maximize_p1 else min(action_vals))
            p1_strat[s] = best
        elif owner == Player.P2:
            best = action_vals.index(min(action_vals) if maximize_p1 else max(action_vals))
            p2_strat[s] = best

    return values, p1_strat, p2_strat


# ============================================================
# High-level API functions
# ============================================================

def check_game_pctl(lgame, formula,
                    quantification=GameQuantification.ADVERSARIAL,
                    tol=1e-10, max_iter=10000):
    """Check PCTL formula against a labeled stochastic game.

    Args:
        lgame: LabeledGame
        formula: PCTL formula
        quantification: GameQuantification mode
        tol, max_iter: convergence parameters

    Returns:
        GamePCTLResult
    """
    checker = GamePCTLChecker(lgame, tol=tol, max_iter=max_iter,
                              quantification=quantification)
    sat = checker.check(formula)

    prob_max = None
    prob_min = None
    p1_strat = None
    p2_strat = None

    # If probability operator, compute quantitative results
    if formula.kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_LEQ,
                        FormulaKind.PROB_GT, FormulaKind.PROB_LT):
        path = formula.path
        prob_max = checker._path_probs_game_value(path)
        p1_strat, p2_strat = checker._extract_strategies(path, prob_max)

        if quantification == GameQuantification.COOPERATIVE:
            prob_min = checker._path_probs_cooperative(path, maximize=False)
        elif quantification == GameQuantification.P2_OPTIMISTIC:
            prob_min = checker._path_probs_p2_optimistic(path)

    return GamePCTLResult(
        formula=formula,
        satisfying_states=sat,
        all_states=lgame.game.n_states,
        prob_max=prob_max,
        prob_min=prob_min,
        quantification=quantification,
        state_labels=lgame.game.state_labels,
        p1_strategy=p1_strat,
        p2_strategy=p2_strat,
    )


def check_game_pctl_state(lgame, state, formula,
                          quantification=GameQuantification.ADVERSARIAL):
    """Check if a specific state satisfies a PCTL formula."""
    result = check_game_pctl(lgame, formula, quantification)
    return state in result.satisfying_states


def game_pctl_quantitative(lgame, path_formula):
    """Compute game value (max) and anti-value (all-min) probability vectors.

    Returns dict with 'game_value' (P1 max, P2 min) and 'all_min' (all minimize).
    """
    checker = GamePCTLChecker(lgame)
    game_val = checker._path_probs_game_value(path_formula)

    # Also compute all-cooperate-to-minimize
    all_min_checker = GamePCTLChecker(lgame)
    all_min = all_min_checker._path_probs_cooperative(path_formula, maximize=False)

    # And all-cooperate-to-maximize
    all_max = all_min_checker._path_probs_cooperative(path_formula, maximize=True)

    return {
        'game_value': game_val,
        'all_min': all_min,
        'all_max': all_max,
    }


def verify_game_property(lgame, formula, initial_state,
                         quantification=GameQuantification.ADVERSARIAL):
    """Verify a PCTL property at a specific initial state.

    Returns a verification dict with satisfied, probability info, strategies.
    """
    result = check_game_pctl(lgame, formula, quantification)
    satisfied = initial_state in result.satisfying_states

    info = {
        'satisfied': satisfied,
        'initial_state': initial_state,
        'formula': str(formula),
        'quantification': quantification.value,
        'satisfying_states': sorted(result.satisfying_states),
    }

    if result.prob_max is not None:
        info['prob_max'] = result.prob_max[initial_state]
        info['prob_max_all'] = result.prob_max
    if result.prob_min is not None:
        info['prob_min'] = result.prob_min[initial_state]
        info['prob_min_all'] = result.prob_min
    if result.p1_strategy is not None:
        info['p1_strategy'] = result.p1_strategy
    if result.p2_strategy is not None:
        info['p2_strategy'] = result.p2_strategy

    return info


def compare_quantifications(lgame, formula):
    """Compare adversarial vs cooperative vs P2-optimistic quantification modes."""
    results = {}
    for q in GameQuantification:
        r = check_game_pctl(lgame, formula, quantification=q)
        results[q.value] = {
            'satisfying_states': sorted(r.satisfying_states),
            'count': len(r.satisfying_states),
            'prob_max': r.prob_max,
            'prob_min': r.prob_min,
        }
    return results


def batch_check_game(lgame, formulas,
                     quantification=GameQuantification.ADVERSARIAL):
    """Check multiple PCTL formulas against the same game."""
    return [check_game_pctl(lgame, f, quantification) for f in formulas]


def induced_mc_comparison(lgame, formula):
    """Compare game PCTL result with induced Markov chain under extracted strategies.

    Extracts P1 and P2 strategies from game value computation,
    builds induced MC, checks the same formula on the MC.
    """
    # Game PCTL check
    game_result = check_game_pctl(lgame, formula)

    if game_result.p1_strategy is None or game_result.p2_strategy is None:
        return {
            'game_result': game_result,
            'note': 'No probability operator -- no strategy to extract',
        }

    # Build strategy pair
    strat = StrategyPair(
        p1_strategy=game_result.p1_strategy,
        p2_strategy=game_result.p2_strategy,
    )

    # Convert to MC
    mc = game_to_mc(lgame.game, strat)

    # Check PCTL on MC using V067
    from pctl_model_check import LabeledMC, PCTLChecker
    lmc = LabeledMC(mc=mc, labels=lgame.labels)
    mc_checker = PCTLChecker(lmc)
    mc_sat = mc_checker.check(formula)

    return {
        'game_satisfying': sorted(game_result.satisfying_states),
        'mc_satisfying': sorted(mc_sat),
        'game_prob_max': game_result.prob_max,
        'p1_strategy': game_result.p1_strategy,
        'p2_strategy': game_result.p2_strategy,
        'consistent': game_result.satisfying_states == mc_sat,
    }
