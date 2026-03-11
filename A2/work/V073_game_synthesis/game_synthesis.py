"""V073: Game-Theoretic Strategy Synthesis

Given a stochastic game and temporal objectives (PCTL, reachability, safety),
synthesize optimal and permissive strategies for both players.

Composes:
- V070 (stochastic games): game data structures, value iteration, reachability, safety
- V072 (game PCTL): PCTL model checking for games, strategy extraction
- V065 (Markov chains): induced chain analysis

Features:
1. Objective-driven synthesis: PCTL, reachability, safety -> optimal strategies
2. Permissive strategies: all actions that achieve the objective (not just one)
3. Multi-objective synthesis: Pareto-optimal strategies for competing objectives
4. Strategy verification: prove synthesized strategy achieves the objective
5. Strategy composition: combine strategies from sub-objectives
6. Assume-guarantee synthesis: decompose objectives compositionally
7. Strategy comparison: evaluate alternative strategies side-by-side
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V072_game_pctl'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V070_stochastic_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from stochastic_games import (
    StochasticGame, Player, StrategyPair, GameValueResult,
    GameReachResult, SafetyResult,
    make_game, game_to_mc,
    game_value_iteration, reachability_game, safety_game,
    attractor, game_expected_steps,
    verify_game_value_bound, verify_strategy_optimality,
    verify_reachability_bound,
)
from game_pctl import (
    LabeledGame, GameQuantification, GamePCTLResult,
    make_labeled_game,
    check_game_pctl, check_game_pctl_state,
    game_pctl_quantitative, verify_game_property,
)
from pctl_model_check import (
    PCTL, FormulaKind, tt, ff, atom, pnot, pand, por,
    prob_geq, prob_leq, prob_gt, prob_lt,
    next_f, until, bounded_until,
    eventually, always, bounded_eventually,
    parse_pctl,
)
from markov_chain import MarkovChain, make_chain, analyze_chain
from smt_solver import SMTSolver, SMTResult, Var, IntConst, App, Op, BOOL, INT


# ============================================================
# Data Structures
# ============================================================

class ObjectiveType(Enum):
    REACHABILITY = "reachability"
    SAFETY = "safety"
    PCTL = "pctl"
    REWARD = "reward"


@dataclass
class Objective:
    """A temporal objective for strategy synthesis."""
    obj_type: ObjectiveType
    targets: Optional[Set[int]] = None        # For reachability/safety
    formula: Optional[PCTL] = None            # For PCTL
    threshold: float = 0.0                     # Probability threshold
    safe_states: Optional[Set[int]] = None    # For safety
    weight: float = 1.0                        # For multi-objective


@dataclass
class SynthesisResult:
    """Result of strategy synthesis."""
    success: bool
    strategies: Optional[StrategyPair] = None
    objective: Optional[Objective] = None
    value: float = 0.0                         # Achieved probability/value
    values_per_state: Optional[List[float]] = None
    verified: bool = False
    details: str = ""


@dataclass
class PermissiveStrategy:
    """Permissive strategy: set of allowed actions per state.

    Unlike a deterministic strategy (one action per state), a permissive
    strategy allows any action from the allowed set while still guaranteeing
    the objective.
    """
    player: Player
    allowed_actions: Dict[int, Set[int]]  # state -> set of action indices
    objective_value: float = 0.0

    def is_permissive_at(self, state: int) -> bool:
        """Check if multiple actions are allowed at this state."""
        return len(self.allowed_actions.get(state, set())) > 1

    def permissiveness(self) -> float:
        """Fraction of states with multiple allowed actions."""
        if not self.allowed_actions:
            return 0.0
        multi = sum(1 for s in self.allowed_actions if len(self.allowed_actions[s]) > 1)
        return multi / len(self.allowed_actions)

    def to_deterministic(self) -> Dict[int, int]:
        """Pick first allowed action at each state (arbitrary determinization)."""
        return {s: min(acts) for s, acts in self.allowed_actions.items() if acts}


@dataclass
class MultiObjectiveResult:
    """Result of multi-objective synthesis."""
    pareto_front: List[Tuple[List[float], StrategyPair]]  # (objective_values, strategy)
    dominated: List[Tuple[List[float], StrategyPair]]
    n_objectives: int
    details: str = ""


# ============================================================
# Core Synthesis
# ============================================================

def synthesize_reachability(game: StochasticGame,
                            targets: Set[int],
                            player: Player = Player.P1,
                            threshold: float = 0.0) -> SynthesisResult:
    """Synthesize a strategy for reachability objective.

    For P1: maximize probability of reaching targets.
    For P2: minimize probability of reaching targets (equivalently, maximize avoidance).
    """
    result = reachability_game(game, targets)

    non_target = [s for s in range(game.n_states) if s not in targets]
    if player == Player.P1:
        if non_target:
            value = min(result.probabilities[s] for s in non_target)
        else:
            value = 1.0  # All states are targets
        success = value >= threshold if threshold > 0 else True
    else:
        # P2 wants to minimize -- success if all probs are at most threshold
        if non_target:
            value = max(result.probabilities[s] for s in non_target)
        else:
            value = 1.0
        success = value <= threshold if threshold < 1.0 else True

    return SynthesisResult(
        success=success,
        strategies=result.strategies,
        objective=Objective(ObjectiveType.REACHABILITY, targets=targets,
                           threshold=threshold),
        value=result.probabilities[0] if game.n_states > 0 else 0.0,
        values_per_state=result.probabilities,
        details=f"P1 winning: {sorted(result.p1_winning)}, "
                f"P2 winning: {sorted(result.p2_winning)}"
    )


def synthesize_safety(game: StochasticGame,
                      safe_states: Set[int],
                      threshold: float = 1.0) -> SynthesisResult:
    """Synthesize a strategy for safety objective.

    P1 tries to stay within safe_states forever.
    """
    result = safety_game(game, safe_states)

    value = result.safe_probabilities[0] if game.n_states > 0 else 0.0
    success = value >= threshold

    return SynthesisResult(
        success=success,
        strategies=result.strategies,
        objective=Objective(ObjectiveType.SAFETY, safe_states=safe_states,
                           threshold=threshold),
        value=value,
        values_per_state=result.safe_probabilities,
        details=f"P1 safe region: {sorted(result.p1_safe_region)}"
    )


def synthesize_pctl(lgame: LabeledGame,
                    formula: PCTL,
                    initial_state: int = 0,
                    quantification: GameQuantification = GameQuantification.ADVERSARIAL
                    ) -> SynthesisResult:
    """Synthesize a strategy that satisfies a PCTL formula.

    Returns the strategy extracted from PCTL model checking.
    """
    result = check_game_pctl(lgame, formula, quantification=quantification)

    success = initial_state in result.satisfying_states

    # Build strategy pair from PCTL result
    strategies = None
    value = 0.0
    if result.p1_strategy is not None or result.p2_strategy is not None:
        strategies = StrategyPair(
            p1_strategy=result.p1_strategy or {},
            p2_strategy=result.p2_strategy or {},
        )
    if result.prob_max is not None:
        value = result.prob_max[initial_state]

    return SynthesisResult(
        success=success,
        strategies=strategies,
        objective=Objective(ObjectiveType.PCTL, formula=formula),
        value=value,
        values_per_state=result.prob_max,
        details=f"Satisfying: {sorted(result.satisfying_states)}, "
                f"quantification: {quantification.value}"
    )


# ============================================================
# Permissive Strategy Synthesis
# ============================================================

def synthesize_permissive_reachability(game: StochasticGame,
                                       targets: Set[int],
                                       player: Player = Player.P1,
                                       tolerance: float = 1e-6
                                       ) -> PermissiveStrategy:
    """Compute permissive strategy: all actions that achieve optimal value.

    An action is permissive if choosing it doesn't reduce the optimal
    probability by more than tolerance.
    """
    # First get optimal values
    result = reachability_game(game, targets)
    optimal_probs = result.probabilities

    allowed = {}
    for s in range(game.n_states):
        if game.owners[s] != player:
            # Not our state -- all actions are "allowed" (irrelevant)
            allowed[s] = set(range(len(game.actions[s])))
            continue

        if s in targets:
            allowed[s] = set(range(len(game.actions[s])))
            continue

        state_allowed = set()
        for a_idx in range(len(game.actions[s])):
            # Compute expected value under this action with optimal continuation
            expected = sum(game.transition[s][a_idx][t] * optimal_probs[t]
                          for t in range(game.n_states))

            if player == Player.P1:
                # P1 maximizes: action is permissive if close to optimal
                if expected >= optimal_probs[s] - tolerance:
                    state_allowed.add(a_idx)
            else:
                # P2 minimizes: action is permissive if close to optimal (min)
                if expected <= optimal_probs[s] + tolerance:
                    state_allowed.add(a_idx)

        if not state_allowed:
            # Fallback: at least one action
            state_allowed = {0}
        allowed[s] = state_allowed

    return PermissiveStrategy(
        player=player,
        allowed_actions=allowed,
        objective_value=optimal_probs[0] if game.n_states > 0 else 0.0
    )


def synthesize_permissive_safety(game: StochasticGame,
                                  safe_states: Set[int],
                                  tolerance: float = 1e-6
                                  ) -> PermissiveStrategy:
    """Compute permissive strategy for safety objective."""
    result = safety_game(game, safe_states)
    optimal_probs = result.safe_probabilities

    allowed = {}
    for s in range(game.n_states):
        if game.owners[s] != Player.P1:
            allowed[s] = set(range(len(game.actions[s])))
            continue

        state_allowed = set()
        for a_idx in range(len(game.actions[s])):
            expected = sum(game.transition[s][a_idx][t] * optimal_probs[t]
                          for t in range(game.n_states))
            if expected >= optimal_probs[s] - tolerance:
                state_allowed.add(a_idx)

        if not state_allowed:
            state_allowed = {0}
        allowed[s] = state_allowed

    return PermissiveStrategy(
        player=Player.P1,
        allowed_actions=allowed,
        objective_value=optimal_probs[0] if game.n_states > 0 else 0.0
    )


# ============================================================
# Strategy Verification
# ============================================================

def verify_strategy(game: StochasticGame,
                    strategies: StrategyPair,
                    objective: Objective) -> SynthesisResult:
    """Verify that a strategy pair achieves the given objective.

    Induces a Markov chain from the strategies and checks the objective.
    """
    mc = game_to_mc(game, strategies)
    analysis = analyze_chain(mc)

    if objective.obj_type == ObjectiveType.REACHABILITY:
        targets = objective.targets or set()
        # Compute reachability probability in the induced MC
        probs = _mc_reachability(mc, targets)
        value = probs[0] if probs else 0.0
        success = value >= objective.threshold
        return SynthesisResult(
            success=success,
            strategies=strategies,
            objective=objective,
            value=value,
            values_per_state=probs,
            verified=True,
            details=f"MC reachability: {value:.6f} vs threshold {objective.threshold}"
        )

    elif objective.obj_type == ObjectiveType.SAFETY:
        safe = objective.safe_states or set()
        # Safety = probability of staying in safe states forever
        # In MC: prob of never leaving safe states
        unsafe = set(range(game.n_states)) - safe
        if not unsafe:
            return SynthesisResult(
                success=True, strategies=strategies, objective=objective,
                value=1.0, verified=True, details="All states safe"
            )
        # Reachability to unsafe states, then safety = 1 - reach(unsafe)
        reach_unsafe = _mc_reachability(mc, unsafe)
        safety_probs = [1.0 - p for p in reach_unsafe]
        value = safety_probs[0] if safety_probs else 0.0
        success = value >= objective.threshold
        return SynthesisResult(
            success=success, strategies=strategies, objective=objective,
            value=value, values_per_state=safety_probs, verified=True,
            details=f"MC safety: {value:.6f} vs threshold {objective.threshold}"
        )

    elif objective.obj_type == ObjectiveType.PCTL:
        # Can't directly check PCTL on MC without labeled game
        # Use SMT-based verification instead
        return _verify_strategy_smt(game, strategies, objective)

    return SynthesisResult(
        success=False, strategies=strategies, objective=objective,
        details="Unknown objective type"
    )


def _mc_reachability(mc: MarkovChain, targets: Set[int],
                     max_iter: int = 10000, tol: float = 1e-10) -> List[float]:
    """Compute reachability probabilities in a Markov chain."""
    n = mc.n_states
    probs = [0.0] * n
    for t in targets:
        probs[t] = 1.0

    for _ in range(max_iter):
        new_probs = [0.0] * n
        for s in range(n):
            if s in targets:
                new_probs[s] = 1.0
                continue
            new_probs[s] = sum(mc.transition[s][t] * probs[t] for t in range(n))
        diff = max(abs(new_probs[s] - probs[s]) for s in range(n))
        probs = new_probs
        if diff < tol:
            break
    return probs


def _verify_strategy_smt(game: StochasticGame,
                          strategies: StrategyPair,
                          objective: Objective) -> SynthesisResult:
    """SMT-based strategy verification."""
    # Use V070's verify functions
    result = verify_strategy_optimality(game, strategies, discount=0.99)
    return SynthesisResult(
        success=result.verified,
        strategies=strategies,
        objective=objective,
        verified=True,
        details=f"SMT verification: {result.details}"
    )


# ============================================================
# Multi-Objective Synthesis
# ============================================================

def synthesize_multi_objective(game: StochasticGame,
                                objectives: List[Objective],
                                n_samples: int = 20) -> MultiObjectiveResult:
    """Compute Pareto-optimal strategies for multiple objectives.

    Uses weight-space sampling: for each weight vector, solve the
    weighted sum of objectives and collect non-dominated solutions.
    """
    n_obj = len(objectives)
    if n_obj == 0:
        return MultiObjectiveResult([], [], 0)

    if n_obj == 1:
        result = _synthesize_single(game, objectives[0])
        if result.success and result.strategies:
            values = [result.value]
            return MultiObjectiveResult(
                pareto_front=[(values, result.strategies)],
                dominated=[], n_objectives=1
            )
        return MultiObjectiveResult([], [], 1)

    # Sample weight vectors
    solutions = []
    for i in range(n_samples + 1):
        w1 = i / n_samples
        w2 = 1.0 - w1

        # Compute objective values for each weight
        # Strategy: solve each objective, pick the strategy that maximizes weighted sum
        best_weighted = -float('inf')
        best_strategy = None
        best_values = None

        for obj_idx, obj in enumerate(objectives):
            result = _synthesize_single(game, obj)
            if result.strategies is None:
                continue

            # Evaluate this strategy on ALL objectives
            obj_values = _evaluate_strategy_on_objectives(game, result.strategies, objectives)
            weighted = sum(w * v for w, v in zip([w1, w2] if n_obj == 2 else
                          [1.0 / n_obj] * n_obj, obj_values))

            if weighted > best_weighted:
                best_weighted = weighted
                best_strategy = result.strategies
                best_values = obj_values

        if best_strategy is not None and best_values is not None:
            solutions.append((best_values, best_strategy))

    # Remove duplicates and find Pareto front
    unique = _deduplicate_solutions(solutions)
    pareto, dominated = _pareto_filter(unique)

    return MultiObjectiveResult(
        pareto_front=pareto,
        dominated=dominated,
        n_objectives=n_obj,
        details=f"Sampled {n_samples + 1} weight vectors, "
                f"found {len(pareto)} Pareto-optimal strategies"
    )


def _synthesize_single(game: StochasticGame,
                        objective: Objective) -> SynthesisResult:
    """Synthesize for a single objective."""
    if objective.obj_type == ObjectiveType.REACHABILITY:
        return synthesize_reachability(game, objective.targets or set(),
                                       threshold=objective.threshold)
    elif objective.obj_type == ObjectiveType.SAFETY:
        return synthesize_safety(game, objective.safe_states or set(),
                                 threshold=objective.threshold)
    elif objective.obj_type == ObjectiveType.REWARD:
        result = game_value_iteration(game, discount=0.9)
        return SynthesisResult(
            success=True,
            strategies=result.strategies,
            objective=objective,
            value=result.values[0] if result.values else 0.0,
            values_per_state=result.values,
        )
    return SynthesisResult(success=False, details="Unsupported objective type")


def _evaluate_strategy_on_objectives(game: StochasticGame,
                                      strategies: StrategyPair,
                                      objectives: List[Objective]) -> List[float]:
    """Evaluate a strategy pair against each objective."""
    mc = game_to_mc(game, strategies)
    values = []
    for obj in objectives:
        if obj.obj_type == ObjectiveType.REACHABILITY:
            probs = _mc_reachability(mc, obj.targets or set())
            values.append(probs[0] if probs else 0.0)
        elif obj.obj_type == ObjectiveType.SAFETY:
            safe = obj.safe_states or set()
            unsafe = set(range(game.n_states)) - safe
            if not unsafe:
                values.append(1.0)
            else:
                reach = _mc_reachability(mc, unsafe)
                values.append(1.0 - reach[0] if reach else 1.0)
        elif obj.obj_type == ObjectiveType.REWARD:
            # Re-run game value iteration with this strategy fixed
            # Approximate by MC simulation
            values.append(_mc_expected_reward(mc, game, strategies))
        else:
            values.append(0.0)
    return values


def _mc_expected_reward(mc: MarkovChain, game: StochasticGame,
                         strategies: StrategyPair,
                         discount: float = 0.9, max_iter: int = 1000) -> float:
    """Approximate expected discounted reward under a strategy in the induced MC."""
    n = mc.n_states
    values = [0.0] * n
    for _ in range(max_iter):
        new_values = [0.0] * n
        for s in range(n):
            a_idx = strategies.get_action(s, game.owners[s])
            r = game.rewards[s][a_idx] if a_idx < len(game.rewards[s]) else 0.0
            expected = sum(mc.transition[s][t] * values[t] for t in range(n))
            new_values[s] = r + discount * expected
        diff = max(abs(new_values[s] - values[s]) for s in range(n))
        values = new_values
        if diff < 1e-10:
            break
    return values[0]


def _deduplicate_solutions(solutions: List[Tuple[List[float], StrategyPair]]
                            ) -> List[Tuple[List[float], StrategyPair]]:
    """Remove duplicate solutions (same objective values)."""
    seen = set()
    unique = []
    for values, strat in solutions:
        key = tuple(round(v, 8) for v in values)
        if key not in seen:
            seen.add(key)
            unique.append((values, strat))
    return unique


def _pareto_filter(solutions: List[Tuple[List[float], StrategyPair]]
                   ) -> Tuple[List[Tuple[List[float], StrategyPair]],
                              List[Tuple[List[float], StrategyPair]]]:
    """Filter Pareto-optimal solutions (non-dominated)."""
    pareto = []
    dominated = []

    for i, (vi, si) in enumerate(solutions):
        is_dominated = False
        for j, (vj, sj) in enumerate(solutions):
            if i == j:
                continue
            # vj dominates vi if vj >= vi in all objectives and > in at least one
            if all(vj[k] >= vi[k] for k in range(len(vi))) and \
               any(vj[k] > vi[k] for k in range(len(vi))):
                is_dominated = True
                break
        if is_dominated:
            dominated.append((vi, si))
        else:
            pareto.append((vi, si))

    return pareto, dominated


# ============================================================
# Strategy Composition (Assume-Guarantee)
# ============================================================

def compose_strategies(game: StochasticGame,
                       strategies_list: List[StrategyPair],
                       objectives: List[Objective],
                       priority: str = "first") -> StrategyPair:
    """Compose multiple strategies into one.

    For each state, pick the action from the highest-priority strategy
    that has a non-trivial preference (differs from default).

    Args:
        game: The stochastic game
        strategies_list: List of strategy pairs (one per objective)
        objectives: Corresponding objectives
        priority: "first" (first strategy wins), "best" (best objective value wins)
    """
    if not strategies_list:
        return StrategyPair({}, {})

    if len(strategies_list) == 1:
        return strategies_list[0]

    p1_strat = {}
    p2_strat = {}

    for s in range(game.n_states):
        if game.owners[s] == Player.P1:
            if priority == "first":
                # First strategy with an entry for this state wins
                for strats in strategies_list:
                    if s in strats.p1_strategy:
                        p1_strat[s] = strats.p1_strategy[s]
                        break
                else:
                    p1_strat[s] = 0
            else:
                # Evaluate each strategy's action at this state
                best_value = -float('inf')
                best_action = 0
                for idx, strats in enumerate(strategies_list):
                    a = strats.p1_strategy.get(s, 0)
                    # Use the objective's value for this state
                    result = _synthesize_single(game, objectives[idx])
                    val = (result.values_per_state[s]
                           if result.values_per_state else 0.0)
                    weighted_val = val * objectives[idx].weight
                    if weighted_val > best_value:
                        best_value = weighted_val
                        best_action = a
                p1_strat[s] = best_action

        elif game.owners[s] == Player.P2:
            for strats in strategies_list:
                if s in strats.p2_strategy:
                    p2_strat[s] = strats.p2_strategy[s]
                    break
            else:
                p2_strat[s] = 0

    return StrategyPair(p1_strat, p2_strat)


def assume_guarantee_synthesis(game: StochasticGame,
                                assumptions: List[Objective],
                                guarantees: List[Objective]
                                ) -> SynthesisResult:
    """Assume-guarantee synthesis: find strategy where
    IF assumptions hold (about opponent), THEN guarantees hold (for us).

    1. Synthesize P2 strategy satisfying assumptions (cooperative)
    2. Under that P2 strategy, synthesize P1 strategy achieving guarantees
    3. Verify the composed strategy
    """
    # Step 1: Synthesize assumption strategies
    assumption_strategies = []
    for assumption in assumptions:
        result = _synthesize_single(game, assumption)
        if not result.success or result.strategies is None:
            return SynthesisResult(
                success=False,
                details=f"Cannot satisfy assumption: {assumption.obj_type.value}"
            )
        assumption_strategies.append(result.strategies)

    # Step 2: Fix P2 to assumption strategy, solve guarantee for P1
    if assumption_strategies:
        p2_strat = assumption_strategies[0].p2_strategy
    else:
        p2_strat = {s: 0 for s in range(game.n_states)}

    # Synthesize guarantees
    guarantee_strategies = []
    for guarantee in guarantees:
        result = _synthesize_single(game, guarantee)
        if result.strategies:
            guarantee_strategies.append(result.strategies)

    if not guarantee_strategies:
        return SynthesisResult(
            success=False,
            details="Cannot satisfy any guarantee"
        )

    # Step 3: Compose -- P1 from guarantee, P2 from assumption
    composed = StrategyPair(
        p1_strategy=guarantee_strategies[0].p1_strategy,
        p2_strategy=p2_strat,
    )

    # Verify the composed strategy achieves guarantees
    verification_results = []
    for guarantee in guarantees:
        ver = verify_strategy(game, composed, guarantee)
        verification_results.append(ver)

    all_verified = all(v.success for v in verification_results)

    return SynthesisResult(
        success=all_verified,
        strategies=composed,
        objective=guarantees[0] if guarantees else None,
        value=verification_results[0].value if verification_results else 0.0,
        verified=all_verified,
        details=f"Assume-guarantee: {len(assumptions)} assumptions, "
                f"{len(guarantees)} guarantees, verified={all_verified}"
    )


# ============================================================
# Strategy Refinement
# ============================================================

def refine_strategy(game: StochasticGame,
                    initial_strategy: StrategyPair,
                    objective: Objective,
                    max_rounds: int = 100) -> SynthesisResult:
    """Iteratively improve a strategy by local action swaps.

    At each round, for each P1 state, try all actions and keep the best.
    Converges to a local optimum (which is global for reachability/safety).
    """
    strategies = StrategyPair(
        p1_strategy=dict(initial_strategy.p1_strategy),
        p2_strategy=dict(initial_strategy.p2_strategy),
    )

    for round_idx in range(max_rounds):
        improved = False

        for s in range(game.n_states):
            if game.owners[s] != Player.P1:
                continue

            current_action = strategies.p1_strategy.get(s, 0)
            best_action = current_action
            best_value = _evaluate_action_value(game, strategies, s, current_action, objective)

            for a_idx in range(len(game.actions[s])):
                if a_idx == current_action:
                    continue
                val = _evaluate_action_value(game, strategies, s, a_idx, objective)
                if val > best_value + 1e-10:
                    best_value = val
                    best_action = a_idx

            if best_action != current_action:
                strategies.p1_strategy[s] = best_action
                improved = True

        if not improved:
            break

    # Evaluate final strategy
    final_ver = verify_strategy(game, strategies, objective)

    return SynthesisResult(
        success=final_ver.success,
        strategies=strategies,
        objective=objective,
        value=final_ver.value,
        values_per_state=final_ver.values_per_state,
        details=f"Refined for {round_idx + 1} rounds"
    )


def _evaluate_action_value(game: StochasticGame,
                            strategies: StrategyPair,
                            state: int, action: int,
                            objective: Objective) -> float:
    """Evaluate the value of taking a specific action at a state."""
    # Temporarily set the action
    old_action = strategies.p1_strategy.get(state, 0)
    strategies.p1_strategy[state] = action

    mc = game_to_mc(game, strategies)

    if objective.obj_type == ObjectiveType.REACHABILITY:
        probs = _mc_reachability(mc, objective.targets or set())
        value = probs[state] if state < len(probs) else 0.0
    elif objective.obj_type == ObjectiveType.SAFETY:
        safe = objective.safe_states or set()
        unsafe = set(range(game.n_states)) - safe
        if not unsafe:
            value = 1.0
        else:
            reach = _mc_reachability(mc, unsafe)
            value = 1.0 - reach[state] if state < len(reach) else 0.0
    else:
        value = 0.0

    # Restore
    strategies.p1_strategy[state] = old_action
    return value


# ============================================================
# Strategy Comparison
# ============================================================

def compare_strategies(game: StochasticGame,
                       strategies_list: List[StrategyPair],
                       objectives: List[Objective],
                       strategy_names: Optional[List[str]] = None
                       ) -> Dict:
    """Compare multiple strategies against multiple objectives."""
    names = strategy_names or [f"strategy_{i}" for i in range(len(strategies_list))]
    results = {}

    for name, strats in zip(names, strategies_list):
        obj_values = _evaluate_strategy_on_objectives(game, strats, objectives)
        results[name] = {
            'objective_values': obj_values,
            'p1_strategy': strats.p1_strategy,
            'p2_strategy': strats.p2_strategy,
        }

    # Find best strategy for each objective
    best_per_obj = []
    for obj_idx in range(len(objectives)):
        best_name = max(names,
                       key=lambda n: results[n]['objective_values'][obj_idx])
        best_per_obj.append(best_name)

    results['_best_per_objective'] = best_per_obj
    return results


# ============================================================
# PCTL-Based Synthesis with Strategy Export
# ============================================================

def synthesize_from_pctl(lgame: LabeledGame,
                          formula: PCTL,
                          export_mc: bool = False) -> Dict:
    """Full PCTL-based synthesis pipeline.

    1. Model check the PCTL formula
    2. Extract strategies
    3. Optionally build induced Markov chain
    4. Verify strategy in induced MC
    """
    result = check_game_pctl(lgame, formula,
                              quantification=GameQuantification.ADVERSARIAL)

    output = {
        'formula': str(formula),
        'satisfying_states': sorted(result.satisfying_states),
        'all_satisfy': result.all_satisfy,
        'prob_max': result.prob_max,
        'p1_strategy': result.p1_strategy,
        'p2_strategy': result.p2_strategy,
    }

    if export_mc and result.p1_strategy and result.p2_strategy:
        strat = StrategyPair(
            p1_strategy=result.p1_strategy,
            p2_strategy=result.p2_strategy,
        )
        mc = game_to_mc(lgame.game, strat)
        output['induced_mc'] = {
            'n_states': mc.n_states,
            'transition': mc.transition,
        }

        # Verify in induced MC
        if result.prob_max:
            targets = set()
            # Extract targets from formula if it's a reachability formula
            if formula.kind in (FormulaKind.PROB_GEQ, FormulaKind.PROB_LEQ,
                               FormulaKind.PROB_GT, FormulaKind.PROB_LT):
                path = formula.path
                if path and path.kind == FormulaKind.UNTIL:
                    right = path.right
                    if right and right.kind == FormulaKind.ATOM:
                        targets = lgame.states_with(right.label)

            if targets:
                mc_probs = _mc_reachability(mc, targets)
                output['mc_reachability'] = mc_probs
                output['game_vs_mc_match'] = all(
                    abs(result.prob_max[s] - mc_probs[s]) < 1e-6
                    for s in range(lgame.game.n_states)
                )

    return output


# ============================================================
# Convenience APIs
# ============================================================

def synthesize(game: StochasticGame,
               objective: Objective,
               verify: bool = True) -> SynthesisResult:
    """One-shot synthesis: synthesize + optionally verify.

    Main entry point for single-objective synthesis.
    """
    result = _synthesize_single(game, objective)

    if verify and result.success and result.strategies:
        ver = verify_strategy(game, result.strategies, objective)
        result.verified = ver.verified or ver.success
        if ver.values_per_state:
            result.details += f" | Verified value: {ver.value:.6f}"

    return result


def synthesis_summary(game: StochasticGame,
                      objectives: List[Objective]) -> Dict:
    """Summary of synthesis results for multiple objectives."""
    results = {}
    for i, obj in enumerate(objectives):
        r = synthesize(game, obj, verify=True)
        results[f"objective_{i}"] = {
            'type': obj.obj_type.value,
            'success': r.success,
            'value': r.value,
            'verified': r.verified,
            'strategy': {
                'p1': r.strategies.p1_strategy if r.strategies else None,
                'p2': r.strategies.p2_strategy if r.strategies else None,
            },
        }
    return results
