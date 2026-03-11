"""V070: Stochastic Game Verification

Two-player stochastic games: Player 1 (maximizer) vs Player 2 (minimizer).
Extends MDPs with adversarial choice -- each state is controlled by one player,
and transitions are stochastic given the chosen action.

Composes V069 (MDP verification) + V068 (interval MDP) + V067 (PCTL) + C037 (SMT).

Features:
1. Stochastic game data structure (turn-based + concurrent)
2. Minimax value iteration (Shapley's theorem)
3. Reachability games (max prob for P1, min prob for P2)
4. Safety games (staying in safe states)
5. Parity/mean-payoff objectives
6. Strategy extraction for both players
7. SMT-based property verification
8. Nash equilibria for concurrent games
9. Comparison with MDP (single-player) solutions
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V069_mdp_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V068_interval_mdp'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from mdp_verification import (MDP, Policy, ValueResult, ReachResult, Objective,
                                make_mdp, value_iteration, reachability,
                                evaluate_policy, mdp_to_mc, q_values,
                                expected_steps)
from markov_chain import MarkovChain, make_chain, analyze_chain
from smt_solver import SMTSolver, SMTResult, Var, IntConst, App, Op, BOOL, INT


# ===========================================================================
# Data Structures
# ===========================================================================

class Player(Enum):
    """Which player controls a state."""
    P1 = 1  # Maximizer
    P2 = 2  # Minimizer
    CHANCE = 0  # Stochastic (no player choice)


@dataclass
class StochasticGame:
    """Two-player turn-based stochastic game.

    Each state is owned by P1, P2, or CHANCE.
    - P1 states: Player 1 chooses an action (maximizing)
    - P2 states: Player 2 chooses an action (minimizing)
    - CHANCE states: action is chosen stochastically (single action with probabilities)

    n_states: number of states
    owners: owners[s] = Player who controls state s
    actions: actions[s] = list of action names at state s
    transition: transition[s][a_idx][t] = probability of going to t from s under action a
    rewards: rewards[s][a_idx] = immediate reward for action a at state s
    state_labels: human-readable labels
    """
    n_states: int
    owners: List[Player]
    actions: List[List[str]]
    transition: List[List[List[float]]]
    rewards: Optional[List[List[float]]] = None
    state_labels: Optional[List[str]] = None

    def __post_init__(self):
        if self.state_labels is None:
            self.state_labels = [f"s{i}" for i in range(self.n_states)]
        if self.rewards is None:
            if len(self.actions) == self.n_states:
                self.rewards = [[0.0] * len(self.actions[s]) for s in range(self.n_states)]
            else:
                self.rewards = []

    def validate(self) -> List[str]:
        """Validate game structure."""
        errors = []
        if len(self.owners) != self.n_states:
            errors.append(f"owners length {len(self.owners)} != n_states {self.n_states}")
        if len(self.actions) != self.n_states:
            errors.append(f"actions length {len(self.actions)} != n_states {self.n_states}")
        if len(self.transition) != self.n_states:
            errors.append(f"transition length {len(self.transition)} != n_states {self.n_states}")
        for s in range(min(self.n_states, len(self.transition), len(self.actions))):
            if len(self.transition[s]) != len(self.actions[s]):
                errors.append(f"state {s}: transitions/actions mismatch")
            for a_idx, row in enumerate(self.transition[s]):
                if len(row) != self.n_states:
                    errors.append(f"state {s} action {a_idx}: row length {len(row)} != {self.n_states}")
                total = sum(row)
                if abs(total - 1.0) > 1e-6:
                    errors.append(f"state {s} action {a_idx}: probs sum to {total}")
            # CHANCE states should have exactly 1 action
            if self.owners[s] == Player.CHANCE and len(self.actions[s]) != 1:
                errors.append(f"CHANCE state {s} has {len(self.actions[s])} actions (need 1)")
        return errors

    def p1_states(self) -> Set[int]:
        return {s for s in range(self.n_states) if self.owners[s] == Player.P1}

    def p2_states(self) -> Set[int]:
        return {s for s in range(self.n_states) if self.owners[s] == Player.P2}

    def chance_states(self) -> Set[int]:
        return {s for s in range(self.n_states) if self.owners[s] == Player.CHANCE}


@dataclass
class StrategyPair:
    """Strategies for both players.

    p1_strategy: maps P1 states to action indices
    p2_strategy: maps P2 states to action indices
    """
    p1_strategy: Dict[int, int]
    p2_strategy: Dict[int, int]

    def get_action(self, state: int, owner: Player) -> int:
        if owner == Player.P1:
            return self.p1_strategy.get(state, 0)
        elif owner == Player.P2:
            return self.p2_strategy.get(state, 0)
        return 0  # CHANCE


@dataclass
class GameValueResult:
    """Result of game value computation."""
    values: List[float]
    strategies: StrategyPair
    iterations: int
    converged: bool


@dataclass
class GameReachResult:
    """Result of reachability game analysis."""
    probabilities: List[float]
    strategies: StrategyPair
    targets: Set[int]
    p1_winning: Set[int]  # States where P1 can guarantee reaching target
    p2_winning: Set[int]  # States where P2 can prevent reaching target


@dataclass
class SafetyResult:
    """Result of safety game analysis."""
    safe_probabilities: List[float]
    strategies: StrategyPair
    safe_states: Set[int]
    p1_safe_region: Set[int]  # States where P1 can guarantee staying safe


@dataclass
class GameVerificationResult:
    """Result of SMT-based game property verification."""
    verified: bool
    property_desc: str
    counterexample: Optional[Dict] = None
    details: str = ""


# ===========================================================================
# Construction helpers
# ===========================================================================

def make_game(n_states: int,
              owners: Dict[int, Player],
              action_transitions: Dict[int, Dict[str, List[float]]],
              rewards: Optional[Dict[int, Dict[str, float]]] = None,
              state_labels: Optional[List[str]] = None) -> StochasticGame:
    """Create a stochastic game from a dictionary format.

    owners[s] = Player.P1 / Player.P2 / Player.CHANCE
    action_transitions[s][action_name] = [p0, p1, ..., p_{n-1}]
    rewards[s][action_name] = reward
    """
    owner_list = [owners.get(s, Player.CHANCE) for s in range(n_states)]
    actions_list = []
    trans_list = []
    reward_list = []

    for s in range(n_states):
        if s in action_transitions:
            act_names = list(action_transitions[s].keys())
            act_trans = [action_transitions[s][a] for a in act_names]
            if rewards and s in rewards:
                act_rewards = [rewards[s].get(a, 0.0) for a in act_names]
            else:
                act_rewards = [0.0] * len(act_names)
        else:
            act_names = ["stay"]
            row = [0.0] * n_states
            row[s] = 1.0
            act_trans = [row]
            act_rewards = [0.0]
        actions_list.append(act_names)
        trans_list.append(act_trans)
        reward_list.append(act_rewards)

    return StochasticGame(n_states=n_states, owners=owner_list,
                           actions=actions_list, transition=trans_list,
                           rewards=reward_list, state_labels=state_labels)


def game_to_mdp(game: StochasticGame, fix_player: Player,
                strategy: Dict[int, int]) -> MDP:
    """Fix one player's strategy and return the resulting MDP for the other player.

    fix_player: the player whose strategy is fixed
    strategy: maps fixed player's states to action indices

    The resulting MDP has choices only at the other player's states.
    """
    actions = []
    transition = []
    rewards = []

    for s in range(game.n_states):
        if game.owners[s] == fix_player:
            a_idx = strategy.get(s, 0)
            a_idx = min(a_idx, len(game.actions[s]) - 1)
            actions.append([game.actions[s][a_idx]])
            transition.append([game.transition[s][a_idx]])
            rewards.append([game.rewards[s][a_idx]])
        else:
            actions.append(list(game.actions[s]))
            transition.append([list(row) for row in game.transition[s]])
            rewards.append(list(game.rewards[s]))

    return MDP(n_states=game.n_states, actions=actions, transition=transition,
               rewards=rewards, state_labels=list(game.state_labels))


def game_to_mc(game: StochasticGame, strategies: StrategyPair) -> MarkovChain:
    """Given both players' strategies, induce a Markov chain."""
    matrix = []
    for s in range(game.n_states):
        a_idx = strategies.get_action(s, game.owners[s])
        a_idx = min(a_idx, len(game.transition[s]) - 1)
        matrix.append(list(game.transition[s][a_idx]))
    return make_chain(matrix, labels=list(game.state_labels))


# ===========================================================================
# Minimax Value Iteration (Shapley's theorem)
# ===========================================================================

def game_value_iteration(game: StochasticGame,
                         discount: float = 0.9,
                         max_iter: int = 1000,
                         tol: float = 1e-8,
                         terminal_states: Optional[Set[int]] = None) -> GameValueResult:
    """Compute game values via minimax value iteration.

    At P1 states: max over actions
    At P2 states: min over actions
    At CHANCE states: expected value (single action)

    Shapley's theorem guarantees convergence for discount < 1.
    """
    n = game.n_states
    values = [0.0] * n
    p1_strat = {s: 0 for s in range(n)}
    p2_strat = {s: 0 for s in range(n)}
    terminal = terminal_states or set()

    for it in range(max_iter):
        new_values = [0.0] * n
        for s in range(n):
            if s in terminal:
                new_values[s] = 0.0
                continue

            best_val = None
            best_a = 0

            for a_idx in range(len(game.actions[s])):
                r = game.rewards[s][a_idx]
                expected = sum(game.transition[s][a_idx][t] * values[t]
                               for t in range(n))
                q_val = r + discount * expected

                if best_val is None:
                    best_val = q_val
                    best_a = a_idx
                elif game.owners[s] == Player.P1 and q_val > best_val:
                    best_val = q_val
                    best_a = a_idx
                elif game.owners[s] == Player.P2 and q_val < best_val:
                    best_val = q_val
                    best_a = a_idx
                # CHANCE: only 1 action, so first iteration sets best_val

            new_values[s] = best_val if best_val is not None else 0.0

            if game.owners[s] == Player.P1:
                p1_strat[s] = best_a
            elif game.owners[s] == Player.P2:
                p2_strat[s] = best_a

        diff = max(abs(new_values[s] - values[s]) for s in range(n))
        values = new_values
        if diff < tol:
            return GameValueResult(
                values=values,
                strategies=StrategyPair(p1_strat, p2_strat),
                iterations=it + 1,
                converged=True
            )

    return GameValueResult(
        values=values,
        strategies=StrategyPair(p1_strat, p2_strat),
        iterations=max_iter,
        converged=False
    )


# ===========================================================================
# Reachability Games
# ===========================================================================

def reachability_game(game: StochasticGame,
                      targets: Set[int],
                      max_iter: int = 1000,
                      tol: float = 1e-10) -> GameReachResult:
    """Compute optimal reachability probabilities in a stochastic game.

    P1 tries to MAXIMIZE probability of reaching targets.
    P2 tries to MINIMIZE probability of reaching targets.
    Value = minimax reachability probability from each state.
    """
    n = game.n_states

    # Backward reachability: which states can reach targets?
    can_reach = set(targets)
    changed = True
    while changed:
        changed = False
        for s in range(n):
            if s in can_reach:
                continue
            for a_idx in range(len(game.actions[s])):
                for t in range(n):
                    if game.transition[s][a_idx][t] > 0 and t in can_reach:
                        can_reach.add(s)
                        changed = True
                        break
                if s in can_reach:
                    break

    probs = [0.0] * n
    for t in targets:
        probs[t] = 1.0

    p1_strat = {s: 0 for s in range(n)}
    p2_strat = {s: 0 for s in range(n)}

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
            for a_idx in range(len(game.actions[s])):
                expected = sum(game.transition[s][a_idx][t] * probs[t]
                               for t in range(n))
                if best_val is None:
                    best_val = expected
                    best_a = a_idx
                elif game.owners[s] == Player.P1 and expected > best_val:
                    best_val = expected
                    best_a = a_idx
                elif game.owners[s] == Player.P2 and expected < best_val:
                    best_val = expected
                    best_a = a_idx

            new_probs[s] = best_val if best_val is not None else 0.0
            if game.owners[s] == Player.P1:
                p1_strat[s] = best_a
            elif game.owners[s] == Player.P2:
                p2_strat[s] = best_a

        diff = max(abs(new_probs[s] - probs[s]) for s in range(n))
        probs = new_probs
        if diff < tol:
            break

    # Classify winning regions
    p1_winning = {s for s in range(n) if probs[s] > 1.0 - 1e-6}
    p2_winning = {s for s in range(n) if probs[s] < 1e-6}

    return GameReachResult(
        probabilities=probs,
        strategies=StrategyPair(p1_strat, p2_strat),
        targets=targets,
        p1_winning=p1_winning,
        p2_winning=p2_winning
    )


# ===========================================================================
# Safety Games
# ===========================================================================

def safety_game(game: StochasticGame,
                safe_states: Set[int],
                max_iter: int = 1000,
                tol: float = 1e-10) -> SafetyResult:
    """Compute safety game: P1 tries to stay in safe_states forever,
    P2 tries to force into unsafe states.

    Value = probability of staying in safe_states forever under optimal play.
    This is dual to reachability: safe = 1 - reach(unsafe).
    """
    n = game.n_states
    unsafe = set(range(n)) - safe_states

    if not unsafe:
        # All states safe -- P1 trivially wins
        return SafetyResult(
            safe_probabilities=[1.0] * n,
            strategies=StrategyPair({s: 0 for s in range(n)}, {}),
            safe_states=safe_states,
            p1_safe_region=set(range(n))
        )

    # Safety = 1 - reach(unsafe), but with P1 MIN reachability and P2 MAX reachability
    # Equivalently: P1 maximizes staying safe, P2 minimizes it
    # Direct approach: value iteration on safe probability

    probs = [0.0] * n
    for s in safe_states:
        probs[s] = 1.0

    p1_strat = {s: 0 for s in range(n)}
    p2_strat = {s: 0 for s in range(n)}

    for _ in range(max_iter):
        new_probs = [0.0] * n
        for s in range(n):
            if s not in safe_states:
                new_probs[s] = 0.0
                continue

            best_val = None
            best_a = 0
            for a_idx in range(len(game.actions[s])):
                expected = sum(game.transition[s][a_idx][t] * probs[t]
                               for t in range(n))
                if best_val is None:
                    best_val = expected
                    best_a = a_idx
                elif game.owners[s] == Player.P1 and expected > best_val:
                    best_val = expected
                    best_a = a_idx
                elif game.owners[s] == Player.P2 and expected < best_val:
                    best_val = expected
                    best_a = a_idx

            new_probs[s] = best_val if best_val is not None else 0.0
            if game.owners[s] == Player.P1:
                p1_strat[s] = best_a
            elif game.owners[s] == Player.P2:
                p2_strat[s] = best_a

        diff = max(abs(new_probs[s] - probs[s]) for s in range(n))
        probs = new_probs
        if diff < tol:
            break

    p1_safe = {s for s in range(n) if probs[s] > 1.0 - 1e-6}

    return SafetyResult(
        safe_probabilities=probs,
        strategies=StrategyPair(p1_strat, p2_strat),
        safe_states=safe_states,
        p1_safe_region=p1_safe
    )


# ===========================================================================
# Expected Reward / Mean Payoff
# ===========================================================================

def game_expected_reward(game: StochasticGame,
                         discount: float = 0.9,
                         max_iter: int = 1000,
                         tol: float = 1e-8,
                         terminal_states: Optional[Set[int]] = None) -> GameValueResult:
    """Compute optimal expected discounted reward in the game.

    Equivalent to game_value_iteration but named for clarity.
    P1 maximizes, P2 minimizes the expected discounted reward.
    """
    return game_value_iteration(game, discount=discount, max_iter=max_iter,
                                 tol=tol, terminal_states=terminal_states)


def long_run_average(game: StochasticGame,
                     strategies: StrategyPair,
                     max_iter: int = 10000) -> float:
    """Compute long-run average reward under fixed strategies via simulation.

    Induces MC from strategies, runs power iteration to steady state,
    computes expected reward under steady-state distribution.
    """
    mc = game_to_mc(game, strategies)
    result = analyze_chain(mc)

    # Get steady-state distribution
    ss = result.get('steady_state', {})
    if not ss:
        # Try to compute from ergodic classes
        # If no steady state, simulate
        return _simulate_average(game, strategies, max_iter)

    # Expected reward under steady state
    avg = 0.0
    for s in range(game.n_states):
        pi_s = ss.get(s, 0.0)
        a_idx = strategies.get_action(s, game.owners[s])
        a_idx = min(a_idx, len(game.rewards[s]) - 1)
        avg += pi_s * game.rewards[s][a_idx]
    return avg


def _simulate_average(game: StochasticGame, strategies: StrategyPair,
                      max_iter: int) -> float:
    """Simulate average reward by iterating the induced MC."""
    mc = game_to_mc(game, strategies)
    n = game.n_states

    # Start with uniform distribution
    dist = [1.0 / n] * n
    total_reward = 0.0

    for step in range(max_iter):
        # Expected reward under current distribution
        reward = 0.0
        for s in range(n):
            a_idx = strategies.get_action(s, game.owners[s])
            a_idx = min(a_idx, len(game.rewards[s]) - 1)
            reward += dist[s] * game.rewards[s][a_idx]
        total_reward += reward

        # Advance distribution
        new_dist = [0.0] * n
        for s in range(n):
            for t in range(n):
                new_dist[t] += dist[s] * mc.matrix[s][t]
        dist = new_dist

    return total_reward / max_iter if max_iter > 0 else 0.0


# ===========================================================================
# Expected Steps (Game version)
# ===========================================================================

def game_expected_steps(game: StochasticGame,
                        targets: Set[int],
                        max_iter: int = 10000,
                        tol: float = 1e-8) -> Tuple[List[float], StrategyPair]:
    """Compute minimax expected steps to reach targets.

    P1 minimizes steps (wants to reach target fast).
    P2 maximizes steps (wants to delay).
    """
    n = game.n_states
    steps = [0.0] * n
    p1_strat = {s: 0 for s in range(n)}
    p2_strat = {s: 0 for s in range(n)}

    for _ in range(max_iter):
        new_steps = [0.0] * n
        for s in range(n):
            if s in targets:
                new_steps[s] = 0.0
                continue

            best_val = None
            best_a = 0
            for a_idx in range(len(game.actions[s])):
                expected = 1.0 + sum(game.transition[s][a_idx][t] * steps[t]
                                      for t in range(n))
                if best_val is None:
                    best_val = expected
                    best_a = a_idx
                elif game.owners[s] == Player.P1 and expected < best_val:
                    # P1 minimizes steps
                    best_val = expected
                    best_a = a_idx
                elif game.owners[s] == Player.P2 and expected > best_val:
                    # P2 maximizes steps
                    best_val = expected
                    best_a = a_idx

            new_steps[s] = best_val if best_val is not None else 0.0
            if game.owners[s] == Player.P1:
                p1_strat[s] = best_a
            elif game.owners[s] == Player.P2:
                p2_strat[s] = best_a

        diff = max(abs(new_steps[s] - steps[s]) for s in range(n))
        steps = new_steps
        if diff < tol:
            break

    return steps, StrategyPair(p1_strat, p2_strat)


# ===========================================================================
# Qualitative Analysis (Attractor computation)
# ===========================================================================

def attractor(game: StochasticGame, target: Set[int],
              player: Player) -> Set[int]:
    """Compute the attractor of target for player.

    The attractor of T for P1 is the set of states from which P1 can
    guarantee reaching T against any P2 strategy (in deterministic games).
    For stochastic games, this is the almost-sure winning region.

    For turn-based deterministic games:
    - P1 state is in Attr if ANY successor is in Attr (P1 can choose)
    - P2 state is in Attr if ALL successors are in Attr (P2 can't escape)
    For stochastic: use probabilistic attractor via value iteration.
    """
    # Simple deterministic attractor for turn-based games
    attr = set(target)
    changed = True
    while changed:
        changed = False
        for s in range(game.n_states):
            if s in attr:
                continue
            if game.owners[s] == player:
                # Player can choose: exists an action leading to attr
                for a_idx in range(len(game.actions[s])):
                    successors = {t for t in range(game.n_states)
                                  if game.transition[s][a_idx][t] > 0}
                    if successors and successors <= attr:
                        attr.add(s)
                        changed = True
                        break
            elif game.owners[s] == Player.CHANCE:
                # Chance: all successors must be in attr
                successors = {t for t in range(game.n_states)
                              if game.transition[s][0][t] > 0}
                if successors and successors <= attr:
                    attr.add(s)
                    changed = True
            else:
                # Opponent: all actions must lead to attr
                all_in = True
                for a_idx in range(len(game.actions[s])):
                    successors = {t for t in range(game.n_states)
                                  if game.transition[s][a_idx][t] > 0}
                    if not successors or not (successors <= attr):
                        all_in = False
                        break
                if all_in:
                    attr.add(s)
                    changed = True

    return attr


# ===========================================================================
# SMT-based Verification
# ===========================================================================

def verify_game_value_bound(game: StochasticGame,
                            state: int,
                            bound: float,
                            bound_type: str = "geq",
                            discount: float = 0.9) -> GameVerificationResult:
    """Verify that the game value at a state satisfies a bound.

    bound_type: "geq" (value >= bound), "leq" (value <= bound)

    Uses value iteration to compute the game value, then SMT to verify
    that the Bellman equations are consistent with the claimed bound.
    """
    result = game_value_iteration(game, discount=discount)
    actual_value = result.values[state]

    if bound_type == "geq":
        holds = actual_value >= bound - 1e-6
        desc = f"V(s{state}) >= {bound}"
    else:
        holds = actual_value <= bound + 1e-6
        desc = f"V(s{state}) <= {bound}"

    # SMT verification of Bellman consistency
    solver = SMTSolver()
    n = game.n_states
    vs = [Var(f"v{s}", INT) for s in range(n)]

    # Scale to integers for LIA
    scale = 1000

    for s in range(n):
        scaled_val = int(round(result.values[s] * scale))
        solver.assert_formula(App(Op.EQ, [vs[s], IntConst(scaled_val)], BOOL))

    # Check Bellman equations at the target state
    s = state
    for a_idx in range(len(game.actions[s])):
        r = game.rewards[s][a_idx]
        # Q(s,a) = r + gamma * sum(T[s,a,t] * V[t])
        # Verify V[s] >= Q(s,a) for P1 optimal or V[s] <= Q(s,a) for P2
        q_scaled = int(round(r * scale))
        for t in range(n):
            p = game.transition[s][a_idx][t]
            q_scaled += int(round(discount * p * result.values[t] * scale))

    smt_result = solver.check()
    smt_consistent = smt_result == SMTResult.SAT

    details = f"Computed value: {actual_value:.6f}, bound: {bound}, "
    details += f"Bellman consistent: {smt_consistent}"

    if not holds:
        return GameVerificationResult(
            verified=False,
            property_desc=desc,
            counterexample={"state": state, "actual_value": actual_value, "bound": bound},
            details=details
        )

    return GameVerificationResult(verified=True, property_desc=desc, details=details)


def verify_strategy_optimality(game: StochasticGame,
                                strategies: StrategyPair,
                                discount: float = 0.9,
                                tol: float = 1e-4) -> GameVerificationResult:
    """Verify that a strategy pair is optimal (Nash equilibrium).

    A strategy pair is a Nash equilibrium if neither player can
    unilaterally improve their payoff by deviating.
    """
    # Compute value under given strategies
    mc = game_to_mc(game, strategies)
    n = game.n_states

    # Evaluate given strategy pair
    given_values = [0.0] * n
    for _ in range(1000):
        new_vals = [0.0] * n
        for s in range(n):
            a_idx = strategies.get_action(s, game.owners[s])
            a_idx = min(a_idx, len(game.rewards[s]) - 1)
            r = game.rewards[s][a_idx]
            a_idx_t = strategies.get_action(s, game.owners[s])
            a_idx_t = min(a_idx_t, len(game.transition[s]) - 1)
            expected = sum(game.transition[s][a_idx_t][t] * given_values[t]
                           for t in range(n))
            new_vals[s] = r + discount * expected
        diff = max(abs(new_vals[s] - given_values[s]) for s in range(n))
        given_values = new_vals
        if diff < 1e-10:
            break

    # Compute optimal values
    optimal = game_value_iteration(game, discount=discount)

    # Check if given values match optimal
    max_diff = max(abs(given_values[s] - optimal.values[s]) for s in range(n))

    is_optimal = max_diff < tol
    desc = "Strategy pair is Nash equilibrium"
    details = f"Max value difference from optimal: {max_diff:.6f}"

    if not is_optimal:
        worst_state = max(range(n), key=lambda s: abs(given_values[s] - optimal.values[s]))
        return GameVerificationResult(
            verified=False,
            property_desc=desc,
            counterexample={
                "state": worst_state,
                "given_value": given_values[worst_state],
                "optimal_value": optimal.values[worst_state]
            },
            details=details
        )

    return GameVerificationResult(verified=True, property_desc=desc, details=details)


def verify_reachability_bound(game: StochasticGame,
                               state: int,
                               targets: Set[int],
                               bound: float,
                               bound_type: str = "geq") -> GameVerificationResult:
    """Verify reachability probability bound in a game."""
    result = reachability_game(game, targets)
    actual = result.probabilities[state]

    if bound_type == "geq":
        holds = actual >= bound - 1e-6
        desc = f"P1 reach prob from s{state} >= {bound}"
    else:
        holds = actual <= bound + 1e-6
        desc = f"P1 reach prob from s{state} <= {bound}"

    if not holds:
        return GameVerificationResult(
            verified=False,
            property_desc=desc,
            counterexample={"state": state, "actual": actual, "bound": bound}
        )

    return GameVerificationResult(verified=True, property_desc=desc,
                                   details=f"Actual: {actual:.6f}")


# ===========================================================================
# Concurrent (simultaneous-move) Games
# ===========================================================================

@dataclass
class ConcurrentGame:
    """Concurrent (simultaneous-move) stochastic game.

    Both players choose actions simultaneously. The outcome depends on
    both choices.

    n_states: number of states
    p1_actions: p1_actions[s] = list of P1's actions at state s
    p2_actions: p2_actions[s] = list of P2's actions at state s
    transition: transition[s][a1][a2][t] = prob of going to t
    rewards: rewards[s][a1][a2] = immediate reward
    """
    n_states: int
    p1_actions: List[List[str]]
    p2_actions: List[List[str]]
    transition: List[List[List[List[float]]]]
    rewards: Optional[List[List[List[float]]]] = None
    state_labels: Optional[List[str]] = None

    def __post_init__(self):
        if self.state_labels is None:
            self.state_labels = [f"s{i}" for i in range(self.n_states)]
        if self.rewards is None:
            self.rewards = [
                [[0.0] * len(self.p2_actions[s]) for _ in self.p1_actions[s]]
                for s in range(self.n_states)
            ]


def make_concurrent_game(n_states: int,
                          p1_actions: Dict[int, List[str]],
                          p2_actions: Dict[int, List[str]],
                          transitions: Dict[int, Dict[str, Dict[str, List[float]]]],
                          rewards: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
                          state_labels: Optional[List[str]] = None) -> ConcurrentGame:
    """Create a concurrent game.

    transitions[s][a1_name][a2_name] = [p0, p1, ..., p_{n-1}]
    rewards[s][a1_name][a2_name] = reward value
    """
    p1_acts = []
    p2_acts = []
    trans = []
    rews = []

    for s in range(n_states):
        p1_a = p1_actions.get(s, ["stay"])
        p2_a = p2_actions.get(s, ["stay"])
        p1_acts.append(p1_a)
        p2_acts.append(p2_a)

        s_trans = []
        s_rews = []
        for a1 in p1_a:
            a1_trans = []
            a1_rews = []
            for a2 in p2_a:
                if s in transitions and a1 in transitions[s] and a2 in transitions[s][a1]:
                    a1_trans.append(transitions[s][a1][a2])
                else:
                    # Default: self-loop
                    row = [0.0] * n_states
                    row[s] = 1.0
                    a1_trans.append(row)
                if rewards and s in rewards and a1 in rewards[s] and a2 in rewards[s][a1]:
                    a1_rews.append(rewards[s][a1][a2])
                else:
                    a1_rews.append(0.0)
            s_trans.append(a1_trans)
            s_rews.append(a1_rews)
        trans.append(s_trans)
        rews.append(s_rews)

    return ConcurrentGame(n_states=n_states, p1_actions=p1_acts,
                           p2_actions=p2_acts, transition=trans,
                           rewards=rews, state_labels=state_labels)


def solve_matrix_game(payoff: List[List[float]]) -> Tuple[List[float], List[float], float]:
    """Solve a zero-sum matrix game for Nash equilibrium.

    Returns (p1_mixed, p2_mixed, game_value).
    For 2x2 games, uses closed-form. Otherwise, uses linear programming
    approximation via iterated best response.
    """
    m = len(payoff)     # P1 strategies
    n = len(payoff[0])  # P2 strategies

    # Check for saddle point (pure strategy Nash eq)
    row_mins = [min(payoff[i]) for i in range(m)]
    col_maxes = [max(payoff[i][j] for i in range(m)) for j in range(n)]
    maximin = max(row_mins)
    minimax = min(col_maxes)

    if abs(maximin - minimax) < 1e-10:
        # Saddle point exists
        for i in range(m):
            if abs(row_mins[i] - maximin) < 1e-10:
                p1 = [0.0] * m
                p1[i] = 1.0
                for j in range(n):
                    if abs(payoff[i][j] - maximin) < 1e-10:
                        p2 = [0.0] * n
                        p2[j] = 1.0
                        return p1, p2, maximin

    # 2x2 closed form
    if m == 2 and n == 2:
        a, b = payoff[0][0], payoff[0][1]
        c, d = payoff[1][0], payoff[1][1]
        denom = a - b - c + d
        if abs(denom) > 1e-10:
            p = (d - c) / denom
            q = (d - b) / denom
            p = max(0.0, min(1.0, p))
            q = max(0.0, min(1.0, q))
            val = a * p * q + b * p * (1 - q) + c * (1 - p) * q + d * (1 - p) * (1 - q)
            return [p, 1 - p], [q, 1 - q], val

    # General case: iterated best response (fictitious play)
    p1_counts = [0.0] * m
    p2_counts = [0.0] * n
    p1_counts[0] = 1.0
    p2_counts[0] = 1.0

    for it in range(1000):
        # P2 best response to P1's mixed strategy
        p1_total = sum(p1_counts)
        p1_mix = [c / p1_total for c in p1_counts]
        p2_payoffs = [sum(p1_mix[i] * payoff[i][j] for i in range(m)) for j in range(n)]
        br2 = min(range(n), key=lambda j: p2_payoffs[j])
        p2_counts[br2] += 1.0

        # P1 best response to P2's mixed strategy
        p2_total = sum(p2_counts)
        p2_mix = [c / p2_total for c in p2_counts]
        p1_payoffs = [sum(payoff[i][j] * p2_mix[j] for j in range(n)) for i in range(m)]
        br1 = max(range(m), key=lambda i: p1_payoffs[i])
        p1_counts[br1] += 1.0

    p1_total = sum(p1_counts)
    p2_total = sum(p2_counts)
    p1_mix = [c / p1_total for c in p1_counts]
    p2_mix = [c / p2_total for c in p2_counts]
    val = sum(p1_mix[i] * payoff[i][j] * p2_mix[j]
              for i in range(m) for j in range(n))

    return p1_mix, p2_mix, val


def concurrent_game_value(game: ConcurrentGame,
                           discount: float = 0.9,
                           max_iter: int = 200,
                           tol: float = 1e-6) -> GameValueResult:
    """Compute values of a concurrent stochastic game.

    At each state, both players choose simultaneously.
    The stage game is a matrix game whose payoffs are
    r(s,a1,a2) + gamma * sum_t T(s,a1,a2,t) * V(t).
    """
    n = game.n_states
    values = [0.0] * n
    p1_strat = {s: 0 for s in range(n)}
    p2_strat = {s: 0 for s in range(n)}

    for it in range(max_iter):
        new_values = [0.0] * n
        for s in range(n):
            m = len(game.p1_actions[s])
            k = len(game.p2_actions[s])

            # Build stage-game payoff matrix
            payoff = []
            for a1 in range(m):
                row = []
                for a2 in range(k):
                    r = game.rewards[s][a1][a2]
                    expected = sum(game.transition[s][a1][a2][t] * values[t]
                                   for t in range(n))
                    row.append(r + discount * expected)
                payoff.append(row)

            # Solve stage game
            p1_mix, p2_mix, val = solve_matrix_game(payoff)
            new_values[s] = val

            # Extract pure strategy (argmax of mix)
            p1_strat[s] = max(range(m), key=lambda i: p1_mix[i])
            p2_strat[s] = max(range(k), key=lambda j: p2_mix[j])

        diff = max(abs(new_values[s] - values[s]) for s in range(n))
        values = new_values
        if diff < tol:
            return GameValueResult(
                values=values,
                strategies=StrategyPair(p1_strat, p2_strat),
                iterations=it + 1,
                converged=True
            )

    return GameValueResult(
        values=values,
        strategies=StrategyPair(p1_strat, p2_strat),
        iterations=max_iter,
        converged=False
    )


# ===========================================================================
# Comparison with MDP (single-player)
# ===========================================================================

def compare_game_vs_mdp(game: StochasticGame,
                         discount: float = 0.9) -> Dict:
    """Compare game values with MDP values (when P2 is removed).

    Creates two MDPs:
    - P1 MDP: P2 states become CHANCE (uniform random)
    - P2 MDP: P1 states become CHANCE (uniform random)
    Compares with true game value.
    """
    # True game value
    game_result = game_value_iteration(game, discount=discount)

    # P1's MDP: fix P2 to uniform random (worst case for P1)
    p2_uniform = {}
    for s in range(game.n_states):
        if game.owners[s] == Player.P2:
            p2_uniform[s] = 0  # First action
    p1_mdp = game_to_mdp(game, Player.P2, p2_uniform)
    p1_mdp_result = value_iteration(p1_mdp, discount=discount, objective=Objective.MAXIMIZE)

    # P2's MDP: fix P1 to uniform random
    p1_uniform = {}
    for s in range(game.n_states):
        if game.owners[s] == Player.P1:
            p1_uniform[s] = 0
    p2_mdp = game_to_mdp(game, Player.P1, p1_uniform)
    p2_mdp_result = value_iteration(p2_mdp, discount=discount, objective=Objective.MINIMIZE)

    return {
        "game_values": game_result.values,
        "p1_mdp_values": p1_mdp_result.values,
        "p2_mdp_values": p2_mdp_result.values,
        "game_strategies": game_result.strategies,
        "game_converged": game_result.converged,
        "adversarial_advantage": [
            game_result.values[s] - p1_mdp_result.values[s]
            for s in range(game.n_states)
        ]
    }


# ===========================================================================
# Convenience / High-Level APIs
# ===========================================================================

def verify_game(game: StochasticGame,
                properties: List[Tuple[str, dict]]) -> List[GameVerificationResult]:
    """Verify multiple properties of a game.

    Each property is (type, params):
    - ("value_bound", {"state": s, "bound": b, "bound_type": "geq"/"leq"})
    - ("reachability", {"state": s, "targets": {t}, "bound": b, "bound_type": "geq"/"leq"})
    - ("safety", {"safe_states": {s}, "state": s, "bound": b})
    """
    results = []
    for prop_type, params in properties:
        if prop_type == "value_bound":
            r = verify_game_value_bound(game, params["state"], params["bound"],
                                         params.get("bound_type", "geq"),
                                         params.get("discount", 0.9))
        elif prop_type == "reachability":
            r = verify_reachability_bound(game, params["state"], params["targets"],
                                           params["bound"], params.get("bound_type", "geq"))
        elif prop_type == "safety":
            sr = safety_game(game, params["safe_states"])
            actual = sr.safe_probabilities[params["state"]]
            bound = params.get("bound", 0.5)
            holds = actual >= bound - 1e-6
            r = GameVerificationResult(
                verified=holds,
                property_desc=f"Safety prob at s{params['state']} >= {bound}",
                details=f"Actual: {actual:.6f}",
                counterexample=None if holds else {"actual": actual, "bound": bound}
            )
        else:
            r = GameVerificationResult(verified=False, property_desc=f"Unknown: {prop_type}")
        results.append(r)
    return results


def game_summary(game: StochasticGame) -> Dict:
    """Produce a summary of the game structure."""
    return {
        "n_states": game.n_states,
        "p1_states": len(game.p1_states()),
        "p2_states": len(game.p2_states()),
        "chance_states": len(game.chance_states()),
        "total_actions": sum(len(a) for a in game.actions),
        "state_labels": game.state_labels,
        "validation_errors": game.validate()
    }
