"""
V074: Omega-Regular Games
=========================
LTL objectives for two-player stochastic games.

Given a StochasticGame and an LTL formula, computes the maximum/minimum
probability that Player 1 can satisfy the LTL objective against Player 2.

Approach:
1. Convert LTL formula to NBA (Buchi automaton) via V023's tableau construction
2. Build product game: StochasticGame x NBA
3. Solve the Buchi game on the product:
   - Qualitative: compute almost-sure and positive winning regions
   - Quantitative: value iteration for optimal probabilities
4. Extract strategies and project back to original game

Composes:
- V023 (LTL model checking): LTL AST, parser, LTL-to-GBA-to-NBA pipeline
- V070 (stochastic games): StochasticGame, Player, StrategyPair, reachability_game
- V072 (game PCTL): LabeledGame, GameQuantification
- V073 (game synthesis): SynthesisResult, verify_strategy, Objective

Key algorithm: Buchi game solving
- Almost-sure winning: largest set W s.t. from every state in W,
  P1 can force reaching accepting states in W (attractor fixpoint)
- Quantitative: value iteration on product game with Buchi acceptance
  reduced to repeated reachability
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)

# V023: LTL AST, parser, automaton construction
sys.path.insert(0, os.path.join(_work, 'V023_ltl_model_checking'))
from ltl_model_checker import (
    LTL, LTLOp, Atom, LTLTrue, LTLFalse, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release, WeakUntil,
    atoms, nnf, ltl_to_gba, gba_to_nba, GBA, NBA, Label,
    parse_ltl
)

# V070: Stochastic games
sys.path.insert(0, os.path.join(_work, 'V070_stochastic_games'))
from stochastic_games import (
    StochasticGame, Player, StrategyPair, GameValueResult,
    GameReachResult, make_game, game_to_mc, reachability_game,
    safety_game
)

# V072: Game PCTL (for LabeledGame)
sys.path.insert(0, os.path.join(_work, 'V072_game_pctl'))
from game_pctl import LabeledGame, make_labeled_game, GameQuantification

# V065: Markov chain (for verification)
sys.path.insert(0, os.path.join(_work, 'V065_markov_chain_analysis'))
from markov_chain import MarkovChain, make_chain, analyze_chain


# ============================================================
# Result Types
# ============================================================

@dataclass
class OmegaRegularResult:
    """Result of LTL game solving."""
    formula: LTL
    satisfied: bool                              # Does initial state satisfy?
    probabilities: List[float]                   # Per-state max probabilities
    p1_strategy: Optional[Dict[int, int]] = None # Product-state -> action index
    p2_strategy: Optional[Dict[int, int]] = None
    game_p1_strategy: Optional[Dict[int, int]] = None  # Original game state -> action
    game_p2_strategy: Optional[Dict[int, int]] = None
    automaton_states: int = 0
    product_states: int = 0
    iterations: int = 0
    method: str = "omega_regular"
    almost_sure_winning: Optional[Set[int]] = None  # Product states where P1 wins w.p.1
    positive_winning: Optional[Set[int]] = None     # Product states where P1 wins w.p.>0

    def summary(self) -> str:
        lines = [f"LTL Formula: {self.formula}"]
        lines.append(f"Satisfied at state 0: {self.satisfied}")
        lines.append(f"P(satisfy) at state 0: {self.probabilities[0]:.6f}" if self.probabilities else "No probabilities")
        lines.append(f"Automaton states: {self.automaton_states}")
        lines.append(f"Product states: {self.product_states}")
        lines.append(f"Iterations: {self.iterations}")
        if self.almost_sure_winning is not None:
            lines.append(f"Almost-sure winning: {len(self.almost_sure_winning)} states")
        return "\n".join(lines)


@dataclass
class ProductGame:
    """Product of StochasticGame x NBA."""
    game: StochasticGame                 # The product game
    accepting: Set[int]                  # Accepting product states
    state_map: Dict[Tuple[int, int], int]  # (game_state, aut_state) -> product_state
    inv_map: Dict[int, Tuple[int, int]]    # product_state -> (game_state, aut_state)
    n_game_states: int
    n_aut_states: int
    initial_product_states: Set[int]     # Product states reachable from initial


# ============================================================
# Product Game Construction
# ============================================================

def _compute_label_match(label: Label, state_labels: Dict[int, Set[str]],
                         game_state: int) -> bool:
    """Check if a game state's labels match an automaton transition label."""
    state_props = state_labels.get(game_state, set())
    # All positive atoms must be present
    for p in label.pos:
        if p not in state_props:
            return False
    # All negative atoms must be absent
    for n in label.neg:
        if n in state_props:
            return False
    return True


def build_product_game(game: StochasticGame, nba: NBA,
                       labels: Dict[int, Set[str]],
                       initial_game_state: int = 0) -> ProductGame:
    """Build product of StochasticGame x NBA.

    Product state (g, q) represents game state g with automaton state q.
    At each step:
    1. Game player chooses action a at state g
    2. Transition to game successor g' with probability T[g][a][g']
    3. Automaton transitions from q to q' based on labels of g'

    The automaton transition is deterministic given the game state labels.
    If NBA is nondeterministic, we take the union (any matching transition).
    """
    n_g = game.n_states
    n_q = len(nba.states)

    # Build state mapping
    state_map = {}
    inv_map = {}
    idx = 0
    for g in range(n_g):
        for q in sorted(nba.states):
            state_map[(g, q)] = idx
            inv_map[idx] = (g, q)
            idx += 1

    n_product = idx
    accepting = set()
    for ps in range(n_product):
        g, q = inv_map[ps]
        if q in nba.accepting:
            accepting.add(ps)

    # Build product transitions
    # Product state owners mirror game state owners
    owners = []
    actions_list = []
    transition_list = []
    rewards_list = [] if game.rewards is not None else None

    for ps in range(n_product):
        g, q = inv_map[ps]
        owners.append(game.owners[g])

        # Same actions as the game state
        state_actions = game.actions[g]
        actions_list.append(list(state_actions))

        state_trans = []
        state_rewards = [] if rewards_list is not None else None

        for a_idx in range(len(state_actions)):
            # Distribution over product successors
            prod_dist = [0.0] * n_product

            for g_prime in range(n_g):
                prob = game.transition[g][a_idx][g_prime]
                if prob <= 0:
                    continue

                # Find automaton successors from q given labels of g_prime
                aut_succs = set()
                if q in nba.transitions:
                    for label, q_prime in nba.transitions[q]:
                        if _compute_label_match(label, labels, g_prime):
                            aut_succs.add(q_prime)

                if aut_succs:
                    # Distribute probability equally among automaton successors
                    per_succ = prob / len(aut_succs)
                    for q_prime in aut_succs:
                        ps_prime = state_map[(g_prime, q_prime)]
                        prod_dist[ps_prime] += per_succ
                # If no automaton successor, probability goes to a sink
                # (implicitly: stays at 0 probability in those product states)

            state_trans.append(prod_dist)
            if state_rewards is not None:
                state_rewards.append(game.rewards[g][a_idx])

        transition_list.append(state_trans)
        if rewards_list is not None:
            rewards_list.append(state_rewards)

    # Build product game
    product = StochasticGame(
        n_states=n_product,
        owners=owners,
        actions=actions_list,
        transition=transition_list,
        rewards=rewards_list,
        state_labels=[f"({inv_map[i][0]},{inv_map[i][1]})" for i in range(n_product)]
    )

    # Find initial product states
    initial_ps = set()
    for q0 in nba.initial:
        if (initial_game_state, q0) in state_map:
            initial_ps.add(state_map[(initial_game_state, q0)])

    return ProductGame(
        game=product,
        accepting=accepting,
        state_map=state_map,
        inv_map=inv_map,
        n_game_states=n_g,
        n_aut_states=n_q,
        initial_product_states=initial_ps
    )


# ============================================================
# Buchi Game Solving (Qualitative)
# ============================================================

def _attractor(game: StochasticGame, target: Set[int], player: Player) -> Set[int]:
    """Compute attractor of target for given player.

    A state s is in Attr(target) if:
    - s is in target, OR
    - s belongs to player and SOME action leads entirely to Attr(target), OR
    - s belongs to opponent and ALL actions lead to Attr(target)

    For CHANCE nodes: treat as opponent (ALL successors must be in attractor).
    """
    attr = set(target)
    changed = True

    while changed:
        changed = False
        for s in range(game.n_states):
            if s in attr:
                continue

            owner = game.owners[s]
            if owner == player:
                # Player: exists an action where all successors in attr
                for a_idx in range(len(game.actions[s])):
                    all_in = True
                    has_succ = False
                    for t in range(game.n_states):
                        if game.transition[s][a_idx][t] > 0:
                            has_succ = True
                            if t not in attr:
                                all_in = False
                                break
                    if has_succ and all_in:
                        attr.add(s)
                        changed = True
                        break
            else:
                # Opponent/CHANCE: all actions must lead entirely to attr
                all_actions_in = True
                has_actions = len(game.actions[s]) > 0
                for a_idx in range(len(game.actions[s])):
                    for t in range(game.n_states):
                        if game.transition[s][a_idx][t] > 0:
                            if t not in attr:
                                all_actions_in = False
                                break
                    if not all_actions_in:
                        break
                if has_actions and all_actions_in:
                    attr.add(s)
                    changed = True

    return attr


def compute_almost_sure_winning(product: ProductGame) -> Set[int]:
    """Compute almost-sure winning region for P1 in a Buchi game.

    A state is almost-sure winning if P1 can force visiting accepting
    states infinitely often with probability 1.

    Algorithm (McNaughton/Zielonka-style):
    Iterate: W = all states
    Repeat until stable:
      1. Remove states from which P1 cannot force reaching accepting states in W
      2. W = attractor of (accepting intersect W) for P1
      Actually: compute P2-attractor of complement, remove it
    """
    game = product.game
    all_states = set(range(game.n_states))
    winning = set(all_states)

    while True:
        # States where P1 can force reaching accepting AND winning
        good_targets = product.accepting & winning

        if not good_targets:
            winning = set()
            break

        # Compute P1's attractor to good targets within winning
        # But we need to restrict to the subgame on 'winning' states
        # Simpler: compute P2's attractor to complement of winning (trap for P2)
        # Actually use the standard Buchi game algorithm:
        # Remove states from which P1 CANNOT force reaching good_targets

        # P1 can force reaching good_targets = Attr_P1(good_targets)
        attr_p1 = _attractor_in_subgame(game, good_targets, Player.P1, winning)

        # States in winning but not in attr_p1: P1 cannot force reaching accepting
        losing = winning - attr_p1

        if not losing:
            break  # Stable

        # Remove losing and their P2-attractor from winning
        p2_attr = _attractor_in_subgame(game, losing, Player.P2, winning)
        winning = winning - p2_attr

    return winning


def _attractor_in_subgame(game: StochasticGame, target: Set[int],
                           player: Player, arena: Set[int]) -> Set[int]:
    """Compute attractor within a subgame restricted to arena states."""
    attr = target & arena
    changed = True

    while changed:
        changed = False
        for s in arena:
            if s in attr:
                continue

            owner = game.owners[s]
            if owner == player:
                # Player: exists action where all successors in arena go to attr
                for a_idx in range(len(game.actions[s])):
                    all_in = True
                    has_succ = False
                    for t in range(game.n_states):
                        if game.transition[s][a_idx][t] > 0:
                            if t not in arena:
                                continue  # Ignore transitions outside arena
                            has_succ = True
                            if t not in attr:
                                all_in = False
                                break
                    if has_succ and all_in:
                        attr.add(s)
                        changed = True
                        break
            else:
                # Opponent: all successors in arena must be in attr
                all_actions_in = True
                has_any = False
                for a_idx in range(len(game.actions[s])):
                    for t in range(game.n_states):
                        if game.transition[s][a_idx][t] > 0 and t in arena:
                            has_any = True
                            if t not in attr:
                                all_actions_in = False
                                break
                    if not all_actions_in:
                        break
                if has_any and all_actions_in:
                    attr.add(s)
                    changed = True

    return attr


def compute_positive_winning(product: ProductGame) -> Set[int]:
    """Compute positive winning region: states where P1 can win with probability > 0.

    A state has positive winning probability if there EXISTS a path to an
    accepting cycle that P1 can force with nonzero probability.
    """
    game = product.game
    all_states = set(range(game.n_states))

    # First find all states that can reach accepting states
    can_reach_accepting = set()
    # BFS backward from accepting states
    queue = list(product.accepting)
    can_reach_accepting = set(product.accepting)
    while queue:
        s = queue.pop(0)
        for pred in range(game.n_states):
            if pred in can_reach_accepting:
                continue
            for a_idx in range(len(game.actions[pred])):
                if game.transition[pred][a_idx][s] > 0:
                    can_reach_accepting.add(pred)
                    queue.append(pred)
                    break

    # Positive winning = states that can reach an accepting state that can
    # reach itself (accepting cycle exists)
    # Find accepting states in SCCs (can reach themselves)
    accepting_in_cycle = set()
    for acc in product.accepting:
        # Can acc reach itself?
        visited = set()
        q = [acc]
        found = False
        while q and not found:
            curr = q.pop(0)
            for a_idx in range(len(game.actions[curr])):
                for t in range(game.n_states):
                    if game.transition[curr][a_idx][t] > 0 and t not in visited:
                        if t == acc:
                            found = True
                            break
                        visited.add(t)
                        q.append(t)
                if found:
                    break
        if found:
            accepting_in_cycle.add(acc)

    # Positive winning = can reach accepting_in_cycle
    positive = set(accepting_in_cycle)
    queue = list(accepting_in_cycle)
    while queue:
        s = queue.pop(0)
        for pred in range(game.n_states):
            if pred in positive:
                continue
            for a_idx in range(len(game.actions[pred])):
                if game.transition[pred][a_idx][s] > 0:
                    positive.add(pred)
                    queue.append(pred)
                    break

    return positive


# ============================================================
# Buchi Game Solving (Quantitative)
# ============================================================

def solve_buchi_game_quantitative(product: ProductGame,
                                   max_iter: int = 10000,
                                   tol: float = 1e-10) -> Tuple[List[float], Dict[int, int], Dict[int, int]]:
    """Compute optimal probabilities for Buchi acceptance in a stochastic game.

    Uses the reduction: Buchi acceptance (visit accepting infinitely often)
    = limit of repeated reachability to accepting states.

    Algorithm:
    1. Compute almost-sure winning region W
    2. States in W get probability 1.0
    3. States that cannot reach accepting get probability 0.0
    4. For remaining states: solve repeated reachability game

    For quantitative Buchi: we compute the VALUE of the game, where P1
    tries to maximize the probability of visiting accepting states
    infinitely often, and P2 tries to minimize it.

    Reduction to reachability: the value of a Buchi game equals the
    value of the repeated reachability game to accepting states,
    computed as a nested fixpoint.
    """
    game = product.game
    n = game.n_states
    accepting = product.accepting

    # Step 1: Qualitative analysis
    as_winning = compute_almost_sure_winning(product)
    pos_winning = compute_positive_winning(product)

    # Step 2: Initialize values
    values = [0.0] * n
    for s in as_winning:
        values[s] = 1.0

    # States that can't win at all stay at 0
    uncertain = pos_winning - as_winning

    if not uncertain:
        # All states are either almost-sure winning or losing
        p1_strat, p2_strat = _extract_buchi_strategies(game, values, accepting)
        return values, p1_strat, p2_strat

    # Step 3: Value iteration for uncertain states
    # Use repeated reachability: outer fixpoint over inner reachability game
    # Inner: solve reachability to (accepting & current_winning) from uncertain
    # Outer: iterate until values stabilize

    for outer in range(max_iter):
        old_values = list(values)

        # Inner: one round of reachability value iteration
        # Target: accepting states (absorb at value 1 if in accepting, else propagate)
        inner_values = list(values)

        for inner in range(max_iter):
            new_inner = list(inner_values)

            for s in range(n):
                if s in as_winning:
                    new_inner[s] = 1.0
                    continue
                if s not in pos_winning:
                    new_inner[s] = 0.0
                    continue

                owner = game.owners[s]
                action_vals = []

                for a_idx in range(len(game.actions[s])):
                    ev = 0.0
                    for t in range(n):
                        p = game.transition[s][a_idx][t]
                        if p > 0:
                            if t in accepting:
                                # Reaching accepting: restart with current value
                                ev += p * values[t]
                            else:
                                ev += p * inner_values[t]
                    action_vals.append(ev)

                if not action_vals:
                    new_inner[s] = 0.0
                    continue

                if owner == Player.P1:
                    new_inner[s] = max(action_vals)
                elif owner == Player.P2:
                    new_inner[s] = min(action_vals)
                else:  # CHANCE
                    new_inner[s] = action_vals[0] if action_vals else 0.0

            diff = max(abs(new_inner[i] - inner_values[i]) for i in range(n))
            inner_values = new_inner
            if diff < tol:
                break

        # Update values for accepting states based on inner results
        for s in range(n):
            if s in as_winning:
                values[s] = 1.0
            elif s not in pos_winning:
                values[s] = 0.0
            elif s in accepting:
                # Accepting state: can restart, value is from inner iteration
                values[s] = inner_values[s]
            else:
                values[s] = inner_values[s]

        diff = max(abs(values[i] - old_values[i]) for i in range(n))
        if diff < tol:
            break

    p1_strat, p2_strat = _extract_buchi_strategies(game, values, accepting)
    return values, p1_strat, p2_strat


def _extract_buchi_strategies(game: StochasticGame, values: List[float],
                               accepting: Set[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Extract optimal strategies from converged values."""
    p1_strat = {}
    p2_strat = {}

    for s in range(game.n_states):
        owner = game.owners[s]
        if owner == Player.CHANCE:
            continue

        best_val = None
        best_idx = 0

        for a_idx in range(len(game.actions[s])):
            ev = 0.0
            for t in range(game.n_states):
                p = game.transition[s][a_idx][t]
                if p > 0:
                    ev += p * values[t]

            if best_val is None:
                best_val = ev
                best_idx = a_idx
            elif owner == Player.P1 and ev > best_val:
                best_val = ev
                best_idx = a_idx
            elif owner == Player.P2 and ev < best_val:
                best_val = ev
                best_idx = a_idx

        if owner == Player.P1:
            p1_strat[s] = best_idx
        else:
            p2_strat[s] = best_idx

    return p1_strat, p2_strat


def _project_strategy(product: ProductGame, prod_strategy: Dict[int, int],
                       player: Player, nba_initial: Set[int] = None) -> Dict[int, int]:
    """Project product-game strategy to original game strategy.

    For each original game state, prefer the action from the initial
    automaton state (most relevant). Falls back to highest-value action
    across automaton states.
    """
    # Collect actions per game state, keyed by automaton state
    state_actions = {}  # game_state -> {aut_state: action_idx}
    for ps, a_idx in prod_strategy.items():
        g, q = product.inv_map[ps]
        if g not in state_actions:
            state_actions[g] = {}
        state_actions[g][q] = a_idx

    result = {}
    initial_qs = nba_initial if nba_initial else set()

    for g in range(product.n_game_states):
        if g not in state_actions:
            result[g] = 0
            continue

        actions = state_actions[g]

        # Prefer initial automaton state action
        chosen = None
        for q0 in initial_qs:
            if q0 in actions:
                chosen = actions[q0]
                break

        if chosen is None:
            # Fall back to first available
            chosen = next(iter(actions.values()))

        result[g] = chosen

    return result


# ============================================================
# Main API
# ============================================================

def check_ltl_game(game: StochasticGame, labels: Dict[int, Set[str]],
                    formula: LTL, initial_state: int = 0,
                    max_iter: int = 10000, tol: float = 1e-10) -> OmegaRegularResult:
    """Check an LTL formula against a stochastic game.

    Args:
        game: The stochastic game
        labels: Atomic proposition labeling {state -> {prop1, prop2, ...}}
        formula: LTL formula to check
        initial_state: Initial game state (default 0)
        max_iter: Maximum iterations for value iteration
        tol: Convergence tolerance

    Returns:
        OmegaRegularResult with probabilities and strategies
    """
    # Step 1: Negate formula and convert to NBA
    neg_formula = nnf(Not(formula))
    gba = ltl_to_gba(neg_formula)
    nba = gba_to_nba(gba)

    # Step 2: Build product game
    product = build_product_game(game, nba, labels, initial_state)

    # Step 3: Solve Buchi game on product
    values, p1_strat, p2_strat = solve_buchi_game_quantitative(
        product, max_iter=max_iter, tol=tol
    )

    # The values give P(negated formula satisfied) = P(formula violated)
    # So P(formula satisfied) = 1 - P(negated formula satisfied)
    # But wait: we built a product for the NEGATION. States where P2 wins
    # in the Buchi game (for negated formula) are states where P1 LOSES
    # the original formula.
    #
    # Actually, for LTL game checking:
    # - P1 wants to SATISFY formula
    # - P2 wants to VIOLATE formula (satisfy NOT formula)
    # - We negate and build Buchi automaton for NOT(formula)
    # - In the product game, ACCEPTING states correspond to runs satisfying NOT(formula)
    # - P2 wants to visit accepting states infinitely often (to violate formula)
    # - P1 wants to avoid accepting states infinitely often (to satisfy formula)
    #
    # So we solve the Buchi game with P2 as the Buchi player:
    # values[s] = max prob P2 can force Buchi acceptance = prob formula violated
    # P(formula satisfied) = 1 - values[s]

    # Re-solve: swap player roles for Buchi acceptance
    # P2 maximizes acceptance of NOT(formula), P1 minimizes
    inv_values, inv_p1, inv_p2 = _solve_buchi_swapped(
        product, max_iter=max_iter, tol=tol
    )

    # P(formula satisfied from state s) = 1 - inv_values[s]
    # But we need per-GAME-state probabilities
    game_probs = [0.0] * game.n_states
    for g in range(game.n_states):
        # Take max over automaton initial states
        best = 0.0
        for q0 in nba.initial:
            if (g, q0) in product.state_map:
                ps = product.state_map[(g, q0)]
                best = max(best, 1.0 - inv_values[ps])
        game_probs[g] = best

    # Compute qualitative regions
    as_winning = compute_almost_sure_winning(product)
    pos_winning = compute_positive_winning(product)

    # Map to game states for almost-sure/positive
    as_game = set()
    pos_game = set()
    for ps in as_winning:
        g, q = product.inv_map[ps]
        as_game.add(g)
    for ps in pos_winning:
        g, q = product.inv_map[ps]
        pos_game.add(g)

    # Project strategies to game level
    game_p1 = _project_strategy(product, inv_p1, Player.P1, nba.initial) if inv_p1 else None
    game_p2 = _project_strategy(product, inv_p2, Player.P2, nba.initial) if inv_p2 else None

    # Initial state probability
    init_prob = game_probs[initial_state]
    satisfied = init_prob > 1.0 - tol

    return OmegaRegularResult(
        formula=formula,
        satisfied=satisfied,
        probabilities=game_probs,
        p1_strategy=inv_p1,
        p2_strategy=inv_p2,
        game_p1_strategy=game_p1,
        game_p2_strategy=game_p2,
        automaton_states=len(nba.states),
        product_states=product.game.n_states,
        iterations=0,
        almost_sure_winning=as_winning,
        positive_winning=pos_winning
    )


def _solve_buchi_swapped(product: ProductGame,
                          max_iter: int = 10000,
                          tol: float = 1e-10) -> Tuple[List[float], Dict[int, int], Dict[int, int]]:
    """Solve Buchi game with P2 as the Buchi player (maximizing acceptance).

    P2 wants to visit accepting states infinitely often.
    P1 wants to avoid this.

    This is the standard formulation when checking NOT(formula).
    """
    game = product.game
    n = game.n_states
    accepting = product.accepting

    # Value iteration: P2 maximizes repeated reachability to accepting
    values = [0.0] * n

    for iteration in range(max_iter):
        new_values = [0.0] * n

        for s in range(n):
            owner = game.owners[s]
            action_vals = []

            for a_idx in range(len(game.actions[s])):
                ev = 0.0
                for t in range(n):
                    p = game.transition[s][a_idx][t]
                    if p > 0:
                        if t in accepting:
                            # Reaching accepting: credit + continue
                            ev += p * max(values[t], 1.0)
                        else:
                            ev += p * values[t]
                action_vals.append(ev)

            if not action_vals:
                new_values[s] = 0.0
                continue

            # P2 (minimizer in original game) MAXIMIZES acceptance of NOT(formula)
            # P1 (maximizer in original game) MINIMIZES acceptance of NOT(formula)
            if owner == Player.P1:
                new_values[s] = min(action_vals)
            elif owner == Player.P2:
                new_values[s] = max(action_vals)
            else:  # CHANCE
                new_values[s] = action_vals[0]

        # Clamp to [0, 1]
        for s in range(n):
            new_values[s] = min(1.0, max(0.0, new_values[s]))

        diff = max(abs(new_values[i] - values[i]) for i in range(n))
        values = new_values

        if diff < tol:
            break

    p1_strat, p2_strat = {}, {}
    for s in range(n):
        owner = game.owners[s]
        if owner == Player.CHANCE:
            continue

        best_val = None
        best_idx = 0

        for a_idx in range(len(game.actions[s])):
            ev = 0.0
            for t in range(n):
                p = game.transition[s][a_idx][t]
                if p > 0:
                    if t in accepting:
                        ev += p * max(values[t], 1.0)
                    else:
                        ev += p * values[t]

            if best_val is None:
                best_val = ev
                best_idx = a_idx
            elif owner == Player.P1 and ev < best_val:  # P1 minimizes
                best_val = ev
                best_idx = a_idx
            elif owner == Player.P2 and ev > best_val:  # P2 maximizes
                best_val = ev
                best_idx = a_idx

        if owner == Player.P1:
            p1_strat[s] = best_idx
        else:
            p2_strat[s] = best_idx

    return values, p1_strat, p2_strat


# ============================================================
# Direct LTL Game (without negation -- P1 maximizes satisfaction)
# ============================================================

def check_ltl_game_direct(game: StochasticGame, labels: Dict[int, Set[str]],
                           formula: LTL, initial_state: int = 0,
                           max_iter: int = 10000, tol: float = 1e-10) -> OmegaRegularResult:
    """Check LTL formula using direct automaton construction (no negation).

    Builds automaton for the formula itself. P1 maximizes acceptance
    (visiting accepting states of formula's automaton infinitely often).
    """
    # Convert formula to NBA directly
    formula_nnf = nnf(formula)
    gba = ltl_to_gba(formula_nnf)
    nba = gba_to_nba(gba)

    # Build product
    product = build_product_game(game, nba, labels, initial_state)

    # Solve Buchi game: P1 maximizes visiting accepting states
    values, p1_strat, p2_strat = solve_buchi_game_quantitative(
        product, max_iter=max_iter, tol=tol
    )

    # Per-game-state probabilities
    game_probs = [0.0] * game.n_states
    for g in range(game.n_states):
        best = 0.0
        for q0 in nba.initial:
            if (g, q0) in product.state_map:
                ps = product.state_map[(g, q0)]
                best = max(best, values[ps])
        game_probs[g] = best

    # Qualitative regions
    as_winning = compute_almost_sure_winning(product)
    pos_winning = compute_positive_winning(product)

    game_p1 = _project_strategy(product, p1_strat, Player.P1, nba.initial)
    game_p2 = _project_strategy(product, p2_strat, Player.P2, nba.initial)

    init_prob = game_probs[initial_state]

    return OmegaRegularResult(
        formula=formula,
        satisfied=init_prob > 1.0 - tol,
        probabilities=game_probs,
        p1_strategy=p1_strat,
        p2_strategy=p2_strat,
        game_p1_strategy=game_p1,
        game_p2_strategy=game_p2,
        automaton_states=len(nba.states),
        product_states=product.game.n_states,
        iterations=0,
        almost_sure_winning=as_winning,
        positive_winning=pos_winning
    )


# ============================================================
# Labeled Game LTL API (composes V072 LabeledGame)
# ============================================================

def check_ltl_labeled_game(lgame: LabeledGame, formula: LTL,
                            initial_state: int = 0,
                            max_iter: int = 10000,
                            tol: float = 1e-10) -> OmegaRegularResult:
    """Check LTL formula against a LabeledGame (V072 format)."""
    return check_ltl_game_direct(
        lgame.game, lgame.labels, formula,
        initial_state=initial_state, max_iter=max_iter, tol=tol
    )


# ============================================================
# Safety and Liveness Decomposition
# ============================================================

def check_safety_game(game: StochasticGame, labels: Dict[int, Set[str]],
                       prop: str, initial_state: int = 0,
                       max_iter: int = 10000, tol: float = 1e-10) -> OmegaRegularResult:
    """Check G(prop) -- prop holds globally (safety property)."""
    formula = Globally(Atom(prop))
    return check_ltl_game_direct(game, labels, formula, initial_state, max_iter, tol)


def check_liveness_game(game: StochasticGame, labels: Dict[int, Set[str]],
                          prop: str, initial_state: int = 0,
                          max_iter: int = 10000, tol: float = 1e-10) -> OmegaRegularResult:
    """Check F(prop) -- prop holds eventually (liveness property)."""
    formula = Finally(Atom(prop))
    return check_ltl_game_direct(game, labels, formula, initial_state, max_iter, tol)


def check_persistence_game(game: StochasticGame, labels: Dict[int, Set[str]],
                             prop: str, initial_state: int = 0,
                             max_iter: int = 10000, tol: float = 1e-10) -> OmegaRegularResult:
    """Check F(G(prop)) -- prop holds from some point on (persistence)."""
    formula = Finally(Globally(Atom(prop)))
    return check_ltl_game_direct(game, labels, formula, initial_state, max_iter, tol)


def check_recurrence_game(game: StochasticGame, labels: Dict[int, Set[str]],
                            prop: str, initial_state: int = 0,
                            max_iter: int = 10000, tol: float = 1e-10) -> OmegaRegularResult:
    """Check G(F(prop)) -- prop holds infinitely often (recurrence/liveness)."""
    formula = Globally(Finally(Atom(prop)))
    return check_ltl_game_direct(game, labels, formula, initial_state, max_iter, tol)


def check_response_game(game: StochasticGame, labels: Dict[int, Set[str]],
                          trigger: str, response: str, initial_state: int = 0,
                          max_iter: int = 10000, tol: float = 1e-10) -> OmegaRegularResult:
    """Check G(trigger -> F(response)) -- every trigger eventually gets a response."""
    formula = Globally(Implies(Atom(trigger), Finally(Atom(response))))
    return check_ltl_game_direct(game, labels, formula, initial_state, max_iter, tol)


# ============================================================
# Multi-Objective LTL
# ============================================================

def check_multi_ltl_game(game: StochasticGame, labels: Dict[int, Set[str]],
                          formulas: List[LTL], initial_state: int = 0,
                          max_iter: int = 10000, tol: float = 1e-10) -> List[OmegaRegularResult]:
    """Check multiple LTL formulas against a game."""
    results = []
    for f in formulas:
        r = check_ltl_game_direct(game, labels, f, initial_state, max_iter, tol)
        results.append(r)
    return results


def check_conjunction_game(game: StochasticGame, labels: Dict[int, Set[str]],
                            formulas: List[LTL], initial_state: int = 0,
                            max_iter: int = 10000, tol: float = 1e-10) -> OmegaRegularResult:
    """Check conjunction of LTL formulas: f1 AND f2 AND ... AND fn."""
    if len(formulas) == 1:
        return check_ltl_game_direct(game, labels, formulas[0], initial_state, max_iter, tol)

    # Build conjunction
    combined = formulas[0]
    for f in formulas[1:]:
        combined = And(combined, f)

    return check_ltl_game_direct(game, labels, combined, initial_state, max_iter, tol)


# ============================================================
# Strategy Verification
# ============================================================

def verify_ltl_strategy(game: StochasticGame, labels: Dict[int, Set[str]],
                         formula: LTL, strategies: StrategyPair,
                         initial_state: int = 0,
                         n_simulations: int = 1000,
                         max_steps: int = 200) -> Dict:
    """Verify an LTL strategy by inducing a Markov chain and simulating.

    Since LTL is an infinite-horizon property, we use bounded simulation
    and check whether the property appears to hold.
    """
    # Induce Markov chain
    mc = game_to_mc(game, strategies)

    # Simulate and check formula satisfaction
    import random
    formula_nnf_form = nnf(formula)
    sat_count = 0

    for _ in range(n_simulations):
        # Simulate a trace
        trace = _simulate_mc_trace(mc, initial_state, max_steps)
        # Label the trace
        labeled_trace = [labels.get(s, set()) for s in trace]
        # Check LTL on finite prefix (bounded semantics)
        if _check_ltl_finite(formula_nnf_form, labeled_trace, 0):
            sat_count += 1

    return {
        'formula': str(formula),
        'simulations': n_simulations,
        'satisfied': sat_count,
        'satisfaction_rate': sat_count / n_simulations,
        'p1_strategy': strategies.p1_strategy,
        'p2_strategy': strategies.p2_strategy,
    }


def _simulate_mc_trace(mc: MarkovChain, initial: int, max_steps: int) -> List[int]:
    """Simulate a trace from the Markov chain."""
    import random
    trace = [initial]
    state = initial
    for _ in range(max_steps):
        # Sample next state
        r = random.random()
        cumulative = 0.0
        next_state = state
        for t in range(len(mc.transition)):
            cumulative += mc.transition[state][t]
            if r <= cumulative:
                next_state = t
                break
        trace.append(next_state)
        state = next_state
    return trace


def _check_ltl_finite(formula: LTL, trace: List[Set[str]], pos: int) -> bool:
    """Check LTL formula on a finite trace (3-valued: optimistic for infinite extension)."""
    if pos >= len(trace):
        # At end of trace: optimistic for safety, pessimistic for liveness
        if formula.op == LTLOp.TRUE:
            return True
        elif formula.op == LTLOp.FALSE:
            return False
        elif formula.op == LTLOp.ATOM:
            return False  # Unknown
        elif formula.op in (LTLOp.G,):
            return True  # Optimistic: G held so far
        elif formula.op in (LTLOp.F,):
            return False  # Pessimistic: F never happened
        return False

    op = formula.op

    if op == LTLOp.TRUE:
        return True
    elif op == LTLOp.FALSE:
        return False
    elif op == LTLOp.ATOM:
        return formula.name in trace[pos]
    elif op == LTLOp.NOT:
        return not _check_ltl_finite(formula.left, trace, pos)
    elif op == LTLOp.AND:
        return _check_ltl_finite(formula.left, trace, pos) and _check_ltl_finite(formula.right, trace, pos)
    elif op == LTLOp.OR:
        return _check_ltl_finite(formula.left, trace, pos) or _check_ltl_finite(formula.right, trace, pos)
    elif op == LTLOp.X:
        return _check_ltl_finite(formula.left, trace, pos + 1)
    elif op == LTLOp.F:
        for i in range(pos, len(trace)):
            if _check_ltl_finite(formula.left, trace, i):
                return True
        return False
    elif op == LTLOp.G:
        for i in range(pos, len(trace)):
            if not _check_ltl_finite(formula.left, trace, i):
                return False
        return True
    elif op == LTLOp.U:
        for i in range(pos, len(trace)):
            if _check_ltl_finite(formula.right, trace, i):
                return True
            if not _check_ltl_finite(formula.left, trace, i):
                return False
        return False
    elif op == LTLOp.R:
        # a R b = NOT(NOT a U NOT b)
        for i in range(pos, len(trace)):
            if not _check_ltl_finite(formula.right, trace, i):
                return False
            if _check_ltl_finite(formula.left, trace, i):
                return True
        return True  # b held throughout
    elif op == LTLOp.W:
        # a W b = (a U b) OR G(a)
        for i in range(pos, len(trace)):
            if _check_ltl_finite(formula.right, trace, i):
                return True
            if not _check_ltl_finite(formula.left, trace, i):
                return False
        return True  # a held throughout (weak until)
    elif op == LTLOp.IMPLIES:
        return (not _check_ltl_finite(formula.left, trace, pos)) or _check_ltl_finite(formula.right, trace, pos)
    elif op == LTLOp.IFF:
        a = _check_ltl_finite(formula.left, trace, pos)
        b = _check_ltl_finite(formula.right, trace, pos)
        return a == b

    return False


# ============================================================
# Comparison APIs
# ============================================================

def compare_ltl_vs_pctl(lgame: LabeledGame, ltl_formula: LTL,
                         pctl_formula=None,
                         initial_state: int = 0) -> Dict:
    """Compare LTL game checking with PCTL game checking.

    Some LTL properties have PCTL equivalents (e.g., F(p) ~ P>=1[F p]).
    This compares the results.
    """
    from game_pctl import check_game_pctl

    ltl_result = check_ltl_labeled_game(lgame, ltl_formula, initial_state)

    pctl_result = None
    if pctl_formula is not None:
        pctl_result = check_game_pctl(lgame, pctl_formula)

    comparison = {
        'ltl_formula': str(ltl_formula),
        'ltl_prob_state0': ltl_result.probabilities[initial_state],
        'ltl_satisfied': ltl_result.satisfied,
        'automaton_states': ltl_result.automaton_states,
        'product_states': ltl_result.product_states,
    }

    if pctl_result is not None:
        comparison['pctl_formula'] = str(pctl_formula)
        comparison['pctl_satisfying'] = len(pctl_result.satisfying_states)
        if pctl_result.prob_max is not None:
            comparison['pctl_prob_state0'] = pctl_result.prob_max[initial_state]

    return comparison


def compare_direct_vs_negation(game: StochasticGame, labels: Dict[int, Set[str]],
                                formula: LTL, initial_state: int = 0) -> Dict:
    """Compare direct automaton vs negation-based LTL game checking."""
    direct = check_ltl_game_direct(game, labels, formula, initial_state)
    negation = check_ltl_game(game, labels, formula, initial_state)

    return {
        'formula': str(formula),
        'direct_prob': direct.probabilities[initial_state],
        'negation_prob': negation.probabilities[initial_state],
        'direct_aut_states': direct.automaton_states,
        'negation_aut_states': negation.automaton_states,
        'direct_product_states': direct.product_states,
        'negation_product_states': negation.product_states,
        'agree': abs(direct.probabilities[initial_state] - negation.probabilities[initial_state]) < 1e-6,
    }
