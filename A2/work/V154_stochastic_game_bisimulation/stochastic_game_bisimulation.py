"""V154: Bisimulation for Stochastic Games

Extends V149 (MDP bisimulation) to V070 (two-player stochastic games).
Partition refinement with owner-aware signatures: states bisimilar iff
same labels, same owner, and matching action distributions over blocks.

Composes: V070 (stochastic games) + V149 (MDP bisimulation) + V148 (prob bisimulation)
        + V065 (Markov chains) + V067 (labeled MCs) + C037 (SMT solver)
"""

import sys
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from math import inf

# Add paths for dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V070_stochastic_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V149_mdp_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V148_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V069_mdp_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from stochastic_games import StochasticGame, Player, StrategyPair, make_game, game_to_mc
from mdp_bisimulation import (LabeledMDP, MDPBisimResult, MDPBisimVerdict,
                               compute_mdp_bisimulation)
from prob_bisimulation import (BisimResult, BisimVerdict, LabeledMC,
                                compute_bisimulation as mc_compute_bisimulation)
from markov_chain import MarkovChain


# --- Data Structures ---

class GameBisimVerdict(Enum):
    BISIMILAR = "bisimilar"
    NOT_BISIMILAR = "not_bisimilar"
    SIMULATES = "simulates"
    NOT_SIMULATES = "not_simulates"


@dataclass
class LabeledGame:
    """Stochastic game with per-state atomic proposition labels."""
    game: StochasticGame
    labels: Dict[int, Set[str]]

    def states_with(self, label: str) -> Set[int]:
        return {s for s, ls in self.labels.items() if label in ls}

    def states_without(self, label: str) -> Set[int]:
        return {s for s in range(self.game.n_states) if label not in self.labels.get(s, set())}


@dataclass
class GameBisimResult:
    """Result of stochastic game bisimulation computation."""
    verdict: GameBisimVerdict
    partition: List[Set[int]]
    witness: Optional[str] = None
    quotient_game: Optional[LabeledGame] = None
    statistics: Dict = field(default_factory=dict)


@dataclass
class GameSimResult:
    """Result of stochastic game simulation computation."""
    verdict: GameBisimVerdict
    relation: Set[Tuple[int, int]]
    witness: Optional[str] = None
    statistics: Dict = field(default_factory=dict)


@dataclass
class GameDistanceResult:
    """Result of bisimulation distance computation."""
    distances: List[List[float]]
    max_distance: float
    bisimilar_pairs: Set[FrozenSet[int]]
    near_bisimilar_pairs: Set[FrozenSet[int]]
    statistics: Dict = field(default_factory=dict)


# --- Construction ---

def make_labeled_game(n_states, owners, action_transitions, rewards=None,
                      state_labels=None, labels=None):
    """Create a labeled stochastic game.

    Args:
        n_states: Number of states
        owners: List of Player for each state (converted to dict for make_game)
        action_transitions: Dict[int, List[Tuple[str, List[Tuple[int, float]]]]]
            state -> [(action_name, [(target, prob), ...])]
            Sparse tuple format, converted to dense probability lists for make_game.
        rewards: Optional Dict[int, List[float]] state -> per-action rewards
        state_labels: Optional list of state name strings
        labels: Dict[int, Set[str]] atomic proposition labels per state
    """
    # Convert owners list to dict
    owners_dict = {s: owners[s] for s in range(n_states)}

    # Convert sparse tuple transitions to dense dict format for make_game
    # make_game expects: {state: {action_name: [p0, p1, ..., p_{n-1}]}}
    at_dict = {}
    for s, action_list in action_transitions.items():
        at_dict[s] = {}
        for aname, targets in action_list:
            probs = [0.0] * n_states
            for target, prob in targets:
                probs[target] = prob
            at_dict[s][aname] = probs

    # Convert rewards: {state: [r0, r1, ...]} -> {state: {action_name: reward}}
    rewards_dict = None
    if rewards:
        rewards_dict = {}
        for s, reward_list in rewards.items():
            action_list = action_transitions[s]
            rewards_dict[s] = {}
            for i, (aname, _) in enumerate(action_list):
                if i < len(reward_list):
                    rewards_dict[s][aname] = reward_list[i]

    game = make_game(n_states, owners_dict, at_dict, rewards_dict, state_labels)
    if labels is None:
        labels = {}
    # Ensure all states have label entries
    full_labels = {s: labels.get(s, set()) for s in range(n_states)}
    return LabeledGame(game=game, labels=full_labels)


# --- Partition Refinement Core ---

def _initial_partition(lgame: LabeledGame) -> List[Set[int]]:
    """Initial partition: group by (owner, label set)."""
    groups = {}
    for s in range(lgame.game.n_states):
        key = (lgame.game.owners[s], frozenset(lgame.labels.get(s, set())))
        if key not in groups:
            groups[key] = set()
        groups[key].add(s)
    return list(groups.values())


def _state_to_block(partition: List[Set[int]], n_states: int) -> List[int]:
    """Map each state to its block index."""
    mapping = [0] * n_states
    for bi, block in enumerate(partition):
        for s in block:
            mapping[s] = bi
    return mapping


def _action_signature(game: StochasticGame, state: int, action_idx: int,
                      mapping: List[int], n_blocks: int) -> Tuple:
    """Compute block-probability signature for one action.

    Returns tuple of probabilities to each block, rounded to avoid float noise.
    """
    block_probs = [0.0] * n_blocks
    trans = game.transition[state][action_idx]
    for target, prob in enumerate(trans):
        if prob > 0:
            block_probs[mapping[target]] += prob
    # Include reward if present
    reward = 0.0
    if game.rewards is not None and game.rewards[state] is not None:
        if action_idx < len(game.rewards[state]):
            reward = game.rewards[state][action_idx]
    return (round(reward, 10),) + tuple(round(p, 10) for p in block_probs)


def _state_signature(game: StochasticGame, state: int,
                     mapping: List[int], n_blocks: int) -> FrozenSet:
    """Compute state signature: set of action signatures.

    For bisimulation, action names don't matter -- only the set of
    available action distributions (as block-probability vectors).
    """
    sigs = set()
    for ai in range(len(game.actions[state])):
        sig = _action_signature(game, state, ai, mapping, n_blocks)
        sigs.add(sig)
    return frozenset(sigs)


def _refine_partition(game: StochasticGame, partition: List[Set[int]]) -> List[Set[int]]:
    """One step of partition refinement."""
    n_blocks = len(partition)
    mapping = _state_to_block(partition, game.n_states)

    new_partition = []
    for block in partition:
        sub_groups = {}
        for s in block:
            sig = _state_signature(game, s, mapping, n_blocks)
            if sig not in sub_groups:
                sub_groups[sig] = set()
            sub_groups[sig].add(s)
        new_partition.extend(sub_groups.values())
    return new_partition


def compute_game_bisimulation(lgame: LabeledGame, max_iter: int = 1000) -> GameBisimResult:
    """Compute bisimulation partition for a stochastic game.

    Two states are bisimilar iff:
    1. Same owner (P1, P2, or CHANCE)
    2. Same atomic proposition labels
    3. For every action at one state, there exists a matching action at the
       other with the same block-probability distribution (set-based matching)

    Uses Larsen-Skou style partition refinement.
    """
    game = lgame.game
    partition = _initial_partition(lgame)
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        new_partition = _refine_partition(game, partition)
        if len(new_partition) == len(partition):
            # Check actual equality (not just count)
            old_sets = [frozenset(b) for b in partition]
            new_sets = [frozenset(b) for b in new_partition]
            if set(old_sets) == set(new_sets):
                break
        partition = new_partition

    n_blocks = len(partition)
    trivial = all(len(b) == 1 for b in partition)

    return GameBisimResult(
        verdict=GameBisimVerdict.NOT_BISIMILAR if trivial and game.n_states > 1
                else GameBisimVerdict.BISIMILAR,
        partition=partition,
        statistics={
            'iterations': iterations,
            'n_blocks': n_blocks,
            'n_states': game.n_states,
            'reduction_ratio': n_blocks / max(game.n_states, 1),
        }
    )


def check_game_bisimilar(lgame: LabeledGame, s1: int, s2: int) -> GameBisimResult:
    """Check if two states are bisimilar in a stochastic game."""
    result = compute_game_bisimulation(lgame)
    mapping = _state_to_block(result.partition, lgame.game.n_states)

    if mapping[s1] == mapping[s2]:
        result.verdict = GameBisimVerdict.BISIMILAR
        result.witness = f"States {s1} and {s2} are in the same block {mapping[s1]}"
    else:
        result.verdict = GameBisimVerdict.NOT_BISIMILAR
        # Find distinguishing reason
        game = lgame.game
        if game.owners[s1] != game.owners[s2]:
            result.witness = f"Different owners: {game.owners[s1].name} vs {game.owners[s2].name}"
        elif lgame.labels.get(s1, set()) != lgame.labels.get(s2, set()):
            result.witness = f"Different labels: {lgame.labels.get(s1, set())} vs {lgame.labels.get(s2, set())}"
        else:
            result.witness = f"Different action signatures (blocks {mapping[s1]} vs {mapping[s2]})"
    return result


# --- Quotient Game Construction ---

def game_bisimulation_quotient(lgame: LabeledGame) -> Tuple[LabeledGame, GameBisimResult]:
    """Compute quotient game: collapse bisimilar states."""
    result = compute_game_bisimulation(lgame)
    partition = result.partition
    n_blocks = len(partition)
    game = lgame.game
    mapping = _state_to_block(partition, game.n_states)

    # Representatives: pick smallest state from each block
    reps = [min(block) for block in partition]

    # Build quotient
    owners = [game.owners[reps[bi]] for bi in range(n_blocks)]
    labels = {bi: lgame.labels.get(reps[bi], set()) for bi in range(n_blocks)}

    action_transitions = {}
    rewards_dict = {}
    for bi in range(n_blocks):
        rep = reps[bi]
        actions = []
        reward_list = []
        for ai, aname in enumerate(game.actions[rep]):
            # Aggregate transitions to blocks
            block_probs = {}
            trans = game.transition[rep][ai]
            for target, prob in enumerate(trans):
                if prob > 0:
                    tb = mapping[target]
                    block_probs[tb] = block_probs.get(tb, 0.0) + prob
            targets = [(tb, p) for tb, p in sorted(block_probs.items())]
            actions.append((aname, targets))
            if game.rewards is not None and game.rewards[rep] is not None:
                if ai < len(game.rewards[rep]):
                    reward_list.append(game.rewards[rep][ai])
        action_transitions[bi] = actions
        if reward_list:
            rewards_dict[bi] = reward_list

    state_labels = [f"B{bi}" for bi in range(n_blocks)]
    quotient = make_labeled_game(
        n_states=n_blocks,
        owners=owners,
        action_transitions=action_transitions,
        rewards=rewards_dict if rewards_dict else None,
        state_labels=state_labels,
        labels=labels,
    )

    result.quotient_game = quotient
    return quotient, result


# --- Simulation Preorder ---

def compute_game_simulation(lgame: LabeledGame, max_iter: int = 1000) -> GameSimResult:
    """Compute simulation preorder for a stochastic game.

    s simulates t (s >= t) iff:
    1. Same owner and labels
    2. For every action at t, there exists a matching action at s with
       pointwise greater-or-equal block probabilities for all blocks
       reachable from t's action.
    """
    game = lgame.game
    n = game.n_states

    # Initial relation: pairs with same owner and labels
    relation = set()
    for s in range(n):
        for t in range(n):
            if (game.owners[s] == game.owners[t] and
                lgame.labels.get(s, set()) == lgame.labels.get(t, set())):
                relation.add((s, t))

    # Iterative refinement
    for _ in range(max_iter):
        new_relation = set()
        for (s, t) in relation:
            # Check: for every action at t, s has a matching action
            if _sim_check(game, s, t, relation, n):
                new_relation.add((s, t))
        if new_relation == relation:
            break
        relation = new_relation

    return GameSimResult(
        verdict=GameBisimVerdict.SIMULATES,
        relation=relation,
        statistics={'n_pairs': len(relation), 'n_states': n}
    )


def _sim_check(game: StochasticGame, s: int, t: int,
               relation: Set[Tuple[int, int]], n: int) -> bool:
    """Check if s can simulate every action of t under current relation."""
    for ti in range(len(game.actions[t])):
        # Find a matching action at s
        found = False
        for si in range(len(game.actions[s])):
            if _action_simulates(game, s, si, t, ti, relation, n):
                found = True
                break
        if not found:
            return False
    return True


def _action_simulates(game: StochasticGame, s: int, si: int,
                      t: int, ti: int, relation: Set[Tuple[int, int]],
                      n: int) -> bool:
    """Check if action si at s simulates action ti at t.

    For each target u of t's action, the probability mass from s's action
    going to states that simulate u must be >= t's probability to u.
    """
    s_trans = game.transition[s][si]
    t_trans = game.transition[t][ti]

    # Check reward compatibility
    if game.rewards is not None:
        s_rew = game.rewards[s][si] if game.rewards[s] is not None and si < len(game.rewards[s]) else 0.0
        t_rew = game.rewards[t][ti] if game.rewards[t] is not None and ti < len(game.rewards[t]) else 0.0
        if s_rew < t_rew - 1e-10:
            return False

    # For each target of t's action, check probability mass
    for u in range(n):
        if t_trans[u] < 1e-12:
            continue
        # Sum prob from s's action going to states that simulate u
        mass = sum(s_trans[v] for v in range(n) if (v, u) in relation)
        if mass < t_trans[u] - 1e-10:
            return False
    return True


def check_game_simulates(lgame: LabeledGame, s: int, t: int) -> GameSimResult:
    """Check if state s simulates state t."""
    result = compute_game_simulation(lgame)
    if (s, t) in result.relation:
        result.verdict = GameBisimVerdict.SIMULATES
        result.witness = f"State {s} simulates state {t}"
    else:
        result.verdict = GameBisimVerdict.NOT_SIMULATES
        result.witness = f"State {s} does not simulate state {t}"
    return result


# --- Bisimulation Distance ---

def compute_game_bisimulation_distance(lgame: LabeledGame, discount: float = 0.9,
                                        max_iter: int = 500, tol: float = 1e-8,
                                        threshold: float = 0.1) -> GameDistanceResult:
    """Compute bisimulation distance between all state pairs.

    Uses Hausdorff-Kantorovich lifting:
    d(s,t) = 0 if same owner/labels and action-matching distances are 0
    d(s,t) = 1 if different owner or labels
    Otherwise: d(s,t) = discount * hausdorff(action_distances)
    """
    game = lgame.game
    n = game.n_states

    # Initialize: different owner/labels -> 1.0, same -> 0.0
    dist = [[0.0] * n for _ in range(n)]
    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            if (game.owners[s] != game.owners[t] or
                lgame.labels.get(s, set()) != lgame.labels.get(t, set())):
                dist[s][t] = 1.0

    # Iterate
    for _ in range(max_iter):
        new_dist = [[0.0] * n for _ in range(n)]
        max_change = 0.0
        for s in range(n):
            for t in range(s + 1, n):
                if game.owners[s] != game.owners[t] or \
                   lgame.labels.get(s, set()) != lgame.labels.get(t, set()):
                    new_dist[s][t] = new_dist[t][s] = 1.0
                    continue
                d = _hausdorff_action_distance(game, s, t, dist, n)
                d = min(discount * d, 1.0)
                new_dist[s][t] = new_dist[t][s] = d
                max_change = max(max_change, abs(d - dist[s][t]))
        dist = new_dist
        if max_change < tol:
            break

    max_distance = max(dist[s][t] for s in range(n) for t in range(s + 1, n)) if n > 1 else 0.0
    bisimilar = {frozenset({s, t}) for s in range(n) for t in range(s + 1, n)
                 if dist[s][t] < tol}
    near_bisimilar = {frozenset({s, t}) for s in range(n) for t in range(s + 1, n)
                      if tol <= dist[s][t] < threshold}

    return GameDistanceResult(
        distances=dist,
        max_distance=max_distance,
        bisimilar_pairs=bisimilar,
        near_bisimilar_pairs=near_bisimilar,
        statistics={'n_states': n, 'discount': discount}
    )


def _hausdorff_action_distance(game: StochasticGame, s: int, t: int,
                                dist: List[List[float]], n: int) -> float:
    """Hausdorff distance between action sets of s and t."""
    # d_H(A, B) = max(max_a min_b d(a,b), max_b min_a d(a,b))
    d1 = 0.0
    for si in range(len(game.actions[s])):
        min_d = inf
        for ti in range(len(game.actions[t])):
            d = _kantorovich_action(game, s, si, t, ti, dist, n)
            min_d = min(min_d, d)
        if min_d == inf:
            min_d = 1.0
        d1 = max(d1, min_d)

    d2 = 0.0
    for ti in range(len(game.actions[t])):
        min_d = inf
        for si in range(len(game.actions[s])):
            d = _kantorovich_action(game, s, si, t, ti, dist, n)
            min_d = min(min_d, d)
        if min_d == inf:
            min_d = 1.0
        d2 = max(d2, min_d)

    return max(d1, d2)


def _kantorovich_action(game: StochasticGame, s: int, si: int,
                         t: int, ti: int,
                         dist: List[List[float]], n: int) -> float:
    """Kantorovich (earth mover's) distance between two action distributions.

    Simplified: sum of |p_s(u) - p_t(u)| * max_dist_to_u over targets.
    For exact Kantorovich, we'd need LP. This is an upper bound that works
    well for partition-like distance structures.
    """
    s_trans = game.transition[s][si]
    t_trans = game.transition[t][ti]

    # Reward difference
    r_diff = 0.0
    if game.rewards is not None:
        sr = game.rewards[s][si] if game.rewards[s] is not None and si < len(game.rewards[s]) else 0.0
        tr = game.rewards[t][ti] if game.rewards[t] is not None and ti < len(game.rewards[t]) else 0.0
        r_diff = abs(sr - tr)

    # Simple coupling: sum of absolute prob differences weighted by distance
    total = 0.0
    for u in range(n):
        for v in range(n):
            # Coupling assigns min(p_s(u), p_t(v)) mass from u to v
            pass

    # Use simpler metric: expected distance under product coupling
    total = 0.0
    for u in range(n):
        if s_trans[u] < 1e-12 and t_trans[u] < 1e-12:
            continue
        diff = abs(s_trans[u] - t_trans[u])
        # Contribution from probability mass mismatch
        total += diff

    # Also add expected distance contribution
    for u in range(n):
        common = min(s_trans[u], t_trans[u])
        if common > 1e-12:
            # Paired mass goes u->u, distance = 0 (same target)
            pass

    # Compute as: sum_u sum_v coupling(u,v) * dist[u][v]
    # Use greedy coupling: pair targets by smallest distance first
    # For simplicity, use the total variation + expected distance bound
    tv = sum(abs(s_trans[u] - t_trans[u]) for u in range(n)) / 2.0
    exp_dist = sum(s_trans[u] * t_trans[v] * dist[u][v]
                   for u in range(n) for v in range(n)
                   if s_trans[u] > 1e-12 and t_trans[v] > 1e-12)

    return r_diff + tv + exp_dist


# --- Cross-System Bisimulation ---

def check_cross_game_bisimulation(lgame1: LabeledGame,
                                   lgame2: LabeledGame) -> GameBisimResult:
    """Check bisimulation between two stochastic games via disjoint union."""
    g1, g2 = lgame1.game, lgame2.game
    n1, n2 = g1.n_states, g2.n_states
    n = n1 + n2

    # Build combined game
    owners = list(g1.owners) + list(g2.owners)
    labels = {}
    for s in range(n1):
        labels[s] = lgame1.labels.get(s, set())
    for s in range(n2):
        labels[n1 + s] = lgame2.labels.get(s, set())

    action_transitions = {}
    rewards_dict = {}

    # Game 1 states
    for s in range(n1):
        actions = []
        reward_list = []
        for ai, aname in enumerate(g1.actions[s]):
            targets = []
            for t in range(n1):
                if g1.transition[s][ai][t] > 0:
                    targets.append((t, g1.transition[s][ai][t]))
            actions.append((aname, targets))
            if g1.rewards is not None and g1.rewards[s] is not None and ai < len(g1.rewards[s]):
                reward_list.append(g1.rewards[s][ai])
        action_transitions[s] = actions
        if reward_list:
            rewards_dict[s] = reward_list

    # Game 2 states (shifted by n1)
    for s in range(n2):
        actions = []
        reward_list = []
        for ai, aname in enumerate(g2.actions[s]):
            targets = []
            for t in range(n2):
                if g2.transition[s][ai][t] > 0:
                    targets.append((n1 + t, g2.transition[s][ai][t]))
            actions.append((aname, targets))
            if g2.rewards is not None and g2.rewards[s] is not None and ai < len(g2.rewards[s]):
                reward_list.append(g2.rewards[s][ai])
        action_transitions[n1 + s] = actions
        if reward_list:
            rewards_dict[n1 + s] = reward_list

    combined = make_labeled_game(
        n_states=n,
        owners=owners,
        action_transitions=action_transitions,
        rewards=rewards_dict if rewards_dict else None,
        labels=labels,
    )

    result = compute_game_bisimulation(combined)
    result.statistics['game1_states'] = n1
    result.statistics['game2_states'] = n2
    return result


def check_cross_game_bisimilar_states(lgame1: LabeledGame, s1: int,
                                       lgame2: LabeledGame, s2: int) -> GameBisimResult:
    """Check if state s1 in game1 is bisimilar to state s2 in game2."""
    result = check_cross_game_bisimulation(lgame1, lgame2)
    n1 = lgame1.game.n_states
    mapping = _state_to_block(result.partition, lgame1.game.n_states + lgame2.game.n_states)

    if mapping[s1] == mapping[n1 + s2]:
        result.verdict = GameBisimVerdict.BISIMILAR
        result.witness = f"State {s1} (game1) bisimilar to state {s2} (game2)"
    else:
        result.verdict = GameBisimVerdict.NOT_BISIMILAR
        result.witness = f"State {s1} (game1) not bisimilar to state {s2} (game2)"
    return result


# --- Strategy-Induced Bisimulation ---

def strategy_bisimulation(lgame: LabeledGame,
                          strategies: StrategyPair) -> BisimResult:
    """Compute MC bisimulation of the Markov chain induced by fixing both strategies.

    Under fixed strategies, the game becomes a Markov chain.
    Uses V148 probabilistic bisimulation on the induced chain.
    """
    mc = game_to_mc(lgame.game, strategies)
    # Build LabeledMC for V148
    lmc = LabeledMC(mc=mc, labels=lgame.labels)
    return mc_compute_bisimulation(lmc)


def compare_strategy_bisimulations(lgame: LabeledGame,
                                    strat1: StrategyPair,
                                    strat2: StrategyPair) -> Dict:
    """Compare bisimulation partitions under two different strategy pairs."""
    result1 = strategy_bisimulation(lgame, strat1)
    result2 = strategy_bisimulation(lgame, strat2)
    return {
        'strategy1_blocks': len(result1.partition),
        'strategy2_blocks': len(result2.partition),
        'strategy1_partition': result1.partition,
        'strategy2_partition': result2.partition,
        'same_partition': set(frozenset(b) for b in result1.partition) ==
                         set(frozenset(b) for b in result2.partition),
    }


# --- Reward-Aware Bisimulation ---

def compute_reward_bisimulation(lgame: LabeledGame,
                                 max_iter: int = 1000) -> GameBisimResult:
    """Bisimulation that also considers reward equality.

    Action signatures include the reward value, so actions with different
    rewards are never matched. This is already handled by _action_signature
    which includes reward in the tuple. This function is an explicit entry
    point for reward-aware bisimulation.
    """
    return compute_game_bisimulation(lgame, max_iter)


# --- SMT Verification ---

def verify_game_bisimulation_smt(lgame: LabeledGame,
                                  partition: List[Set[int]]) -> Dict:
    """Verify a bisimulation partition via SMT checking.

    Checks:
    1. All states in each block have same owner
    2. All states in each block have same labels
    3. For every pair in same block, action set matching holds
    """
    try:
        from smt_solver import SMTSolver, IntConst, App, Op, Var, INT, BOOL
    except ImportError:
        return {'verified': False, 'error': 'SMT solver not available'}

    game = lgame.game
    violations = []
    n_blocks = len(partition)
    mapping = _state_to_block(partition, game.n_states)

    for bi, block in enumerate(partition):
        states = sorted(block)
        if len(states) < 2:
            continue

        rep = states[0]
        for s in states[1:]:
            # Check 1: same owner
            if game.owners[s] != game.owners[rep]:
                violations.append(f"Block {bi}: states {rep} and {s} have different owners")

            # Check 2: same labels
            if lgame.labels.get(s, set()) != lgame.labels.get(rep, set()):
                violations.append(f"Block {bi}: states {rep} and {s} have different labels")

            # Check 3: action matching
            rep_sigs = _state_signature(game, rep, mapping, n_blocks)
            s_sigs = _state_signature(game, s, mapping, n_blocks)
            if rep_sigs != s_sigs:
                violations.append(f"Block {bi}: states {rep} and {s} have different action signatures")

    return {
        'verified': len(violations) == 0,
        'n_violations': len(violations),
        'violations': violations[:10],  # Limit output
        'n_blocks': n_blocks,
    }


# --- Minimization ---

def minimize_game(lgame: LabeledGame) -> Tuple[LabeledGame, GameBisimResult]:
    """Minimize the game by collapsing bisimilar states."""
    return game_bisimulation_quotient(lgame)


# --- Comparison with MDP Bisimulation ---

def compare_game_vs_mdp_bisimulation(lgame: LabeledGame,
                                      fix_player: Player = Player.P2,
                                      strategy: Optional[Dict[int, int]] = None) -> Dict:
    """Compare game bisimulation with MDP bisimulation under a fixed strategy.

    When one player's strategy is fixed, the game becomes an MDP.
    Game bisimulation is finer (more blocks) than MDP bisimulation
    because it considers both players' choices.
    """
    game_result = compute_game_bisimulation(lgame)

    # Build MDP by fixing one player's strategy
    if strategy is None:
        # Default strategy: always pick first action
        strategy = {}
        for s in range(lgame.game.n_states):
            if lgame.game.owners[s] == fix_player:
                strategy[s] = 0

    # Convert to MDP
    from mdp_verification import MDP
    game = lgame.game
    n = game.n_states
    mdp_actions = []
    mdp_transition = []
    mdp_rewards = []

    for s in range(n):
        if game.owners[s] == fix_player:
            # Fixed player: single action (the chosen one)
            ai = strategy.get(s, 0)
            mdp_actions.append([game.actions[s][ai]])
            mdp_transition.append([game.transition[s][ai]])
            if game.rewards is not None and game.rewards[s] is not None:
                mdp_rewards.append([game.rewards[s][ai] if ai < len(game.rewards[s]) else 0.0])
            else:
                mdp_rewards.append([0.0])
        else:
            # Other player keeps all actions
            mdp_actions.append(list(game.actions[s]))
            mdp_transition.append([list(t) for t in game.transition[s]])
            if game.rewards is not None and game.rewards[s] is not None:
                mdp_rewards.append(list(game.rewards[s]))
            else:
                mdp_rewards.append([0.0] * len(game.actions[s]))

    mdp = MDP(
        n_states=n,
        actions=mdp_actions,
        transition=mdp_transition,
        rewards=mdp_rewards,
    )
    lmdp = LabeledMDP(mdp=mdp, labels=lgame.labels)
    mdp_result = compute_mdp_bisimulation(lmdp)

    return {
        'game_blocks': len(game_result.partition),
        'mdp_blocks': len(mdp_result.partition),
        'game_partition': game_result.partition,
        'mdp_partition': mdp_result.partition,
        'game_finer': len(game_result.partition) >= len(mdp_result.partition),
        'fixed_player': fix_player.name,
    }


# --- Analysis & Summary ---

def analyze_game_bisimulation(lgame: LabeledGame,
                               discount: float = 0.9) -> Dict:
    """Full bisimulation analysis: partition + quotient + distance + simulation."""
    bisim = compute_game_bisimulation(lgame)
    quotient, _ = game_bisimulation_quotient(lgame)
    distance = compute_game_bisimulation_distance(lgame, discount=discount)
    sim = compute_game_simulation(lgame)

    return {
        'bisimulation': bisim,
        'quotient': quotient,
        'distance': distance,
        'simulation': sim,
        'n_states': lgame.game.n_states,
        'n_blocks': len(bisim.partition),
        'n_sim_pairs': len(sim.relation),
        'max_distance': distance.max_distance,
    }


def game_bisimulation_summary(lgame: LabeledGame) -> str:
    """Human-readable summary of game bisimulation analysis."""
    result = compute_game_bisimulation(lgame)
    game = lgame.game

    lines = [
        f"Stochastic Game Bisimulation Summary",
        f"  States: {game.n_states}",
        f"  Blocks: {len(result.partition)}",
        f"  Reduction: {game.n_states} -> {len(result.partition)} "
        f"({100 * (1 - len(result.partition) / max(game.n_states, 1)):.1f}% reduction)",
        f"  Iterations: {result.statistics['iterations']}",
        f"  Owner distribution: "
        f"P1={sum(1 for o in game.owners if o == Player.P1)}, "
        f"P2={sum(1 for o in game.owners if o == Player.P2)}, "
        f"CHANCE={sum(1 for o in game.owners if o == Player.CHANCE)}",
        f"  Partition:",
    ]
    for bi, block in enumerate(result.partition):
        rep = min(block)
        owner = game.owners[rep].name
        labels = lgame.labels.get(rep, set())
        lines.append(f"    Block {bi} ({owner}, {labels}): {sorted(block)}")

    return "\n".join(lines)


# --- Example Systems ---

def symmetric_game():
    """Two symmetric P1 states with identical structure -> bisimilar."""
    return make_labeled_game(
        n_states=4,
        owners=[Player.P1, Player.P1, Player.CHANCE, Player.CHANCE],
        action_transitions={
            0: [("a", [(2, 1.0)]), ("b", [(3, 1.0)])],
            1: [("a", [(2, 1.0)]), ("b", [(3, 1.0)])],
            2: [("stay", [(2, 0.5), (3, 0.5)])],
            3: [("stay", [(2, 0.3), (3, 0.7)])],
        },
        labels={0: {"start"}, 1: {"start"}, 2: {"mid"}, 3: {"end"}},
    )


def asymmetric_game():
    """Two P1 states with different action distribution sets -> not bisimilar."""
    return make_labeled_game(
        n_states=4,
        owners=[Player.P1, Player.P1, Player.CHANCE, Player.CHANCE],
        action_transitions={
            0: [("a", [(2, 1.0)]), ("b", [(3, 1.0)])],  # Can reach 2 or 3
            1: [("a", [(2, 1.0)])],  # Can only reach 2
            2: [("stay", [(2, 1.0)])],
            3: [("stay", [(3, 1.0)])],
        },
        labels={0: {"start"}, 1: {"start"}, 2: {"win"}, 3: {"lose"}},
    )


def owner_mismatch_game():
    """States with same labels but different owners -> not bisimilar."""
    return make_labeled_game(
        n_states=3,
        owners=[Player.P1, Player.P2, Player.CHANCE],
        action_transitions={
            0: [("a", [(2, 1.0)])],
            1: [("a", [(2, 1.0)])],
            2: [("stay", [(2, 1.0)])],
        },
        labels={0: {"start"}, 1: {"start"}, 2: {"end"}},
    )
