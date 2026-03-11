"""
V163: Symbolic Mean-Payoff Games

BDD-based symbolic solving of mean-payoff parity games. Encodes game graphs,
weights, priorities, and ownership as BDDs, then performs symbolic operations
(attractor computation, reachability) alongside explicit value iteration for
the numeric mean-payoff/energy computations.

Key idea: vertex sets and transition relations are BDD-encoded for efficient
attractor and reachability operations, while energy/mean-payoff values remain
explicit (since they are numeric, not naturally Boolean).

Composes: V021 (BDD model checking) + V161 (Mean-Payoff Parity Games) + V160 (Energy Games).

Algorithms:
1. Symbolic encoding of MPP games (vertices, edges, owners, priorities as BDDs)
2. Symbolic attractor computation (fixpoint BDD iteration)
3. Symbolic mean-payoff parity solving (Zielonka parity via BDD + energy check)
4. Symbolic reachability and safety checking
5. Symbolic decomposition analysis (parity-only vs MP-only vs combined)
6. Comparison APIs: symbolic vs explicit (V161) side-by-side
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V160_energy_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V161_mean_payoff_parity'))

from bdd_model_checker import BDD, BDDNode
from parity_games import Player, ParityGame, Solution, zielonka, attractor
from energy_games import EnergyGame, solve_energy
from mean_payoff_parity import (
    MeanPayoffParityGame, MPPResult,
    solve_mpp, solve_mpp_threshold, compute_mpp_values,
    verify_mpp_strategy, simulate_play, decompose_mpp,
    make_chain_mpp, make_choice_mpp, make_adversarial_mpp,
)


# ============================================================
# Symbolic Encoding
# ============================================================

@dataclass
class SymbolicMPPEncoding:
    """BDD encoding of a mean-payoff parity game's structure."""
    bdd: BDD
    num_bits: int                          # bits per vertex index
    curr_indices: List[int]                # BDD var indices for current vertex
    next_indices: List[int]                # BDD var indices for next vertex
    owner_even: BDDNode                    # BDD: vertices owned by Even
    owner_odd: BDDNode                     # BDD: vertices owned by Odd
    vertices_bdd: BDDNode                  # BDD: valid vertices
    # Edge encoding: list of (guard_bdd, weight) where guard_bdd is over curr+next vars
    edges: List[Tuple[BDDNode, int]]
    # Weight groups: weight -> BDD over (curr, next) for all edges with that weight
    weight_groups: Dict[int, BDDNode]
    # Priority groups: priority -> BDD over curr for all vertices with that priority
    priority_groups: Dict[int, BDDNode]
    # Vertex count
    num_vertices: int
    # Mapping from vertex id to bit assignment
    vertex_bits: Dict[int, Tuple[bool, ...]]
    # Mapping from vertex id to contiguous index
    vert_to_idx: Dict[int, int]
    idx_to_vert: Dict[int, int]


def _int_to_bits(val: int, num_bits: int) -> Tuple[bool, ...]:
    """Convert integer to little-endian bit tuple."""
    return tuple(bool((val >> i) & 1) for i in range(num_bits))


def _make_vertex_bdd(bdd: BDD, val: int, var_indices: List[int], num_bits: int) -> BDDNode:
    """Create BDD that is true exactly when var_indices encode 'val'."""
    bits = _int_to_bits(val, num_bits)
    result = bdd.TRUE
    for i, idx in enumerate(var_indices):
        v = bdd.var(idx)
        if bits[i]:
            result = bdd.AND(result, v)
        else:
            result = bdd.AND(result, bdd.NOT(v))
    return result


def encode_mpp_game(game: MeanPayoffParityGame) -> SymbolicMPPEncoding:
    """Encode an explicit mean-payoff parity game as BDDs."""
    verts = sorted(game.vertices)
    n = len(verts)
    if n == 0:
        raise ValueError("Empty game")
    num_bits = max(1, math.ceil(math.log2(n + 1)))

    # Map vertex ids to contiguous indices 0..n-1
    vert_to_idx = {v: i for i, v in enumerate(verts)}
    idx_to_vert = {i: v for v, i in vert_to_idx.items()}

    # Create BDD with curr + next variables
    total_vars = 2 * num_bits
    bdd = BDD(total_vars)
    curr_indices = list(range(num_bits))
    next_indices = list(range(num_bits, 2 * num_bits))

    # Precompute vertex BDDs
    vertex_curr_bdds = {}
    vertex_next_bdds = {}
    vertex_bits_map = {}
    for v in verts:
        idx = vert_to_idx[v]
        vertex_bits_map[v] = _int_to_bits(idx, num_bits)
        vertex_curr_bdds[v] = _make_vertex_bdd(bdd, idx, curr_indices, num_bits)
        vertex_next_bdds[v] = _make_vertex_bdd(bdd, idx, next_indices, num_bits)

    # Valid vertices BDD
    vertices_bdd = bdd.FALSE
    for v in verts:
        vertices_bdd = bdd.OR(vertices_bdd, vertex_curr_bdds[v])

    # Owner BDDs
    owner_even = bdd.FALSE
    owner_odd = bdd.FALSE
    for v in verts:
        if game.owner[v] == Player.EVEN:
            owner_even = bdd.OR(owner_even, vertex_curr_bdds[v])
        else:
            owner_odd = bdd.OR(owner_odd, vertex_curr_bdds[v])

    # Priority BDDs
    priority_groups: Dict[int, BDDNode] = {}
    for v in verts:
        p = game.priority.get(v, 0)
        if p not in priority_groups:
            priority_groups[p] = bdd.FALSE
        priority_groups[p] = bdd.OR(priority_groups[p], vertex_curr_bdds[v])

    # Edge encoding
    edge_list = []
    weight_groups: Dict[int, BDDNode] = {}
    for u in verts:
        for (t, w) in game.edges.get(u, []):
            if t not in vert_to_idx:
                continue
            guard = bdd.AND(vertex_curr_bdds[u], vertex_next_bdds[t])
            edge_list.append((guard, w))
            if w not in weight_groups:
                weight_groups[w] = bdd.FALSE
            weight_groups[w] = bdd.OR(weight_groups[w], guard)

    return SymbolicMPPEncoding(
        bdd=bdd,
        num_bits=num_bits,
        curr_indices=curr_indices,
        next_indices=next_indices,
        owner_even=owner_even,
        owner_odd=owner_odd,
        vertices_bdd=vertices_bdd,
        edges=edge_list,
        weight_groups=weight_groups,
        priority_groups=priority_groups,
        num_vertices=n,
        vertex_bits=vertex_bits_map,
        vert_to_idx=vert_to_idx,
        idx_to_vert=idx_to_vert,
    )


# ============================================================
# Symbolic Operations
# ============================================================

def _symbolic_successors(enc: SymbolicMPPEncoding, states: BDDNode) -> BDDNode:
    """Compute set of vertices reachable in one step from 'states'."""
    bdd = enc.bdd
    edge_rel = bdd.FALSE
    for guard, _w in enc.edges:
        edge_rel = bdd.OR(edge_rel, guard)
    combined = bdd.AND(states, edge_rel)
    result = bdd.exists_multi(enc.curr_indices, combined)
    rename_map = {enc.next_indices[i]: enc.curr_indices[i] for i in range(enc.num_bits)}
    return bdd.rename(result, rename_map)


def _symbolic_predecessors(enc: SymbolicMPPEncoding, targets: BDDNode) -> BDDNode:
    """Compute set of vertices that have an edge to 'targets'."""
    bdd = enc.bdd
    edge_rel = bdd.FALSE
    for guard, _w in enc.edges:
        edge_rel = bdd.OR(edge_rel, guard)
    rename_map = {enc.curr_indices[i]: enc.next_indices[i] for i in range(enc.num_bits)}
    targets_next = bdd.rename(targets, rename_map)
    combined = bdd.AND(targets_next, edge_rel)
    return bdd.exists_multi(enc.next_indices, combined)


def _symbolic_has_successor(enc: SymbolicMPPEncoding, states: BDDNode) -> BDDNode:
    """Return subset of 'states' that has at least one successor."""
    bdd = enc.bdd
    edge_rel = bdd.FALSE
    for guard, _w in enc.edges:
        edge_rel = bdd.OR(edge_rel, guard)
    combined = bdd.AND(states, edge_rel)
    has_succ = bdd.exists_multi(enc.next_indices, combined)
    return bdd.AND(states, has_succ)


def _symbolic_all_succ_in(enc: SymbolicMPPEncoding, states: BDDNode, target: BDDNode) -> BDDNode:
    """Return subset of 'states' where ALL successors are in 'target'."""
    bdd = enc.bdd
    edge_rel = bdd.FALSE
    for guard, _w in enc.edges:
        edge_rel = bdd.OR(edge_rel, guard)
    rename_map = {enc.curr_indices[i]: enc.next_indices[i] for i in range(enc.num_bits)}
    target_next = bdd.rename(target, rename_map)
    bad_succ = bdd.AND(edge_rel, bdd.NOT(target_next))
    has_bad = bdd.exists_multi(enc.next_indices, bad_succ)
    return bdd.AND(states, bdd.NOT(has_bad))


def _symbolic_some_succ_in(enc: SymbolicMPPEncoding, states: BDDNode, target: BDDNode) -> BDDNode:
    """Return subset of 'states' that has at least one successor in 'target'."""
    bdd = enc.bdd
    edge_rel = bdd.FALSE
    for guard, _w in enc.edges:
        edge_rel = bdd.OR(edge_rel, guard)
    rename_map = {enc.curr_indices[i]: enc.next_indices[i] for i in range(enc.num_bits)}
    target_next = bdd.rename(target, rename_map)
    good_succ = bdd.AND(edge_rel, target_next)
    has_good = bdd.exists_multi(enc.next_indices, good_succ)
    return bdd.AND(states, has_good)


def _extract_vertices(enc: SymbolicMPPEncoding, bdd_set: BDDNode) -> Set[int]:
    """Extract explicit vertex set from BDD."""
    bdd = enc.bdd
    result = set()
    for v, bits in enc.vertex_bits.items():
        check = bdd_set
        for i, idx in enumerate(enc.curr_indices):
            check = bdd.restrict(check, idx, bits[i])
        if check._id == bdd.TRUE._id:
            result.add(v)
    return result


def _make_vertex_set_bdd(enc: SymbolicMPPEncoding, vertices: Set[int]) -> BDDNode:
    """Encode a set of vertices as a BDD."""
    bdd = enc.bdd
    result = bdd.FALSE
    for v in vertices:
        if v in enc.vert_to_idx:
            result = bdd.OR(result, _make_vertex_bdd(
                bdd, enc.vert_to_idx[v], enc.curr_indices, enc.num_bits))
    return result


# ============================================================
# Symbolic Attractor
# ============================================================

def symbolic_attractor(enc: SymbolicMPPEncoding, target: BDDNode, player: Player,
                       subgame: BDDNode = None) -> BDDNode:
    """Compute attractor of 'target' for 'player' within the (sub)game.

    Attractor(target, player) = fixpoint of:
      target UNION {v in player_verts | some successor in attr}
             UNION {v in opponent_verts | all successors in attr (and has succ)}
    """
    bdd = enc.bdd
    if player == Player.EVEN:
        player_verts = enc.owner_even
        opponent_verts = enc.owner_odd
    else:
        player_verts = enc.owner_odd
        opponent_verts = enc.owner_even

    # Restrict to subgame if specified
    if subgame is not None:
        player_verts = bdd.AND(player_verts, subgame)
        opponent_verts = bdd.AND(opponent_verts, subgame)

    attr = bdd.AND(target, enc.vertices_bdd if subgame is None else subgame)
    while True:
        # Player vertices with at least one successor in attr
        player_pull = bdd.AND(player_verts, _symbolic_some_succ_in(enc, player_verts, attr))
        # Opponent vertices with ALL successors in attr (must have successors)
        opp_with_succ = _symbolic_has_successor(enc, opponent_verts)
        opp_pull = bdd.AND(opp_with_succ, _symbolic_all_succ_in(enc, opp_with_succ, attr))

        new_attr = bdd.OR(attr, bdd.OR(player_pull, opp_pull))
        if new_attr._id == attr._id:
            break
        attr = new_attr
    return attr


# ============================================================
# Symbolic Reachability
# ============================================================

def symbolic_reachability(game: MeanPayoffParityGame, start_vertices: Set[int]) -> Set[int]:
    """Compute all vertices reachable from start_vertices using BDD-based forward BFS."""
    enc = encode_mpp_game(game)
    bdd = enc.bdd

    start_bdd = _make_vertex_set_bdd(enc, start_vertices)
    reached = start_bdd
    while True:
        new_reached = bdd.OR(reached, _symbolic_successors(enc, reached))
        new_reached = bdd.AND(new_reached, enc.vertices_bdd)
        if new_reached._id == reached._id:
            break
        reached = new_reached

    return _extract_vertices(enc, reached)


# ============================================================
# Symbolic Safety Checking
# ============================================================

def symbolic_safety_check(game: MeanPayoffParityGame, safe_set: Set[int]) -> Dict:
    """Check if Even can keep play within safe_set forever.

    Uses symbolic attractor: Odd's attractor to unsafe set is removed.
    """
    enc = encode_mpp_game(game)
    bdd = enc.bdd

    safe_bdd = _make_vertex_set_bdd(enc, safe_set)
    unsafe_bdd = bdd.AND(enc.vertices_bdd, bdd.NOT(safe_bdd))

    odd_attr = symbolic_attractor(enc, unsafe_bdd, Player.ODD)
    even_safe = bdd.AND(enc.vertices_bdd, bdd.NOT(odd_attr))

    return {
        'safe_vertices': _extract_vertices(enc, even_safe),
        'unsafe_vertices': _extract_vertices(enc, odd_attr),
        'safe_count': len(_extract_vertices(enc, even_safe)),
        'unsafe_count': len(_extract_vertices(enc, odd_attr)),
    }


# ============================================================
# Symbolic Parity Solving (Zielonka on BDDs)
# ============================================================

def _symbolic_zielonka(enc: SymbolicMPPEncoding, subgame: BDDNode) -> Tuple[BDDNode, BDDNode]:
    """Symbolic Zielonka algorithm for parity games.

    Returns (win_even_bdd, win_odd_bdd) within the subgame.
    """
    bdd = enc.bdd

    # Check if subgame is empty
    sub_verts = _extract_vertices(enc, subgame)
    if not sub_verts:
        return (bdd.FALSE, bdd.FALSE)

    # Find max priority in subgame
    max_p = -1
    for p, p_bdd in enc.priority_groups.items():
        check = bdd.AND(subgame, p_bdd)
        if check._id != bdd.FALSE._id:
            max_p = max(max_p, p)

    if max_p < 0:
        return (subgame, bdd.FALSE)

    # Player who benefits from max priority
    if max_p % 2 == 0:
        player = Player.EVEN
    else:
        player = Player.ODD

    # Vertices with max priority
    target = bdd.AND(subgame, enc.priority_groups[max_p])

    # Compute attractor restricted to subgame edges
    # We need to restrict edges to subgame for attractor computation
    attr = _symbolic_attractor_subgame(enc, target, player, subgame)

    # Recurse on subgame minus attractor
    sub_minus = bdd.AND(subgame, bdd.NOT(attr))
    if sub_minus._id == bdd.FALSE._id:
        # Whole subgame is attractor -> player wins everything
        if player == Player.EVEN:
            return (subgame, bdd.FALSE)
        else:
            return (bdd.FALSE, subgame)

    # Recursive call
    rec_even, rec_odd = _symbolic_zielonka(enc, sub_minus)

    # Check which region the opponent wins
    if player == Player.EVEN:
        opponent = Player.ODD
        opp_wins = rec_odd
    else:
        opponent = Player.EVEN
        opp_wins = rec_even

    if opp_wins._id == bdd.FALSE._id:
        # Opponent wins nothing in subproblem -> player wins everything
        if player == Player.EVEN:
            return (subgame, bdd.FALSE)
        else:
            return (bdd.FALSE, subgame)

    # Opponent's attractor to their winning region
    opp_attr = _symbolic_attractor_subgame(enc, opp_wins, opponent, subgame)

    # Recurse on subgame minus opponent's attractor
    sub_minus2 = bdd.AND(subgame, bdd.NOT(opp_attr))
    if sub_minus2._id == bdd.FALSE._id:
        if opponent == Player.EVEN:
            return (subgame, bdd.FALSE)
        else:
            return (bdd.FALSE, subgame)

    rec2_even, rec2_odd = _symbolic_zielonka(enc, sub_minus2)

    # Combine: opponent wins their attractor + whatever they won in recursion
    if opponent == Player.EVEN:
        return (bdd.OR(opp_attr, rec2_even), rec2_odd)
    else:
        return (rec2_even, bdd.OR(opp_attr, rec2_odd))


def _symbolic_attractor_subgame(enc: SymbolicMPPEncoding, target: BDDNode,
                                player: Player, subgame: BDDNode) -> BDDNode:
    """Compute attractor restricted to subgame edges.

    Only considers edges where both source and target are in subgame.
    """
    bdd = enc.bdd
    if player == Player.EVEN:
        player_verts = bdd.AND(enc.owner_even, subgame)
        opponent_verts = bdd.AND(enc.owner_odd, subgame)
    else:
        player_verts = bdd.AND(enc.owner_odd, subgame)
        opponent_verts = bdd.AND(enc.owner_even, subgame)

    # Build edge relation restricted to subgame
    rename_map = {enc.curr_indices[i]: enc.next_indices[i] for i in range(enc.num_bits)}
    subgame_next = bdd.rename(subgame, rename_map)

    edges_sub = bdd.FALSE
    for guard, _w in enc.edges:
        restricted = bdd.AND(guard, bdd.AND(subgame, subgame_next))
        edges_sub = bdd.OR(edges_sub, restricted)

    attr = bdd.AND(target, subgame)
    while True:
        # Rename attr to next vars
        attr_next = bdd.rename(attr, rename_map)

        # Player vertices with some successor in attr
        good_edges = bdd.AND(edges_sub, attr_next)
        has_good = bdd.exists_multi(enc.next_indices, good_edges)
        player_pull = bdd.AND(player_verts, has_good)

        # Opponent vertices with ALL successors in attr
        # has_succ = exists next. edges_sub(curr, next)
        has_succ = bdd.exists_multi(enc.next_indices, edges_sub)
        opp_with_succ = bdd.AND(opponent_verts, has_succ)
        # bad = exists next. edges_sub(curr, next) AND NOT attr(next)
        bad_edges = bdd.AND(edges_sub, bdd.NOT(attr_next))
        has_bad = bdd.exists_multi(enc.next_indices, bad_edges)
        opp_all_in = bdd.AND(opp_with_succ, bdd.NOT(has_bad))

        # Dead ends: vertices in subgame with no successors in subgame
        dead_ends = bdd.AND(subgame, bdd.NOT(has_succ))
        # Dead Even -> Odd wins, Dead Odd -> Even wins
        if player == Player.EVEN:
            dead_pull = bdd.AND(dead_ends, enc.owner_even)  # dead Even verts attracted
        else:
            dead_pull = bdd.AND(dead_ends, enc.owner_odd)  # dead Odd verts attracted

        new_attr = bdd.OR(attr, bdd.OR(player_pull, bdd.OR(opp_all_in, dead_pull)))
        new_attr = bdd.AND(new_attr, subgame)
        if new_attr._id == attr._id:
            break
        attr = new_attr
    return attr


# ============================================================
# Symbolic Mean-Payoff Parity Solving
# ============================================================

@dataclass
class SymbolicMPPResult:
    """Result of symbolic mean-payoff parity game solving."""
    win_even: Set[int]
    win_odd: Set[int]
    strategy_even: Dict[int, int]
    strategy_odd: Dict[int, int]
    values: Optional[Dict[int, float]] = None
    threshold: float = 0.0
    iterations: int = 0
    encoding_stats: Dict = field(default_factory=dict)


def solve_symbolic_mpp(game: MeanPayoffParityGame, threshold: float = 0.0) -> SymbolicMPPResult:
    """Solve mean-payoff parity game using symbolic BDD operations.

    Algorithm:
    1. Encode game as BDDs
    2. Solve parity via symbolic Zielonka
    3. Check mean-payoff (via energy game reduction) under parity strategy
    4. Iteratively refine: remove MP failures + Odd attractor, re-solve parity

    Symbolic operations handle the parity/attractor part; energy check is explicit
    since energy values are numeric.
    """
    if not game.vertices:
        return SymbolicMPPResult(set(), set(), {}, {}, threshold=threshold)

    enc = encode_mpp_game(game)
    bdd = enc.bdd

    remaining = set(game.vertices)
    final_strategy_even = {}
    final_strategy_odd = {}
    iterations = 0

    for _ in range(len(game.vertices) + 1):
        if not remaining:
            break

        iterations += 1

        # Step 1: Solve parity symbolically on current subgame
        remaining_bdd = _make_vertex_set_bdd(enc, remaining)
        parity_even_bdd, parity_odd_bdd = _symbolic_zielonka(enc, remaining_bdd)
        parity_win_even = _extract_vertices(enc, parity_even_bdd) & remaining
        parity_win_odd = _extract_vertices(enc, parity_odd_bdd) & remaining

        if not parity_win_even:
            break

        # Extract parity strategy for Even (explicit, since we need edges)
        parity_strategy_even = _extract_even_strategy(game, parity_win_even)
        parity_strategy_odd = _extract_odd_strategy(game, parity_win_odd | (remaining - parity_win_even))

        # Step 2: Check mean-payoff under Even's parity strategy
        mp_ok = _check_mp_under_strategy(game, parity_win_even, parity_strategy_even, threshold)

        if mp_ok == parity_win_even:
            # Even's parity strategy also achieves mean-payoff
            final_strategy_even.update(parity_strategy_even)
            final_strategy_odd.update(parity_strategy_odd)
            win_even = parity_win_even
            win_odd = game.vertices - win_even
            return SymbolicMPPResult(
                win_even=win_even,
                win_odd=win_odd,
                strategy_even=final_strategy_even,
                strategy_odd=final_strategy_odd,
                threshold=threshold,
                iterations=iterations,
                encoding_stats=_make_encoding_stats(enc),
            )

        # Step 3: Remove MP failures + Odd attractor
        mp_lost = parity_win_even - mp_ok
        to_remove = parity_win_odd | mp_lost

        if not to_remove:
            win_even = mp_ok
            win_odd = game.vertices - win_even
            final_strategy_even.update(parity_strategy_even)
            final_strategy_odd.update(parity_strategy_odd)
            return SymbolicMPPResult(
                win_even=win_even,
                win_odd=win_odd,
                strategy_even=final_strategy_even,
                strategy_odd=final_strategy_odd,
                threshold=threshold,
                iterations=iterations,
                encoding_stats=_make_encoding_stats(enc),
            )

        # Symbolic Odd attractor to removed vertices
        to_remove_bdd = _make_vertex_set_bdd(enc, to_remove)
        odd_attr = _symbolic_attractor_subgame(enc, to_remove_bdd, Player.ODD, remaining_bdd)
        odd_attr_verts = _extract_vertices(enc, odd_attr)

        final_strategy_odd.update(parity_strategy_odd)
        remaining -= odd_attr_verts

    # Exhausted: Even can't win anywhere
    return SymbolicMPPResult(
        win_even=set(),
        win_odd=set(game.vertices),
        strategy_even=final_strategy_even,
        strategy_odd=final_strategy_odd,
        threshold=threshold,
        iterations=iterations,
        encoding_stats=_make_encoding_stats(enc),
    )


def _extract_even_strategy(game: MeanPayoffParityGame, win_even: Set[int]) -> Dict[int, int]:
    """Extract a strategy for Even: at each Even vertex in win_even, pick a successor in win_even."""
    strategy = {}
    for v in win_even:
        if game.owner[v] == Player.EVEN:
            succs = game.successors(v)
            # Prefer successors in win_even
            for (t, w) in succs:
                if t in win_even:
                    strategy[v] = t
                    break
            else:
                if succs:
                    strategy[v] = succs[0][0]
    return strategy


def _extract_odd_strategy(game: MeanPayoffParityGame, win_odd: Set[int]) -> Dict[int, int]:
    """Extract a strategy for Odd: at each Odd vertex in win_odd, pick a successor in win_odd."""
    strategy = {}
    for v in win_odd:
        if game.owner[v] == Player.ODD:
            succs = game.successors(v)
            for (t, w) in succs:
                if t in win_odd:
                    strategy[v] = t
                    break
            else:
                if succs:
                    strategy[v] = succs[0][0]
    return strategy


def _check_mp_under_strategy(game: MeanPayoffParityGame, verts: Set[int],
                              strategy_even: Dict[int, int], threshold: float) -> Set[int]:
    """Check mean-payoff under Even's fixed strategy via energy game reduction."""
    if not verts:
        return set()

    n = len(verts)
    if threshold == int(threshold):
        shift = int(threshold)
        scale = 1
    else:
        scale = n
        shift = int(round(threshold * n))

    eg = EnergyGame()
    for v in verts:
        eg.add_vertex(v, game.owner[v])
    for v in verts:
        if game.owner[v] == Player.EVEN and v in strategy_even:
            target = strategy_even[v]
            for (t, w) in game.edges.get(v, []):
                if t == target and t in verts:
                    eg.add_edge(v, t, w * scale - shift)
                    break
        else:
            for (t, w) in game.edges.get(v, []):
                if t in verts:
                    eg.add_edge(v, t, w * scale - shift)

    result = solve_energy(eg)
    return set(result.win_energy)


def _make_encoding_stats(enc: SymbolicMPPEncoding) -> Dict:
    """Build encoding statistics dict."""
    return {
        'num_vertices': enc.num_vertices,
        'num_bits': enc.num_bits,
        'num_edges': len(enc.edges),
        'num_weight_groups': len(enc.weight_groups),
        'num_priority_groups': len(enc.priority_groups),
        'bdd_vars': 2 * enc.num_bits,
    }


# ============================================================
# Symbolic Value Computation
# ============================================================

def compute_symbolic_mpp_values(game: MeanPayoffParityGame) -> SymbolicMPPResult:
    """Compute optimal mean-payoff values using symbolic parity + binary search.

    For each vertex, binary-search over thresholds to find the optimal value.
    """
    if not game.vertices:
        return SymbolicMPPResult(set(), set(), {}, {}, values={}, threshold=0.0)

    n = len(game.vertices)
    W = game.max_weight()

    if n == 0 or W == 0:
        result = solve_symbolic_mpp(game)
        result.values = {v: 0.0 for v in game.vertices}
        return result

    precision = 1.0 / (2 * n * n) if n > 0 else 0.5
    values = {}

    for v in game.vertices:
        lo, hi = -float(W), float(W)
        for _ in range(int(math.log2(2 * W / precision)) + 5):
            mid = (lo + hi) / 2.0
            result = solve_symbolic_mpp(game, mid)
            if v in result.win_even:
                lo = mid
            else:
                hi = mid
        values[v] = _snap_rational(lo, n, W)

    base_result = solve_symbolic_mpp(game)
    base_result.values = values
    return base_result


def _snap_rational(x: float, n: int, W: int) -> float:
    """Snap a float to the nearest rational p/q with |q| <= n and |p/q| <= W."""
    best = round(x)
    best_dist = abs(x - best)
    for q in range(1, n + 1):
        p = round(x * q)
        val = p / q
        if abs(val) <= W + 1:
            dist = abs(x - val)
            if dist < best_dist:
                best = val
                best_dist = dist
    return best


# ============================================================
# Symbolic Decomposition
# ============================================================

def symbolic_decompose_mpp(game: MeanPayoffParityGame, threshold: float = 0.0) -> Dict:
    """Compare winning regions under different objective combinations using symbolic solving.

    Returns parity-only, mean-payoff-only, and combined winning regions.
    """
    enc = encode_mpp_game(game)
    bdd = enc.bdd

    # Parity only (symbolic Zielonka)
    parity_even_bdd, parity_odd_bdd = _symbolic_zielonka(enc, enc.vertices_bdd)
    parity_win_even = _extract_vertices(enc, parity_even_bdd)
    parity_win_odd = _extract_vertices(enc, parity_odd_bdd)

    # Mean-payoff only (via energy game, no parity constraint)
    mp_win_even = _check_mp_free(game, set(game.vertices), threshold)
    mp_win_odd = game.vertices - mp_win_even

    # Combined (symbolic MPP)
    combined = solve_symbolic_mpp(game, threshold)

    # Analysis
    parity_only_extra = parity_win_even - combined.win_even
    mp_only_extra = mp_win_even - combined.win_even
    both_individual = parity_win_even & mp_win_even
    combined_loss = both_individual - combined.win_even

    return {
        'parity_only': {
            'win_even': parity_win_even,
            'win_odd': parity_win_odd,
        },
        'mean_payoff_only': {
            'win_even': mp_win_even,
            'win_odd': mp_win_odd,
        },
        'combined': {
            'win_even': combined.win_even,
            'win_odd': combined.win_odd,
        },
        'analysis': {
            'parity_wins_but_combined_loses': parity_only_extra,
            'mp_wins_but_combined_loses': mp_only_extra,
            'both_individual_win': both_individual,
            'lost_to_interaction': combined_loss,
        },
        'threshold': threshold,
    }


def _check_mp_free(game: MeanPayoffParityGame, verts: Set[int], threshold: float) -> Set[int]:
    """Check mean-payoff freely (Even chooses best edge). No parity constraint."""
    if not verts:
        return set()
    n = len(verts)
    if threshold == int(threshold):
        shift = int(threshold)
        scale = 1
    else:
        scale = n
        shift = int(round(threshold * n))
    eg = EnergyGame()
    for v in verts:
        eg.add_vertex(v, game.owner[v])
    for v in verts:
        for (t, w) in game.edges.get(v, []):
            if t in verts:
                eg.add_edge(v, t, w * scale - shift)
    result = solve_energy(eg)
    return set(result.win_energy)


# ============================================================
# Comparison API
# ============================================================

def compare_with_explicit(game: MeanPayoffParityGame, threshold: float = 0.0) -> Dict:
    """Compare symbolic vs explicit mean-payoff parity solving."""
    explicit_result = solve_mpp_threshold(game, threshold)
    symbolic_result = solve_symbolic_mpp(game, threshold)

    agree = (explicit_result.win_even == symbolic_result.win_even and
             explicit_result.win_odd == symbolic_result.win_odd)

    return {
        'explicit': {
            'win_even': explicit_result.win_even,
            'win_odd': explicit_result.win_odd,
        },
        'symbolic': {
            'win_even': symbolic_result.win_even,
            'win_odd': symbolic_result.win_odd,
            'iterations': symbolic_result.iterations,
            'encoding_stats': symbolic_result.encoding_stats,
        },
        'agree': agree,
    }


def compare_values(game: MeanPayoffParityGame) -> Dict:
    """Compare optimal values computed by symbolic vs explicit methods."""
    explicit_result = compute_mpp_values(game)
    symbolic_result = compute_symbolic_mpp_values(game)

    value_agree = True
    max_diff = 0.0
    if explicit_result.values and symbolic_result.values:
        for v in game.vertices:
            ev = explicit_result.values.get(v, 0.0)
            sv = symbolic_result.values.get(v, 0.0)
            diff = abs(ev - sv)
            max_diff = max(max_diff, diff)
            if diff > 0.5:
                value_agree = False

    return {
        'explicit_values': explicit_result.values,
        'symbolic_values': symbolic_result.values,
        'value_agree': value_agree,
        'max_diff': max_diff,
        'explicit_win_even': explicit_result.win_even,
        'symbolic_win_even': symbolic_result.win_even,
    }


def compare_decompositions(game: MeanPayoffParityGame, threshold: float = 0.0) -> Dict:
    """Compare symbolic vs explicit decomposition analysis."""
    explicit = decompose_mpp(game, threshold)
    symbolic = symbolic_decompose_mpp(game, threshold)

    parity_agree = (explicit['parity_only']['win_even'] == symbolic['parity_only']['win_even'])
    mp_agree = (explicit['mean_payoff_only']['win_even'] == symbolic['mean_payoff_only']['win_even'])
    combined_agree = (explicit['combined']['win_even'] == symbolic['combined']['win_even'])

    return {
        'explicit': explicit,
        'symbolic': symbolic,
        'parity_agree': parity_agree,
        'mp_agree': mp_agree,
        'combined_agree': combined_agree,
        'all_agree': parity_agree and mp_agree and combined_agree,
    }


# ============================================================
# Construction Helpers
# ============================================================

def make_symbolic_chain_mpp(n: int, weights: Optional[List[int]] = None,
                             priorities: Optional[List[int]] = None) -> MeanPayoffParityGame:
    """Chain game: 0->1->...->n-1->0 with specified weights and priorities."""
    return make_chain_mpp(n, weights, priorities)


def make_symbolic_choice_mpp(good_weight: int = 1, bad_weight: int = -1,
                              good_prio: int = 0, bad_prio: int = 1) -> MeanPayoffParityGame:
    """Even chooses between a good cycle (even prio, pos weight) and bad cycle."""
    return make_choice_mpp(good_weight, bad_weight, good_prio, bad_prio)


def make_symbolic_diamond_mpp(top_weight: int = 0, left_weight: int = 2,
                                right_weight: int = -1) -> MeanPayoffParityGame:
    """Diamond game: Even top chooses left(+) or right(-), both to Odd bottom, back to top."""
    g = MeanPayoffParityGame()
    g.add_vertex(0, Player.EVEN, 0)   # top
    g.add_vertex(1, Player.ODD, 0)    # left
    g.add_vertex(2, Player.ODD, 0)    # right
    g.add_vertex(3, Player.EVEN, 0)   # bottom
    g.add_edge(0, 1, left_weight)
    g.add_edge(0, 2, right_weight)
    g.add_edge(1, 3, 0)
    g.add_edge(2, 3, 0)
    g.add_edge(3, 0, top_weight)
    return g


def make_symbolic_grid_mpp(rows: int, cols: int) -> MeanPayoffParityGame:
    """Grid game: rows x cols, alternating owners, right/down +1, left/up -1."""
    g = MeanPayoffParityGame()
    def vid(r, c):
        return r * cols + c
    for r in range(rows):
        for c in range(cols):
            v = vid(r, c)
            g.add_vertex(v, Player.EVEN if (r + c) % 2 == 0 else Player.ODD,
                          (r + c) % 4)  # priorities 0-3
            if c + 1 < cols:
                g.add_edge(v, vid(r, c + 1), 1)
                g.add_edge(vid(r, c + 1), v, -1)
            if r + 1 < rows:
                g.add_edge(v, vid(r + 1, c), 1)
                g.add_edge(vid(r + 1, c), v, -1)
    return g


# ============================================================
# Statistics
# ============================================================

def symbolic_mpp_statistics(game: MeanPayoffParityGame) -> Dict:
    """Comprehensive statistics using symbolic encoding and solving."""
    enc = encode_mpp_game(game)
    result = solve_symbolic_mpp(game)

    bdd = enc.bdd
    total_bdd_nodes = sum(bdd.node_count(g) for g, _w in enc.edges)

    return {
        'num_vertices': enc.num_vertices,
        'num_edges': len(enc.edges),
        'num_bits': enc.num_bits,
        'total_bdd_vars': 2 * enc.num_bits,
        'weight_groups': len(enc.weight_groups),
        'priority_groups': len(enc.priority_groups),
        'total_edge_bdd_nodes': total_bdd_nodes,
        'vertices_bdd_nodes': bdd.node_count(enc.vertices_bdd),
        'win_even': len(result.win_even),
        'win_odd': len(result.win_odd),
        'iterations': result.iterations,
        'max_priority': max(game.priority.values()) if game.priority else 0,
        'max_weight': game.max_weight(),
    }
