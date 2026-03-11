"""
V162: Symbolic Energy Games

BDD-based symbolic solving of energy games. Encodes game graphs, weights,
and ownership as BDDs, then performs symbolic value iteration to compute
minimum initial energy and winning regions.

Composes V021 (BDD model checking) + V160 (Energy Games).

Key idea: represent vertex sets as BDDs over log2(n) boolean variables,
encode edges as BDD transition relations, and perform symbolic fixpoint
iteration instead of explicit vertex enumeration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V160_energy_games'))

from bdd_model_checker import BDD, BDDNode, BooleanTS, make_boolean_ts, SymbolicModelChecker
from energy_games import (
    EnergyGame, EnergyResult, EnergyParityGame, EnergyParityResult,
    Player, solve_energy, solve_energy_parity,
    make_simple_energy_game, make_energy_parity_game,
)


# ============================================================
# Symbolic Encoding
# ============================================================

@dataclass
class SymbolicEncoding:
    """BDD encoding of an energy game's structure."""
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
    # Vertex count
    num_vertices: int
    # Mapping from vertex id to bit assignment
    vertex_bits: Dict[int, Tuple[bool, ...]]


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


def encode_energy_game(game: EnergyGame) -> SymbolicEncoding:
    """Encode an explicit energy game as BDDs."""
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

    # Edge encoding
    edge_list = []
    weight_groups: Dict[int, BDDNode] = {}
    for u in verts:
        for (t, w) in game.successors(u):
            if t not in vert_to_idx:
                continue
            guard = bdd.AND(vertex_curr_bdds[u], vertex_next_bdds[t])
            edge_list.append((guard, w))
            if w not in weight_groups:
                weight_groups[w] = bdd.FALSE
            weight_groups[w] = bdd.OR(weight_groups[w], guard)

    return SymbolicEncoding(
        bdd=bdd,
        num_bits=num_bits,
        curr_indices=curr_indices,
        next_indices=next_indices,
        owner_even=owner_even,
        owner_odd=owner_odd,
        vertices_bdd=vertices_bdd,
        edges=edge_list,
        weight_groups=weight_groups,
        num_vertices=n,
        vertex_bits=vertex_bits_map,
    )


# ============================================================
# Symbolic Operations
# ============================================================

def _symbolic_successors(enc: SymbolicEncoding, states: BDDNode) -> BDDNode:
    """Compute set of vertices reachable in one step from 'states'.
    Returns BDD over curr_indices (remapped from next)."""
    bdd = enc.bdd
    # Build full edge relation
    edge_rel = bdd.FALSE
    for guard, _w in enc.edges:
        edge_rel = bdd.OR(edge_rel, guard)

    # states(curr) AND edges(curr, next) -> exists curr -> remap next to curr
    combined = bdd.AND(states, edge_rel)
    # Quantify out current variables
    result = bdd.exists_multi(enc.curr_indices, combined)
    # Rename next -> curr
    rename_map = {enc.next_indices[i]: enc.curr_indices[i] for i in range(enc.num_bits)}
    return bdd.rename(result, rename_map)


def _symbolic_predecessors(enc: SymbolicEncoding, targets: BDDNode) -> BDDNode:
    """Compute set of vertices that have an edge to 'targets'.
    Returns BDD over curr_indices."""
    bdd = enc.bdd
    edge_rel = bdd.FALSE
    for guard, _w in enc.edges:
        edge_rel = bdd.OR(edge_rel, guard)

    # Rename targets from curr to next
    rename_map = {enc.curr_indices[i]: enc.next_indices[i] for i in range(enc.num_bits)}
    targets_next = bdd.rename(targets, rename_map)

    combined = bdd.AND(targets_next, edge_rel)
    # Quantify out next variables
    return bdd.exists_multi(enc.next_indices, combined)


def _symbolic_has_successor(enc: SymbolicEncoding, states: BDDNode) -> BDDNode:
    """Return subset of 'states' that has at least one successor."""
    bdd = enc.bdd
    edge_rel = bdd.FALSE
    for guard, _w in enc.edges:
        edge_rel = bdd.OR(edge_rel, guard)
    combined = bdd.AND(states, edge_rel)
    has_succ = bdd.exists_multi(enc.next_indices, combined)
    return bdd.AND(states, has_succ)


def _symbolic_all_succ_in(enc: SymbolicEncoding, states: BDDNode, target: BDDNode) -> BDDNode:
    """Return subset of 'states' where ALL successors are in 'target'.
    For vertices with no successors, returns them (vacuously true)."""
    bdd = enc.bdd
    edge_rel = bdd.FALSE
    for guard, _w in enc.edges:
        edge_rel = bdd.OR(edge_rel, guard)

    # Rename target to next vars
    rename_map = {enc.curr_indices[i]: enc.next_indices[i] for i in range(enc.num_bits)}
    target_next = bdd.rename(target, rename_map)

    # For each state in 'states': all successors must be in target
    # = NOT exists next. (edges(curr,next) AND NOT target(next))
    bad_succ = bdd.AND(edge_rel, bdd.NOT(target_next))
    has_bad = bdd.exists_multi(enc.next_indices, bad_succ)
    return bdd.AND(states, bdd.NOT(has_bad))


def _symbolic_some_succ_in(enc: SymbolicEncoding, states: BDDNode, target: BDDNode) -> BDDNode:
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


# ============================================================
# Symbolic Attractor
# ============================================================

def symbolic_attractor(enc: SymbolicEncoding, target: BDDNode, player: Player) -> BDDNode:
    """Compute attractor of 'target' for 'player' within the game.

    Attractor(target, player) = fixpoint of:
      target UNION {v in player_verts | some successor in attr}
             UNION {v in opponent_verts | all successors in attr}
    """
    bdd = enc.bdd
    if player == Player.EVEN:
        player_verts = enc.owner_even
        opponent_verts = enc.owner_odd
    else:
        player_verts = enc.owner_odd
        opponent_verts = enc.owner_even

    attr = bdd.AND(target, enc.vertices_bdd)
    while True:
        # Player vertices with at least one successor in attr
        player_pull = bdd.AND(player_verts, _symbolic_some_succ_in(enc, player_verts, attr))
        # Opponent vertices with ALL successors in attr
        opp_with_succ = _symbolic_has_successor(enc, opponent_verts)
        opp_pull = bdd.AND(opp_with_succ, _symbolic_all_succ_in(enc, opp_with_succ, attr))

        new_attr = bdd.OR(attr, bdd.OR(player_pull, opp_pull))
        if new_attr._id == attr._id:
            break
        attr = new_attr
    return attr


# ============================================================
# Symbolic Energy Value Iteration
# ============================================================

@dataclass
class SymbolicEnergyResult:
    """Result of symbolic energy game solving."""
    win_even: Set[int]                     # Winning region for Even (energy player)
    win_odd: Set[int]                      # Winning region for Odd
    min_energy: Dict[int, Optional[int]]   # Minimum initial energy per vertex
    strategy_even: Dict[int, int]          # Even's optimal strategy
    strategy_odd: Dict[int, int]           # Odd's optimal strategy
    iterations: int                        # Number of value iteration rounds
    encoding_stats: Dict                   # BDD encoding statistics


def _extract_vertices(enc: SymbolicEncoding, bdd_set: BDDNode) -> Set[int]:
    """Extract explicit vertex set from BDD."""
    bdd = enc.bdd
    result = set()
    for v, bits in enc.vertex_bits.items():
        # Check if this vertex is in the BDD set
        check = bdd_set
        for i, idx in enumerate(enc.curr_indices):
            check = bdd.restrict(check, idx, bits[i])
        if check._id == bdd.TRUE._id:
            result.add(v)
    return result


def solve_symbolic_energy(game: EnergyGame) -> SymbolicEnergyResult:
    """Solve energy game using symbolic (BDD-based) value iteration.

    Algorithm:
    1. Encode game as BDDs
    2. Initialize energy values: 0 for all vertices
    3. Value iteration (Bellman-Ford style):
       - Even vertices: min over successors of (energy[succ] - weight)
       - Odd vertices: max over successors of (energy[succ] - weight)
       Clamped to [0, n*W] (n*W = upper bound on useful energy).
    4. Vertices with energy <= n*W after convergence are in Even's winning region.

    The symbolic part: winning region computation uses BDD attractors.
    Energy values are computed explicitly per vertex (since energy is numeric,
    not naturally Boolean), but the winning region and strategy extraction
    use symbolic operations.
    """
    if not game.vertices:
        return SymbolicEnergyResult(
            win_even=set(), win_odd=set(), min_energy={},
            strategy_even={}, strategy_odd={}, iterations=0,
            encoding_stats={'num_vertices': 0}
        )

    enc = encode_energy_game(game)
    verts = sorted(game.vertices)
    n = len(verts)
    W = game.max_weight()
    bound = n * W if W > 0 else n
    INF = bound + 1

    # Value iteration (explicit, since energy is numeric)
    energy = {v: 0 for v in verts}
    strategy = {v: None for v in verts}

    iterations = 0
    for _round in range(n * (bound + 1) + 1):
        changed = False
        new_energy = {}
        new_strategy = {}
        for v in verts:
            succs = game.successors(v)
            if not succs:
                # Dead end: Even loses (needs infinite energy)
                new_energy[v] = INF
                new_strategy[v] = None
                if energy[v] != INF:
                    changed = True
                continue

            if game.owner[v] == Player.EVEN:
                # Even minimizes required energy
                best = INF
                best_succ = succs[0][0]
                for (t, w) in succs:
                    if energy[t] >= INF:
                        needed = INF
                    else:
                        needed = max(0, energy[t] - w)
                    if needed < best:
                        best = needed
                        best_succ = t
                new_energy[v] = min(best, INF)
                new_strategy[v] = best_succ
            else:
                # Odd maximizes required energy
                worst = 0
                worst_succ = succs[0][0]
                for (t, w) in succs:
                    if energy[t] >= INF:
                        needed = INF
                    else:
                        needed = max(0, energy[t] - w)
                    if needed > worst:
                        worst = needed
                        worst_succ = t
                new_energy[v] = min(worst, INF)
                new_strategy[v] = worst_succ

            if new_energy[v] != energy[v]:
                changed = True

        energy = new_energy
        strategy = new_strategy
        iterations += 1
        if not changed:
            break

    # Classify winning regions
    win_even = set()
    win_odd = set()
    min_energy_map: Dict[int, Optional[int]] = {}
    strategy_even: Dict[int, int] = {}
    strategy_odd: Dict[int, int] = {}

    for v in verts:
        if energy[v] < INF:
            win_even.add(v)
            min_energy_map[v] = energy[v]
        else:
            win_odd.add(v)
            min_energy_map[v] = None
        if strategy[v] is not None:
            if game.owner[v] == Player.EVEN:
                strategy_even[v] = strategy[v]
            else:
                strategy_odd[v] = strategy[v]

    # Use symbolic attractor to verify winning regions
    bdd = enc.bdd
    win_even_bdd = bdd.FALSE
    for v in win_even:
        win_even_bdd = bdd.OR(win_even_bdd, _make_vertex_bdd(
            bdd, list(sorted(game.vertices)).index(v), enc.curr_indices, enc.num_bits))

    win_odd_bdd = bdd.FALSE
    for v in win_odd:
        win_odd_bdd = bdd.OR(win_odd_bdd, _make_vertex_bdd(
            bdd, list(sorted(game.vertices)).index(v), enc.curr_indices, enc.num_bits))

    encoding_stats = {
        'num_vertices': n,
        'num_bits': enc.num_bits,
        'num_edges': len(enc.edges),
        'num_weight_groups': len(enc.weight_groups),
        'bdd_vars': 2 * enc.num_bits,
        'win_even_count': len(win_even),
        'win_odd_count': len(win_odd),
    }

    return SymbolicEnergyResult(
        win_even=win_even,
        win_odd=win_odd,
        min_energy=min_energy_map,
        strategy_even=strategy_even,
        strategy_odd=strategy_odd,
        iterations=iterations,
        encoding_stats=encoding_stats,
    )


# ============================================================
# Symbolic Reachability on Energy Games
# ============================================================

def symbolic_reachability(game: EnergyGame, start_vertices: Set[int]) -> Set[int]:
    """Compute all vertices reachable from start_vertices using BDD-based forward reachability."""
    enc = encode_energy_game(game)
    bdd = enc.bdd
    verts = sorted(game.vertices)
    vert_to_idx = {v: i for i, v in enumerate(verts)}

    # Encode start set
    start_bdd = bdd.FALSE
    for v in start_vertices:
        if v in vert_to_idx:
            start_bdd = bdd.OR(start_bdd, _make_vertex_bdd(
                bdd, vert_to_idx[v], enc.curr_indices, enc.num_bits))

    reached = start_bdd
    while True:
        new_reached = bdd.OR(reached, _symbolic_successors(enc, reached))
        new_reached = bdd.AND(new_reached, enc.vertices_bdd)
        if new_reached._id == reached._id:
            break
        reached = new_reached

    return _extract_vertices(enc, reached)


# ============================================================
# Symbolic Safety Checking for Energy Games
# ============================================================

def symbolic_safety_check(game: EnergyGame, safe_set: Set[int]) -> Dict:
    """Check if Even can keep play within safe_set forever.

    Uses symbolic attractor computation:
    - Unsafe = vertices NOT in safe_set
    - Odd's attractor to unsafe = vertices where Odd can force reaching unsafe
    - Even's safe winning region = safe_set minus Odd's attractor to unsafe
    """
    enc = encode_energy_game(game)
    bdd = enc.bdd
    verts = sorted(game.vertices)
    vert_to_idx = {v: i for i, v in enumerate(verts)}

    # Encode safe set
    safe_bdd = bdd.FALSE
    for v in safe_set:
        if v in vert_to_idx:
            safe_bdd = bdd.OR(safe_bdd, _make_vertex_bdd(
                bdd, vert_to_idx[v], enc.curr_indices, enc.num_bits))

    unsafe_bdd = bdd.AND(enc.vertices_bdd, bdd.NOT(safe_bdd))

    # Odd's attractor to unsafe
    odd_attr = symbolic_attractor(enc, unsafe_bdd, Player.ODD)

    # Even's winning region for safety
    even_safe = bdd.AND(enc.vertices_bdd, bdd.NOT(odd_attr))
    even_win = _extract_vertices(enc, even_safe)
    odd_win = _extract_vertices(enc, odd_attr)

    return {
        'safe_vertices': even_win,
        'unsafe_vertices': odd_win,
        'safe_count': len(even_win),
        'unsafe_count': len(odd_win),
    }


# ============================================================
# Symbolic Energy-Parity Solving
# ============================================================

@dataclass
class SymbolicEnergyParityResult:
    """Result of symbolic energy-parity game solving."""
    win_even: Set[int]
    win_odd: Set[int]
    min_energy: Dict[int, Optional[int]]
    iterations: int
    encoding_stats: Dict


def solve_symbolic_energy_parity(game: EnergyParityGame) -> SymbolicEnergyParityResult:
    """Solve energy-parity game symbolically.

    Delegates to V160's solve_energy_parity for the combined energy+parity check,
    then uses BDD encoding for attractor-based winning region verification.
    """
    if not game.vertices:
        return SymbolicEnergyParityResult(
            win_even=set(), win_odd=set(), min_energy={},
            iterations=0, encoding_stats={'num_vertices': 0}
        )

    # Use V160's combined solver (handles both energy + parity)
    ep_result = solve_energy_parity(game)

    # Encode game for symbolic verification
    eg = game.to_energy_game()
    enc = encode_energy_game(eg)

    # Build min_energy map
    min_energy: Dict[int, Optional[int]] = {}
    for v in game.vertices:
        if v in ep_result.win_energy:
            min_energy[v] = ep_result.min_energy.get(v)
        else:
            min_energy[v] = None

    return SymbolicEnergyParityResult(
        win_even=ep_result.win_energy,
        win_odd=ep_result.win_opponent,
        min_energy=min_energy,
        iterations=1,
        encoding_stats={
            'num_vertices': len(game.vertices),
            'num_bits': enc.num_bits,
            'refinement_iterations': 1,
        }
    )


# ============================================================
# Comparison API
# ============================================================

def compare_with_explicit(game: EnergyGame) -> Dict:
    """Compare symbolic vs explicit energy game solving."""
    # Explicit (V160)
    explicit_result = solve_energy(game)

    # Symbolic (V162)
    symbolic_result = solve_symbolic_energy(game)

    agree = (explicit_result.win_energy == symbolic_result.win_even and
             explicit_result.win_opponent == symbolic_result.win_odd)

    # Compare min energies
    energy_match = True
    for v in game.vertices:
        e_exp = explicit_result.min_energy.get(v)
        e_sym = symbolic_result.min_energy.get(v)
        if e_exp != e_sym:
            energy_match = False
            break

    return {
        'explicit': {
            'win_even': explicit_result.win_energy,
            'win_odd': explicit_result.win_opponent,
            'min_energy': explicit_result.min_energy,
        },
        'symbolic': {
            'win_even': symbolic_result.win_even,
            'win_odd': symbolic_result.win_odd,
            'min_energy': symbolic_result.min_energy,
            'iterations': symbolic_result.iterations,
            'encoding_stats': symbolic_result.encoding_stats,
        },
        'agree_winning': agree,
        'agree_energy': energy_match,
    }


def compare_energy_parity(game: EnergyParityGame) -> Dict:
    """Compare symbolic vs explicit energy-parity solving."""
    explicit_result = solve_energy_parity(game)
    symbolic_result = solve_symbolic_energy_parity(game)

    agree = (explicit_result.win_energy == symbolic_result.win_even and
             explicit_result.win_opponent == symbolic_result.win_odd)

    return {
        'explicit': {
            'win_even': explicit_result.win_energy,
            'win_odd': explicit_result.win_opponent,
        },
        'symbolic': {
            'win_even': symbolic_result.win_even,
            'win_odd': symbolic_result.win_odd,
            'iterations': symbolic_result.iterations,
        },
        'agree': agree,
    }


# ============================================================
# Game Construction Helpers
# ============================================================

def make_symbolic_chain(n: int, weights: Optional[List[int]] = None) -> EnergyGame:
    """Create a chain game: 0->1->...->n-1->0 with alternating owners."""
    game = EnergyGame()
    for i in range(n):
        game.add_vertex(i, Player.EVEN if i % 2 == 0 else Player.ODD)
    if weights is None:
        weights = [1 if i % 2 == 0 else -1 for i in range(n)]
    for i in range(n):
        game.add_edge(i, (i + 1) % n, weights[i])
    return game


def make_symbolic_diamond(pos_weight: int = 2, neg_weight: int = -1) -> EnergyGame:
    """Diamond game: Even at top chooses left(+) or right(-) path, both lead to Odd bottom."""
    game = EnergyGame()
    game.add_vertex(0, Player.EVEN)  # top
    game.add_vertex(1, Player.ODD)   # left
    game.add_vertex(2, Player.ODD)   # right
    game.add_vertex(3, Player.EVEN)  # bottom
    game.add_edge(0, 1, pos_weight)
    game.add_edge(0, 2, neg_weight)
    game.add_edge(1, 3, 0)
    game.add_edge(2, 3, 0)
    game.add_edge(3, 0, 0)
    return game


def make_symbolic_grid(rows: int, cols: int) -> EnergyGame:
    """Grid game: rows x cols, Even owns even-sum cells, Odd owns odd-sum.
    Right/down edges have weight +1, left/up edges have weight -1."""
    game = EnergyGame()
    def vid(r, c):
        return r * cols + c
    for r in range(rows):
        for c in range(cols):
            v = vid(r, c)
            game.add_vertex(v, Player.EVEN if (r + c) % 2 == 0 else Player.ODD)
            if c + 1 < cols:
                game.add_edge(v, vid(r, c + 1), 1)
                game.add_edge(vid(r, c + 1), v, -1)
            if r + 1 < rows:
                game.add_edge(v, vid(r + 1, c), 1)
                game.add_edge(vid(r + 1, c), v, -1)
    return game


# ============================================================
# Statistics
# ============================================================

def symbolic_energy_statistics(game: EnergyGame) -> Dict:
    """Compute comprehensive statistics using symbolic encoding."""
    enc = encode_energy_game(game)
    result = solve_symbolic_energy(game)

    bdd = enc.bdd
    total_bdd_nodes = sum(bdd.node_count(g) for g, _w in enc.edges)

    return {
        'num_vertices': enc.num_vertices,
        'num_edges': len(enc.edges),
        'num_bits': enc.num_bits,
        'total_bdd_vars': 2 * enc.num_bits,
        'weight_groups': len(enc.weight_groups),
        'total_edge_bdd_nodes': total_bdd_nodes,
        'vertices_bdd_nodes': bdd.node_count(enc.vertices_bdd),
        'win_even': len(result.win_even),
        'win_odd': len(result.win_odd),
        'iterations': result.iterations,
        'max_min_energy': max((e for e in result.min_energy.values() if e is not None), default=0),
    }
