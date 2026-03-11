"""V169: Symbolic Stochastic Parity Games -- BDD-based stochastic parity solving.

Composes V159 (Symbolic Parity Games) + V165 (Stochastic Parity Games).

Key ideas:
- BDD encoding for vertices, edges, ownership, priorities (from V159)
- Three vertex types: EVEN, ODD, RANDOM (from V165)
- Random vertices have explicit probability distributions (reals can't be BDD-encoded)
- Symbolic attractor adapted for stochastic semantics:
  * Almost-sure: RANDOM needs ALL positive-prob successors in target
  * Positive-prob: RANDOM needs ANY positive-prob successor in target
- Symbolic Zielonka with iterative RANDOM-closure refinement for almost-sure
- Positive-probability: treat RANDOM as EVEN (any good successor suffices)

Winning semantics:
- Almost-sure (AS): Even wins iff parity condition holds with probability 1
- Positive-probability (PP): Even wins iff parity condition holds with probability > 0
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V159_symbolic_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V165_stochastic_parity_games'))

from bdd_model_checker import BDD, BDDNode
from parity_games import ParityGame, Player
from symbolic_parity_games import (
    SymbolicParityGame, SymbolicSolution,
    _state_bdd, _rename_to_next, _rename_to_curr,
    _preimage, _image, _forall_successors_in,
    _extract_states, extract_winning_sets, extract_strategy,
    _compute_symbolic_strategy,
)
from stochastic_parity import (
    StochasticParityGame, VertexType, StochasticParityResult,
    solve_stochastic_parity,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SymbolicStochasticParityGame:
    """BDD-encoded stochastic parity game."""
    bdd: BDD
    n_bits: int
    state_vars: List[str]
    next_vars: List[str]
    var_indices: Dict[str, int]
    next_indices: Dict[str, int]
    vertices: BDDNode
    edges: BDDNode
    owner_even: BDDNode
    owner_random: BDDNode
    priority_bdds: Dict[int, BDDNode]
    max_priority: int
    probabilities: Dict[int, Dict[int, float]] = field(default_factory=dict)


@dataclass
class SymbolicStochasticResult:
    """Result of symbolic stochastic parity game solving."""
    win_even_as: Set[int]
    win_odd_as: Set[int]
    win_even_pp: Set[int]
    win_odd_pp: Set[int]
    strategy_even_as: Dict[int, int]
    strategy_odd_as: Dict[int, int]
    strategy_even_pp: Dict[int, int]
    strategy_odd_pp: Dict[int, int]


# ---------------------------------------------------------------------------
# Conversion: StochasticParityGame -> SymbolicStochasticParityGame
# ---------------------------------------------------------------------------

def stochastic_to_symbolic(game: StochasticParityGame) -> SymbolicStochasticParityGame:
    """Convert an explicit StochasticParityGame to symbolic BDD encoding."""
    if not game.vertices:
        n_bits = 1
        bdd = BDD(num_vars=2 * n_bits)
        sv = ["s0"]
        nv = ["s0'"]
        vi = {}
        ni = {}
        for name in sv:
            bdd.named_var(name)
            vi[name] = bdd.var_index(name)
        for name in nv:
            bdd.named_var(name)
            ni[name] = bdd.var_index(name)
        return SymbolicStochasticParityGame(
            bdd=bdd, n_bits=n_bits, state_vars=sv, next_vars=nv,
            var_indices=vi, next_indices=ni,
            vertices=bdd.FALSE, edges=bdd.FALSE,
            owner_even=bdd.FALSE, owner_random=bdd.FALSE,
            priority_bdds={}, max_priority=0, probabilities={}
        )

    max_v = max(game.vertices)
    n_bits = max(1, math.ceil(math.log2(max_v + 1))) if max_v > 0 else 1
    if (1 << n_bits) < max_v + 1:
        n_bits += 1

    bdd = BDD(num_vars=2 * n_bits)
    state_vars = [f"s{i}" for i in range(n_bits)]
    next_vars = [f"s{i}'" for i in range(n_bits)]
    var_indices = {}
    next_indices = {}

    for i in range(n_bits):
        bdd.named_var(state_vars[i])
        var_indices[state_vars[i]] = bdd.var_index(state_vars[i])

    for i in range(n_bits):
        bdd.named_var(next_vars[i])
        next_indices[next_vars[i]] = bdd.var_index(next_vars[i])

    # Encode vertex set
    vertices_bdd = bdd.FALSE
    for v in game.vertices:
        vertices_bdd = bdd.OR(vertices_bdd, _state_bdd(bdd, v, n_bits, var_indices, state_vars))

    # Encode owner_even
    owner_even_bdd = bdd.FALSE
    for v in game.vertices:
        if game.vertex_type.get(v) == VertexType.EVEN:
            owner_even_bdd = bdd.OR(owner_even_bdd, _state_bdd(bdd, v, n_bits, var_indices, state_vars))

    # Encode owner_random
    owner_random_bdd = bdd.FALSE
    for v in game.vertices:
        if game.vertex_type.get(v) == VertexType.RANDOM:
            owner_random_bdd = bdd.OR(owner_random_bdd, _state_bdd(bdd, v, n_bits, var_indices, state_vars))

    # Encode edges
    edges_bdd = bdd.FALSE
    for v in game.vertices:
        v_bdd = _state_bdd(bdd, v, n_bits, var_indices, state_vars)
        for w in game.edges.get(v, set()):
            w_bdd = _state_bdd(bdd, w, n_bits, next_indices, next_vars)
            edges_bdd = bdd.OR(edges_bdd, bdd.AND(v_bdd, w_bdd))

    # Encode priorities
    priority_bdds = {}
    max_priority = 0
    for v in game.vertices:
        p = game.priority.get(v, 0)
        max_priority = max(max_priority, p)
        if p not in priority_bdds:
            priority_bdds[p] = bdd.FALSE
        priority_bdds[p] = bdd.OR(priority_bdds[p], _state_bdd(bdd, v, n_bits, var_indices, state_vars))

    return SymbolicStochasticParityGame(
        bdd=bdd, n_bits=n_bits,
        state_vars=state_vars, next_vars=next_vars,
        var_indices=var_indices, next_indices=next_indices,
        vertices=vertices_bdd, edges=edges_bdd,
        owner_even=owner_even_bdd, owner_random=owner_random_bdd,
        priority_bdds=priority_bdds, max_priority=max_priority,
        probabilities=dict(game.probabilities),
    )


def symbolic_to_stochastic(sspg: SymbolicStochasticParityGame) -> StochasticParityGame:
    """Convert a symbolic stochastic parity game back to explicit form."""
    game = StochasticParityGame(
        vertices=set(), edges={}, vertex_type={}, priority={}, probabilities={}
    )
    bdd = sspg.bdd

    all_verts = _extract_states_sspg(sspg, sspg.vertices)
    even_verts = _extract_states_sspg(sspg, sspg.owner_even)
    random_verts = _extract_states_sspg(sspg, sspg.owner_random)

    for v in all_verts:
        if v in even_verts:
            vt = VertexType.EVEN
        elif v in random_verts:
            vt = VertexType.RANDOM
        else:
            vt = VertexType.ODD

        # Find priority
        prio = 0
        v_bdd = _state_bdd(bdd, v, sspg.n_bits, sspg.var_indices, sspg.state_vars)
        for p, p_bdd in sspg.priority_bdds.items():
            test = bdd.AND(v_bdd, p_bdd)
            if test._id != bdd.FALSE._id:
                prio = p
                break

        game.add_vertex(v, vt, prio)

    # Extract edges
    spg = _make_spg_adapter(sspg)
    for v in all_verts:
        v_bdd = _state_bdd(bdd, v, sspg.n_bits, sspg.var_indices, sspg.state_vars)
        v_edges = bdd.AND(v_bdd, sspg.edges)
        curr_idxs = list(sspg.var_indices.values())
        projected = v_edges
        for idx in curr_idxs:
            projected = bdd.exists(idx, projected)
        succs_bdd = _rename_to_curr(spg, projected)
        succs = _extract_states_sspg(sspg, succs_bdd)
        for w in succs:
            prob = sspg.probabilities.get(v, {}).get(w, 1.0)
            game.add_edge(v, w, prob)

    # Restore probabilities
    for v, dist in sspg.probabilities.items():
        if v in game.vertices:
            game.probabilities[v] = dict(dist)

    return game


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_spg_adapter(sspg: SymbolicStochasticParityGame) -> SymbolicParityGame:
    """Create a SymbolicParityGame adapter for reusing V159 helpers."""
    return SymbolicParityGame(
        bdd=sspg.bdd, n_bits=sspg.n_bits,
        state_vars=sspg.state_vars, next_vars=sspg.next_vars,
        var_indices=sspg.var_indices, next_indices=sspg.next_indices,
        vertices=sspg.vertices, edges=sspg.edges,
        owner_even=sspg.owner_even,
        priority_bdds=sspg.priority_bdds, max_priority=sspg.max_priority,
    )


def _extract_states_sspg(sspg: SymbolicStochasticParityGame, bdd_node: BDDNode) -> Set[int]:
    """Extract concrete state IDs from a BDD."""
    spg = _make_spg_adapter(sspg)
    return _extract_states(spg, bdd_node)


def _owner_odd(sspg: SymbolicStochasticParityGame) -> BDDNode:
    """BDD of Odd-owned vertices (not Even, not Random)."""
    bdd = sspg.bdd
    not_even = bdd.NOT(sspg.owner_even)
    not_random = bdd.NOT(sspg.owner_random)
    odd = bdd.AND(not_even, not_random)
    return bdd.AND(odd, sspg.vertices)


# ---------------------------------------------------------------------------
# Symbolic stochastic attractor
# ---------------------------------------------------------------------------

def symbolic_stochastic_attractor(
    sspg: SymbolicStochasticParityGame,
    target: BDDNode,
    player_even: bool,
    restrict: BDDNode = None,
    mode: str = 'almost_sure'
) -> BDDNode:
    """Compute attractor for a player in a stochastic game."""
    bdd = sspg.bdd
    if restrict is None:
        restrict = sspg.vertices

    spg = _make_spg_adapter(sspg)

    # Restrict edges to subgame
    verts_next = _rename_to_next(spg, restrict)
    edges_r = bdd.AND(sspg.edges, restrict)
    edges_r = bdd.AND(edges_r, verts_next)

    # Player's vertices in restrict
    if player_even:
        player_verts = bdd.AND(sspg.owner_even, restrict)
    else:
        player_verts = bdd.AND(_owner_odd(sspg), restrict)

    # Opponent vertices (non-player, non-random)
    if player_even:
        opponent_verts = bdd.AND(_owner_odd(sspg), restrict)
    else:
        opponent_verts = bdd.AND(sspg.owner_even, restrict)

    # Random vertices in restrict
    random_verts = bdd.AND(sspg.owner_random, restrict)

    attr = bdd.AND(target, restrict)
    max_iters = 2 ** sspg.n_bits + 2
    next_idxs = [sspg.next_indices[nv] for nv in sspg.next_vars]

    for _ in range(max_iters):
        # Preimage of attr through restricted edges
        attr_next = _rename_to_next(spg, attr)
        conj = bdd.AND(edges_r, attr_next)
        pre = bdd.exists_multi(next_idxs, conj)

        # Player vertices: any successor in attr
        player_attracted = bdd.AND(player_verts, pre)

        # Opponent vertices: all successors in attr
        not_attr_next = _rename_to_next(spg, bdd.NOT(attr))
        has_outside = bdd.AND(edges_r, not_attr_next)
        pre_outside = bdd.exists_multi(next_idxs, has_outside)
        # Has any successor at all
        pre_any = bdd.exists_multi(next_idxs, edges_r)
        opp_all_in = bdd.AND(opponent_verts, bdd.NOT(pre_outside))
        opp_all_in = bdd.AND(opp_all_in, pre_any)

        # Random vertices: depends on mode
        if mode == 'almost_sure':
            random_attracted = _random_attracted_as(sspg, random_verts, attr, restrict)
        else:
            random_attracted = bdd.AND(random_verts, pre)

        new_attr = bdd.OR(attr, player_attracted)
        new_attr = bdd.OR(new_attr, opp_all_in)
        new_attr = bdd.OR(new_attr, random_attracted)

        if new_attr._id == attr._id:
            break
        attr = new_attr

    return attr


def _random_attracted_as(
    sspg: SymbolicStochasticParityGame,
    random_verts_bdd: BDDNode,
    attr: BDDNode,
    restrict: BDDNode
) -> BDDNode:
    """RANDOM vertices attracted in almost-sure sense.

    A RANDOM vertex is attracted if ALL its positive-prob successors
    (within restrict) are in attr.
    """
    bdd = sspg.bdd
    random_ids = _extract_states_sspg(sspg, random_verts_bdd)
    attr_ids = _extract_states_sspg(sspg, attr)
    restrict_ids = _extract_states_sspg(sspg, restrict)

    result = bdd.FALSE
    for rv in random_ids:
        probs = sspg.probabilities.get(rv, {})
        all_in = True
        has_succ = False
        for succ, prob in probs.items():
            if prob > 0 and succ in restrict_ids:
                has_succ = True
                if succ not in attr_ids:
                    all_in = False
                    break
        if all_in and has_succ:
            rv_bdd = _state_bdd(bdd, rv, sspg.n_bits, sspg.var_indices, sspg.state_vars)
            result = bdd.OR(result, rv_bdd)

    return result


# ---------------------------------------------------------------------------
# Symbolic Zielonka for deterministic parity (used as sub-solver)
# ---------------------------------------------------------------------------

def _symbolic_zielonka(sspg: SymbolicStochasticParityGame, verts: BDDNode,
                       treat_random_as_even: bool = True) -> Tuple[BDDNode, BDDNode]:
    """Recursive Zielonka on a subgame.

    If treat_random_as_even=True, RANDOM vertices are treated as Even-controlled.
    If treat_random_as_even=False, RANDOM vertices are treated as Odd-controlled.
    """
    bdd = sspg.bdd
    if verts._id == bdd.FALSE._id:
        return (bdd.FALSE, bdd.FALSE)

    # Find max priority in verts
    d = -1
    for p in range(sspg.max_priority, -1, -1):
        if p in sspg.priority_bdds:
            test = bdd.AND(verts, sspg.priority_bdds[p])
            if test._id != bdd.FALSE._id:
                d = p
                break
    if d < 0:
        return (verts, bdd.FALSE)

    i_is_even = (d % 2 == 0)

    # U = vertices with max priority d
    u = bdd.AND(verts, sspg.priority_bdds[d])

    # Compute attractor for player i
    a = _deterministic_attractor(sspg, u, i_is_even, verts, treat_random_as_even)

    # Recurse on G \ A
    remaining = bdd.AND(verts, bdd.NOT(a))
    we1, wo1 = _symbolic_zielonka(sspg, remaining, treat_random_as_even)

    # Check opponent's region
    opp_region = wo1 if i_is_even else we1

    if opp_region._id == bdd.FALSE._id:
        if i_is_even:
            return (verts, bdd.FALSE)
        else:
            return (bdd.FALSE, verts)

    # Attractor for opponent
    b = _deterministic_attractor(sspg, opp_region, not i_is_even, verts, treat_random_as_even)

    # Recurse on G \ B
    remaining2 = bdd.AND(verts, bdd.NOT(b))
    we2, wo2 = _symbolic_zielonka(sspg, remaining2, treat_random_as_even)

    if i_is_even:
        return (we2, bdd.OR(b, wo2))
    else:
        return (bdd.OR(b, we2), wo2)


def _deterministic_attractor(
    sspg: SymbolicStochasticParityGame,
    target: BDDNode,
    player_even: bool,
    restrict: BDDNode,
    treat_random_as_even: bool = True
) -> BDDNode:
    """Deterministic attractor treating RANDOM as one of the players."""
    bdd = sspg.bdd
    spg = _make_spg_adapter(sspg)

    verts_next = _rename_to_next(spg, restrict)
    edges_r = bdd.AND(sspg.edges, restrict)
    edges_r = bdd.AND(edges_r, verts_next)

    # Determine which vertices are "player" vs "opponent"
    if player_even:
        player_base = sspg.owner_even
        if treat_random_as_even:
            player_verts = bdd.OR(player_base, sspg.owner_random)
        else:
            player_verts = player_base
    else:
        player_base = _owner_odd(sspg)
        if not treat_random_as_even:
            player_verts = bdd.OR(player_base, sspg.owner_random)
        else:
            player_verts = player_base

    player_verts = bdd.AND(player_verts, restrict)
    opponent_verts = bdd.AND(restrict, bdd.NOT(player_verts))

    attr = bdd.AND(target, restrict)
    max_iters = 2 ** sspg.n_bits + 2
    next_idxs = [sspg.next_indices[nv] for nv in sspg.next_vars]

    for _ in range(max_iters):
        attr_next = _rename_to_next(spg, attr)
        conj = bdd.AND(edges_r, attr_next)
        pre = bdd.exists_multi(next_idxs, conj)

        # Player: any successor in attr
        player_attracted = bdd.AND(player_verts, pre)

        # Opponent: all successors in attr
        not_attr_next = _rename_to_next(spg, bdd.NOT(attr))
        has_outside = bdd.AND(edges_r, not_attr_next)
        pre_outside = bdd.exists_multi(next_idxs, has_outside)
        pre_any = bdd.exists_multi(next_idxs, edges_r)
        opp_all_in = bdd.AND(opponent_verts, bdd.NOT(pre_outside))
        opp_all_in = bdd.AND(opp_all_in, pre_any)

        new_attr = bdd.OR(attr, player_attracted)
        new_attr = bdd.OR(new_attr, opp_all_in)

        if new_attr._id == attr._id:
            break
        attr = new_attr

    return attr


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_almost_sure_symbolic(sspg: SymbolicStochasticParityGame) -> Tuple[Set[int], Set[int], Dict[int, int], Dict[int, int]]:
    """Solve for almost-sure winning regions symbolically.

    Algorithm: iterative refinement
    1. Solve deterministic parity (RANDOM -> EVEN) via Zielonka
    2. Check RANDOM closure: all positive-prob successors of RANDOM in Even's
       region must also be in Even's region
    3. If violated: remove bad RANDOM vertices + Odd attractor, shrink game
       (removed vertices go to Odd's winning region)
    4. Repeat until stable

    Returns: (win_even, win_odd, strategy_even, strategy_odd)
    """
    bdd = sspg.bdd
    verts = sspg.vertices
    all_ids = _extract_states_sspg(sspg, sspg.vertices)

    # Track vertices removed during refinement -- they go to Odd's winning set
    odd_from_refinement = set()

    max_outer = len(all_ids) + 2

    for _ in range(max_outer):
        # Step 1: solve deterministic parity with RANDOM as EVEN
        we_bdd, wo_bdd = _symbolic_zielonka(sspg, verts, treat_random_as_even=True)

        # Step 2: check RANDOM closure
        we_ids = _extract_states_sspg(sspg, we_bdd)
        verts_ids = _extract_states_sspg(sspg, verts)
        random_ids = _extract_states_sspg(sspg, bdd.AND(sspg.owner_random, verts))

        bad_random = set()
        for rv in random_ids:
            if rv not in we_ids:
                continue
            probs = sspg.probabilities.get(rv, {})
            for succ, prob in probs.items():
                if prob > 0:
                    # Bad if successor is in current Odd region OR was already
                    # removed to Odd in a previous refinement iteration
                    if succ in odd_from_refinement:
                        bad_random.add(rv)
                        break
                    if succ in verts_ids and succ not in we_ids:
                        bad_random.add(rv)
                        break

        if not bad_random:
            # Stable -- combine Zielonka's Odd with refinement-removed vertices
            wo_ids = _extract_states_sspg(sspg, wo_bdd)
            final_wo = wo_ids | odd_from_refinement
            strat_even = _extract_stochastic_strategy(sspg, we_ids, True, all_ids)
            strat_odd = _extract_stochastic_strategy(sspg, final_wo, False, all_ids)
            return (we_ids, final_wo, strat_even, strat_odd)

        # Step 3: remove bad RANDOM + Odd attractor
        bad_bdd = bdd.FALSE
        for rv in bad_random:
            bad_bdd = bdd.OR(bad_bdd, _state_bdd(bdd, rv, sspg.n_bits, sspg.var_indices, sspg.state_vars))

        odd_attr = symbolic_stochastic_attractor(sspg, bad_bdd, player_even=False,
                                                  restrict=verts, mode='almost_sure')

        # Vertices in odd_attr go to Odd's winning set
        removed = _extract_states_sspg(sspg, bdd.AND(odd_attr, verts))
        odd_from_refinement |= removed

        verts = bdd.AND(verts, bdd.NOT(odd_attr))

        if verts._id == bdd.FALSE._id:
            return (set(), all_ids, {}, {})

    we_ids = _extract_states_sspg(sspg, verts)
    return (we_ids, all_ids - we_ids, {}, {})


def solve_positive_prob_symbolic(sspg: SymbolicStochasticParityGame) -> Tuple[Set[int], Set[int], Dict[int, int], Dict[int, int]]:
    """Solve for positive-probability winning regions symbolically.

    For PP, treat RANDOM as Even-controlled. Reduces to deterministic parity.
    """
    we_bdd, wo_bdd = _symbolic_zielonka(sspg, sspg.vertices, treat_random_as_even=True)
    we_ids = _extract_states_sspg(sspg, we_bdd)
    wo_ids = _extract_states_sspg(sspg, wo_bdd)
    verts_ids = _extract_states_sspg(sspg, sspg.vertices)

    strat_even = _extract_stochastic_strategy(sspg, we_ids, True, verts_ids)
    strat_odd = _extract_stochastic_strategy(sspg, wo_ids, False, verts_ids)

    return (we_ids, wo_ids, strat_even, strat_odd)


def solve_symbolic_stochastic(game: StochasticParityGame) -> SymbolicStochasticResult:
    """Main API: solve a stochastic parity game symbolically."""
    sspg = stochastic_to_symbolic(game)

    we_as, wo_as, se_as, so_as = solve_almost_sure_symbolic(sspg)
    we_pp, wo_pp, se_pp, so_pp = solve_positive_prob_symbolic(sspg)

    return SymbolicStochasticResult(
        win_even_as=we_as, win_odd_as=wo_as,
        win_even_pp=we_pp, win_odd_pp=wo_pp,
        strategy_even_as=se_as, strategy_odd_as=so_as,
        strategy_even_pp=se_pp, strategy_odd_pp=so_pp,
    )


def solve_symbolic_stochastic_from_sspg(sspg: SymbolicStochasticParityGame) -> SymbolicStochasticResult:
    """Solve directly from a SymbolicStochasticParityGame."""
    we_as, wo_as, se_as, so_as = solve_almost_sure_symbolic(sspg)
    we_pp, wo_pp, se_pp, so_pp = solve_positive_prob_symbolic(sspg)

    return SymbolicStochasticResult(
        win_even_as=we_as, win_odd_as=wo_as,
        win_even_pp=we_pp, win_odd_pp=wo_pp,
        strategy_even_as=se_as, strategy_odd_as=so_as,
        strategy_even_pp=se_pp, strategy_odd_pp=so_pp,
    )


# ---------------------------------------------------------------------------
# Strategy extraction
# ---------------------------------------------------------------------------

def _extract_stochastic_strategy(
    sspg: SymbolicStochasticParityGame,
    win_region: Set[int],
    for_even: bool,
    all_verts: Set[int]
) -> Dict[int, int]:
    """Extract a concrete strategy for a player in their winning region."""
    bdd = sspg.bdd
    spg = _make_spg_adapter(sspg)
    strategy = {}

    for v in win_region:
        v_bdd = _state_bdd(bdd, v, sspg.n_bits, sspg.var_indices, sspg.state_vars)
        is_even = bdd.AND(v_bdd, sspg.owner_even)._id != bdd.FALSE._id
        is_random = bdd.AND(v_bdd, sspg.owner_random)._id != bdd.FALSE._id

        if is_random:
            continue

        if for_even and not is_even:
            continue
        if not for_even and (is_even or is_random):
            continue

        # Find a successor in win_region
        v_edges = bdd.AND(v_bdd, sspg.edges)
        curr_idxs = list(sspg.var_indices.values())
        projected = v_edges
        for idx in curr_idxs:
            projected = bdd.exists(idx, projected)
        succs_bdd = _rename_to_curr(spg, projected)
        succs = _extract_states_sspg(sspg, succs_bdd)

        for w in sorted(succs):
            if w in win_region:
                strategy[v] = w
                break
        else:
            if succs:
                strategy[v] = min(succs)

    return strategy


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_symbolic_stochastic(
    game: StochasticParityGame,
    result: SymbolicStochasticResult
) -> Dict:
    """Verify a symbolic stochastic result against explicit solver."""
    explicit = solve_stochastic_parity(game)

    return {
        'as_agree': result.win_even_as == explicit.win_even_as and result.win_odd_as == explicit.win_odd_as,
        'pp_agree': result.win_even_pp == explicit.win_even_pp and result.win_odd_pp == explicit.win_odd_pp,
        'explicit_as': {'win_even': explicit.win_even_as, 'win_odd': explicit.win_odd_as},
        'symbolic_as': {'win_even': result.win_even_as, 'win_odd': result.win_odd_as},
        'explicit_pp': {'win_even': explicit.win_even_pp, 'win_odd': explicit.win_odd_pp},
        'symbolic_pp': {'win_even': result.win_even_pp, 'win_odd': result.win_odd_pp},
        'as_subset_pp': result.win_even_as.issubset(result.win_even_pp),
    }


def verify_strategy_symbolic(
    game: StochasticParityGame,
    strategy: Dict[int, int],
    win_region: Set[int],
    for_even: bool,
    mode: str = 'almost_sure'
) -> Dict:
    """Verify a strategy for a player in their winning region."""
    errors = []
    checked = 0

    for v in win_region:
        vt = game.vertex_type.get(v)
        if vt == VertexType.RANDOM:
            probs = game.probabilities.get(v, {})
            if mode == 'almost_sure':
                for succ, prob in probs.items():
                    if prob > 0 and succ not in win_region:
                        errors.append(f"RANDOM {v}: succ {succ} (prob {prob}) not in win region")
            else:
                has_good = any(prob > 0 and succ in win_region for succ, prob in probs.items())
                if not has_good:
                    errors.append(f"RANDOM {v}: no positive-prob succ in win region")
            checked += 1
            continue

        is_even = (vt == VertexType.EVEN)
        if for_even and not is_even:
            succs = game.edges.get(v, set())
            for w in succs:
                if w not in win_region:
                    errors.append(f"Opponent {v}: succ {w} not in win region")
            checked += 1
            continue

        if not for_even and is_even:
            succs = game.edges.get(v, set())
            for w in succs:
                if w not in win_region:
                    errors.append(f"Opponent {v}: succ {w} not in win region")
            checked += 1
            continue

        if v in strategy:
            w = strategy[v]
            if w not in game.edges.get(v, set()):
                errors.append(f"Player {v}: strategy {w} not a valid successor")
            elif w not in win_region:
                errors.append(f"Player {v}: strategy {w} not in win region")
        checked += 1

    return {'valid': len(errors) == 0, 'errors': errors, 'checked': checked}


# ---------------------------------------------------------------------------
# Game construction helpers
# ---------------------------------------------------------------------------

def make_stochastic_game(
    vertices: List[Tuple[int, str, int]],
    edges: List[Tuple[int, int]],
    probs: Optional[Dict[int, Dict[int, float]]] = None
) -> StochasticParityGame:
    """Convenience: create a StochasticParityGame."""
    type_map = {'even': VertexType.EVEN, 'odd': VertexType.ODD, 'random': VertexType.RANDOM}
    game = StochasticParityGame(
        vertices=set(), edges={}, vertex_type={}, priority={}, probabilities={}
    )
    for vid, vtype, prio in vertices:
        game.add_vertex(vid, type_map[vtype], prio)

    if probs:
        for u, v in edges:
            prob = probs.get(u, {}).get(v, 1.0)
            game.add_edge(u, v, prob)
    else:
        for u, v in edges:
            game.add_edge(u, v)

    return game


def make_symbolic_stochastic_chain(n: int, random_vertex: int,
                                    prob_forward: float = 0.5) -> StochasticParityGame:
    """Chain game with one RANDOM vertex."""
    game = StochasticParityGame(
        vertices=set(), edges={}, vertex_type={}, priority={}, probabilities={}
    )
    for v in range(n):
        if v == random_vertex:
            vt = VertexType.RANDOM
        elif v % 2 == 0:
            vt = VertexType.EVEN
        else:
            vt = VertexType.ODD
        game.add_vertex(v, vt, v % 3)

    for v in range(n):
        nxt = (v + 1) % n
        if v == random_vertex:
            game.add_edge(v, nxt, prob_forward)
            if v != nxt:
                game.add_edge(v, v, 1.0 - prob_forward)
        else:
            game.add_edge(v, nxt)

    return game


def make_symbolic_reachability_stochastic(
    n_states: int,
    target: Set[int],
    even_states: Set[int],
    random_states: Set[int],
    transitions: List[Tuple[int, int]],
    probs: Optional[Dict[int, Dict[int, float]]] = None
) -> StochasticParityGame:
    """Reachability stochastic game."""
    game = StochasticParityGame(
        vertices=set(), edges={}, vertex_type={}, priority={}, probabilities={}
    )
    for v in range(n_states):
        if v in random_states:
            vt = VertexType.RANDOM
        elif v in even_states:
            vt = VertexType.EVEN
        else:
            vt = VertexType.ODD
        prio = 2 if v in target else 1
        game.add_vertex(v, vt, prio)

    for u, w in transitions:
        prob = 1.0
        if probs and u in probs:
            prob = probs[u].get(w, 1.0)
        game.add_edge(u, w, prob)

    for t in target:
        if t not in game.edges or not game.edges[t]:
            game.add_edge(t, t)

    return game


def make_symbolic_safety_stochastic(
    n_states: int,
    bad: Set[int],
    even_states: Set[int],
    random_states: Set[int],
    transitions: List[Tuple[int, int]],
    probs: Optional[Dict[int, Dict[int, float]]] = None
) -> StochasticParityGame:
    """Safety stochastic game."""
    game = StochasticParityGame(
        vertices=set(), edges={}, vertex_type={}, priority={}, probabilities={}
    )
    for v in range(n_states):
        if v in random_states:
            vt = VertexType.RANDOM
        elif v in even_states:
            vt = VertexType.EVEN
        else:
            vt = VertexType.ODD
        prio = 1 if v in bad else 0
        game.add_vertex(v, vt, prio)

    for u, w in transitions:
        prob = 1.0
        if probs and u in probs:
            prob = probs[u].get(w, 1.0)
        game.add_edge(u, w, prob)

    return game


def make_symbolic_buchi_stochastic(
    n_states: int,
    accepting: Set[int],
    even_states: Set[int],
    random_states: Set[int],
    transitions: List[Tuple[int, int]],
    probs: Optional[Dict[int, Dict[int, float]]] = None
) -> StochasticParityGame:
    """Buchi stochastic game."""
    game = StochasticParityGame(
        vertices=set(), edges={}, vertex_type={}, priority={}, probabilities={}
    )
    for v in range(n_states):
        if v in random_states:
            vt = VertexType.RANDOM
        elif v in even_states:
            vt = VertexType.EVEN
        else:
            vt = VertexType.ODD
        prio = 2 if v in accepting else 1
        game.add_vertex(v, vt, prio)

    for u, w in transitions:
        prob = 1.0
        if probs and u in probs:
            prob = probs[u].get(w, 1.0)
        game.add_edge(u, w, prob)

    return game


# ---------------------------------------------------------------------------
# Comparison & statistics
# ---------------------------------------------------------------------------

def compare_explicit_vs_symbolic(game: StochasticParityGame) -> Dict:
    """Compare explicit V165 solver with symbolic V169 solver."""
    explicit = solve_stochastic_parity(game)
    symbolic = solve_symbolic_stochastic(game)

    return {
        'explicit': {
            'win_even_as': explicit.win_even_as,
            'win_odd_as': explicit.win_odd_as,
            'win_even_pp': explicit.win_even_pp,
            'win_odd_pp': explicit.win_odd_pp,
        },
        'symbolic': {
            'win_even_as': symbolic.win_even_as,
            'win_odd_as': symbolic.win_odd_as,
            'win_even_pp': symbolic.win_even_pp,
            'win_odd_pp': symbolic.win_odd_pp,
        },
        'as_agree': explicit.win_even_as == symbolic.win_even_as and explicit.win_odd_as == symbolic.win_odd_as,
        'pp_agree': explicit.win_even_pp == symbolic.win_even_pp and explicit.win_odd_pp == symbolic.win_odd_pp,
        'as_subset_pp': symbolic.win_even_as.issubset(symbolic.win_even_pp),
    }


def symbolic_stochastic_statistics(game: StochasticParityGame) -> Dict:
    """Compute statistics about a stochastic parity game."""
    sspg = stochastic_to_symbolic(game)
    even_count = len(_extract_states_sspg(sspg, sspg.owner_even))
    random_count = len(_extract_states_sspg(sspg, sspg.owner_random))
    odd_count = len(game.vertices) - even_count - random_count

    prio_dist = {}
    for p, p_bdd in sspg.priority_bdds.items():
        prio_dist[p] = len(_extract_states_sspg(sspg, p_bdd))

    return {
        'vertices': len(game.vertices),
        'edges': sum(len(s) for s in game.edges.values()),
        'even_vertices': even_count,
        'odd_vertices': odd_count,
        'random_vertices': random_count,
        'max_priority': sspg.max_priority,
        'n_bits': sspg.n_bits,
        'distinct_priorities': len(sspg.priority_bdds),
        'priority_distribution': prio_dist,
    }


def batch_solve(games: List[Tuple[str, StochasticParityGame]]) -> Dict[str, SymbolicStochasticResult]:
    """Solve multiple named stochastic parity games symbolically."""
    results = {}
    for name, game in games:
        results[name] = solve_symbolic_stochastic(game)
    return results
