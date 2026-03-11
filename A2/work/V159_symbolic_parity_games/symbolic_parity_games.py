"""V159: Symbolic Parity Games -- BDD-based parity game solving.

Composes V021 (BDD) + V156 (Parity Games) to solve parity games symbolically.
Enables solving games with exponentially large state spaces via compact BDD
representation.

Key ideas:
- Vertices encoded as bit-vectors (n_bits = ceil(log2(num_vertices)))
- Owner: BDD over state vars (TRUE = Even, FALSE = Odd)
- Priority: one BDD per priority level (priority_p = set of vertices with priority p)
- Edges: transition relation BDD over curr + next state vars
- Attractor: symbolic fixpoint using preimage
- Zielonka: recursive algorithm using BDD set operations
"""

import sys, os, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))

from bdd_model_checker import BDD, BDDNode
from parity_games import ParityGame, Player, Solution, verify_solution


@dataclass
class SymbolicParityGame:
    """BDD-encoded parity game."""
    bdd: BDD
    n_bits: int                          # bits per vertex
    state_vars: List[str]                # current-state var names
    next_vars: List[str]                 # next-state var names
    var_indices: Dict[str, int]          # curr var name -> BDD index
    next_indices: Dict[str, int]         # next var name -> BDD index
    vertices: BDDNode                    # valid vertices (may exclude padding)
    edges: BDDNode                       # transition relation (curr x next)
    owner_even: BDDNode                  # vertices owned by Even
    priority_bdds: Dict[int, BDDNode]    # priority -> BDD of vertices with that priority
    max_priority: int                    # highest priority in game


@dataclass
class SymbolicSolution:
    """Solution to a symbolic parity game."""
    win_even: BDDNode        # BDD of vertices won by Even
    win_odd: BDDNode         # BDD of vertices won by Odd
    strategy_even: BDDNode   # BDD over curr x next: Even's strategy edges
    strategy_odd: BDDNode    # BDD over curr x next: Odd's strategy edges


# --- BDD Helper Operations ---

def _state_bdd(bdd: BDD, state_id: int, n_bits: int,
               var_indices: Dict[str, int], var_names: List[str]) -> BDDNode:
    """Encode a single state as a BDD conjunction."""
    result = bdd.TRUE
    for bit in range(n_bits):
        var_node = bdd.var(var_indices[var_names[bit]])
        if (state_id >> bit) & 1:
            result = bdd.AND(result, var_node)
        else:
            result = bdd.AND(result, bdd.NOT(var_node))
    return result


def _rename_to_next(spg: SymbolicParityGame, phi: BDDNode) -> BDDNode:
    """Rename current-state variables to next-state variables in phi."""
    var_map = {}
    for i, sv in enumerate(spg.state_vars):
        curr_idx = spg.var_indices[sv]
        nxt_idx = spg.next_indices[spg.next_vars[i]]
        var_map[curr_idx] = nxt_idx
    return spg.bdd.rename(phi, var_map)


def _rename_to_curr(spg: SymbolicParityGame, phi: BDDNode) -> BDDNode:
    """Rename next-state variables to current-state variables in phi."""
    var_map = {}
    for i, sv in enumerate(spg.state_vars):
        curr_idx = spg.var_indices[sv]
        nxt_idx = spg.next_indices[spg.next_vars[i]]
        var_map[nxt_idx] = curr_idx
    return spg.bdd.rename(phi, var_map)


def _preimage(spg: SymbolicParityGame, target: BDDNode) -> BDDNode:
    """Compute predecessors: {v | exists w in target. (v,w) in edges}."""
    target_next = _rename_to_next(spg, target)
    conj = spg.bdd.AND(spg.edges, target_next)
    next_idxs = [spg.next_indices[nv] for nv in spg.next_vars]
    return spg.bdd.exists_multi(next_idxs, conj)


def _image(spg: SymbolicParityGame, source: BDDNode) -> BDDNode:
    """Compute successors: {w | exists v in source. (v,w) in edges}."""
    conj = spg.bdd.AND(spg.edges, source)
    curr_idxs = [spg.var_indices[sv] for sv in spg.state_vars]
    projected = spg.bdd.exists_multi(curr_idxs, conj)
    return _rename_to_curr(spg, projected)


def _forall_successors_in(spg: SymbolicParityGame, source: BDDNode,
                          target: BDDNode) -> BDDNode:
    """Vertices in source where ALL successors are in target.
    {v in source | forall w. (v,w) in edges => w in target}
    = source AND NOT(pre(NOT target))
    """
    not_target = spg.bdd.NOT(target)
    pre_not = _preimage(spg, not_target)
    return spg.bdd.AND(source, spg.bdd.NOT(pre_not))


# --- Symbolic Attractor ---

def symbolic_attractor(spg: SymbolicParityGame, target: BDDNode,
                       player_even: bool, restrict: BDDNode = None) -> BDDNode:
    """Compute attractor of target for player within restrict.

    Attr_p(T, V) = least fixpoint of:
      X = T | {v in V_p | exists w in X. (v,w) in E}
            | {v in V_{1-p} | forall w in V. (v,w) in E => w in X}

    Args:
        spg: symbolic parity game
        target: initial target set (BDD)
        player_even: True for Even's attractor, False for Odd's
        restrict: restrict to this vertex set (default: all vertices)
    Returns:
        BDD of attractor set
    """
    bdd = spg.bdd
    if restrict is None:
        restrict = spg.vertices

    # Restrict edges to vertices in restrict
    restrict_next = _rename_to_next(spg, restrict)
    edges_restricted = bdd.AND(spg.edges, bdd.AND(restrict, restrict_next))

    player_verts = bdd.AND(spg.owner_even if player_even else
                           bdd.NOT(spg.owner_even), restrict)
    opponent_verts = bdd.AND(bdd.NOT(spg.owner_even) if player_even else
                             spg.owner_even, restrict)

    attr = bdd.AND(target, restrict)

    for _ in range(2 ** spg.n_bits + 2):
        # Player vertices with at least one successor in attr
        attr_next = _rename_to_next(spg, attr)
        conj = bdd.AND(edges_restricted, attr_next)
        next_idxs = [spg.next_indices[nv] for nv in spg.next_vars]
        pre_attr = bdd.exists_multi(next_idxs, conj)
        player_attracted = bdd.AND(player_verts, pre_attr)

        # Opponent vertices with ALL successors in attr
        # v in opponent AND for all w: (v,w) in E_restricted => w in attr
        not_attr = bdd.NOT(attr)
        not_attr_next = _rename_to_next(spg, not_attr)
        has_escape = bdd.exists_multi(next_idxs, bdd.AND(edges_restricted, not_attr_next))
        opponent_attracted = bdd.AND(opponent_verts, bdd.NOT(has_escape))

        new_attr = bdd.OR(attr, bdd.OR(player_attracted, opponent_attracted))
        if new_attr._id == attr._id:
            break
        attr = new_attr

    return attr


# --- Symbolic Zielonka ---

def symbolic_zielonka(spg: SymbolicParityGame) -> SymbolicSolution:
    """Solve a symbolic parity game using Zielonka's recursive algorithm.

    Returns SymbolicSolution with winning regions and strategies.
    """
    bdd = spg.bdd
    win_even = bdd.FALSE
    win_odd = bdd.FALSE

    _symbolic_zielonka_rec(spg, spg.vertices, win_even, win_odd,
                           result_holder=[bdd.FALSE, bdd.FALSE])

    we = result_holder_hack[0]
    wo = result_holder_hack[1]
    # Compute strategies
    strat_even = _compute_symbolic_strategy(spg, we, True)
    strat_odd = _compute_symbolic_strategy(spg, wo, False)

    return SymbolicSolution(
        win_even=we, win_odd=wo,
        strategy_even=strat_even, strategy_odd=strat_odd
    )


# Use a cleaner approach: return (win_even, win_odd)
def _solve_rec(spg: SymbolicParityGame, verts: BDDNode) -> Tuple[BDDNode, BDDNode]:
    """Recursive Zielonka on symbolic parity game.

    Args:
        spg: the symbolic game
        verts: BDD of vertices in current subgame

    Returns:
        (win_even, win_odd) BDDs within verts
    """
    bdd = spg.bdd

    # Base case: empty
    if verts._id == bdd.FALSE._id:
        return (bdd.FALSE, bdd.FALSE)

    # Handle dead ends: vertices with no successors in verts
    verts_next = _rename_to_next(spg, verts)
    edges_sub = bdd.AND(spg.edges, bdd.AND(verts, verts_next))
    next_idxs = [spg.next_indices[nv] for nv in spg.next_vars]
    has_succ = bdd.exists_multi(next_idxs, edges_sub)
    dead_ends = bdd.AND(verts, bdd.NOT(has_succ))

    if dead_ends._id != bdd.FALSE._id:
        # Dead-end Even vertices -> Odd wins; dead-end Odd vertices -> Even wins
        dead_even = bdd.AND(dead_ends, spg.owner_even)
        dead_odd = bdd.AND(dead_ends, bdd.NOT(spg.owner_even))

        # Remove dead ends and their attractors
        odd_attr = symbolic_attractor(spg, dead_even, False, verts)  # Odd wins dead Even
        even_attr = symbolic_attractor(spg, dead_odd, True, verts)   # Even wins dead Odd

        remaining = bdd.AND(verts, bdd.AND(bdd.NOT(odd_attr), bdd.NOT(even_attr)))
        we_rec, wo_rec = _solve_rec(spg, remaining)

        return (bdd.OR(even_attr, we_rec), bdd.OR(odd_attr, wo_rec))

    # Find max priority in verts
    d = -1
    for p in range(spg.max_priority, -1, -1):
        if p in spg.priority_bdds:
            overlap = bdd.AND(spg.priority_bdds[p], verts)
            if overlap._id != bdd.FALSE._id:
                d = p
                break

    if d < 0:
        return (bdd.FALSE, bdd.FALSE)

    # Player i "likes" priority d (Even likes even, Odd likes odd)
    i_is_even = (d % 2 == 0)

    # U = vertices with priority d in verts
    u = bdd.AND(spg.priority_bdds[d], verts)

    # A = Attr_i(U) within verts
    a = symbolic_attractor(spg, u, i_is_even, verts)

    # Recursively solve subgame G \ A
    remaining = bdd.AND(verts, bdd.NOT(a))
    we1, wo1 = _solve_rec(spg, remaining)

    # Check if opponent wins anything
    opp_region = wo1 if i_is_even else we1
    if opp_region._id == bdd.FALSE._id:
        # Player i wins everything
        if i_is_even:
            return (verts, bdd.FALSE)
        else:
            return (bdd.FALSE, verts)

    # Opponent wins something: compute Attr_{1-i}(opp_region) within verts
    b = symbolic_attractor(spg, opp_region, not i_is_even, verts)

    # Recursively solve G \ B
    remaining2 = bdd.AND(verts, bdd.NOT(b))
    we2, wo2 = _solve_rec(spg, remaining2)

    if i_is_even:
        return (we2, bdd.OR(b, wo2))
    else:
        return (bdd.OR(b, we2), wo2)


def solve_symbolic(spg: SymbolicParityGame) -> SymbolicSolution:
    """Solve a symbolic parity game using Zielonka's recursive algorithm."""
    bdd = spg.bdd
    we, wo = _solve_rec(spg, spg.vertices)

    strat_even = _compute_symbolic_strategy(spg, we, True)
    strat_odd = _compute_symbolic_strategy(spg, wo, False)

    return SymbolicSolution(
        win_even=we, win_odd=wo,
        strategy_even=strat_even, strategy_odd=strat_odd
    )


def _compute_symbolic_strategy(spg: SymbolicParityGame, win_region: BDDNode,
                                for_even: bool) -> BDDNode:
    """Compute a winning strategy BDD for a player in their winning region.

    Strategy is a BDD over curr x next vars selecting one successor per vertex.
    For player-owned vertices: pick any successor in win_region.
    """
    bdd = spg.bdd
    player_verts = bdd.AND(win_region,
                           spg.owner_even if for_even else bdd.NOT(spg.owner_even))

    # Strategy: edges from player_verts to win_region
    win_next = _rename_to_next(spg, win_region)
    strategy = bdd.AND(spg.edges, bdd.AND(player_verts, win_next))
    return strategy


# --- Conversion: Explicit <-> Symbolic ---

def explicit_to_symbolic(game: ParityGame) -> SymbolicParityGame:
    """Convert a V156 ParityGame to a SymbolicParityGame."""
    if not game.vertices:
        n_bits = 1
    else:
        n_bits = max(1, math.ceil(math.log2(max(game.vertices) + 1)))

    # If max vertex + 1 is a power of 2, n_bits is exact; otherwise add 1 if needed
    if (1 << n_bits) < max(game.vertices, default=0) + 1:
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

    # Encode vertices
    vertices_bdd = bdd.FALSE
    for v in game.vertices:
        vertices_bdd = bdd.OR(vertices_bdd,
                              _state_bdd(bdd, v, n_bits, var_indices, state_vars))

    # Encode owner (Even)
    owner_even_bdd = bdd.FALSE
    for v in game.vertices:
        if game.owner.get(v) == Player.EVEN:
            owner_even_bdd = bdd.OR(owner_even_bdd,
                                     _state_bdd(bdd, v, n_bits, var_indices, state_vars))

    # Encode edges
    edges_bdd = bdd.FALSE
    for u in game.vertices:
        u_bdd = _state_bdd(bdd, u, n_bits, var_indices, state_vars)
        for w in game.edges.get(u, set()):
            w_bdd = _state_bdd(bdd, w, n_bits, next_indices, next_vars)
            edges_bdd = bdd.OR(edges_bdd, bdd.AND(u_bdd, w_bdd))

    # Encode priorities
    priority_bdds = {}
    max_prio = game.max_priority()
    for v in game.vertices:
        p = game.priority[v]
        if p not in priority_bdds:
            priority_bdds[p] = bdd.FALSE
        priority_bdds[p] = bdd.OR(priority_bdds[p],
                                   _state_bdd(bdd, v, n_bits, var_indices, state_vars))

    return SymbolicParityGame(
        bdd=bdd, n_bits=n_bits,
        state_vars=state_vars, next_vars=next_vars,
        var_indices=var_indices, next_indices=next_indices,
        vertices=vertices_bdd, edges=edges_bdd,
        owner_even=owner_even_bdd,
        priority_bdds=priority_bdds,
        max_priority=max_prio
    )


def symbolic_to_explicit(spg: SymbolicParityGame) -> ParityGame:
    """Convert a SymbolicParityGame back to explicit V156 ParityGame."""
    game = ParityGame()
    bdd = spg.bdd

    # Extract vertices
    vert_ids = _extract_states(spg, spg.vertices)

    for v in vert_ids:
        v_bdd = _state_bdd(bdd, v, spg.n_bits, spg.var_indices, spg.state_vars)
        # Check owner
        is_even = bdd.AND(v_bdd, spg.owner_even)
        player = Player.EVEN if is_even._id != bdd.FALSE._id else Player.ODD
        # Find priority
        prio = 0
        for p, p_bdd in spg.priority_bdds.items():
            if bdd.AND(v_bdd, p_bdd)._id != bdd.FALSE._id:
                prio = p
                break
        game.add_vertex(v, player, prio)

    # Extract edges
    for u in vert_ids:
        u_bdd = _state_bdd(bdd, u, spg.n_bits, spg.var_indices, spg.state_vars)
        # Find successors
        conj = bdd.AND(spg.edges, u_bdd)
        curr_idxs = [spg.var_indices[sv] for sv in spg.state_vars]
        projected = bdd.exists_multi(curr_idxs, conj)
        succ_bdd = _rename_to_curr(spg, projected)
        succ_ids = _extract_states(spg, succ_bdd)
        for w in succ_ids:
            game.add_edge(u, w)

    return game


def _extract_states(spg: SymbolicParityGame, states_bdd: BDDNode) -> Set[int]:
    """Extract concrete state IDs from a BDD over current-state variables."""
    bdd = spg.bdd
    if states_bdd._id == bdd.FALSE._id:
        return set()

    curr_idxs = [spg.var_indices[sv] for sv in spg.state_vars]
    max_idx = max(curr_idxs) + 1 if curr_idxs else 0
    assignments = bdd.all_sat(states_bdd, num_vars=max_idx)

    states = set()
    for asgn in assignments:
        fixed_bits = {}
        free_bits = []
        for i, sv in enumerate(spg.state_vars):
            idx = spg.var_indices[sv]
            if idx in asgn:
                fixed_bits[i] = 1 if asgn[idx] else 0
            else:
                free_bits.append(i)

        for mask in range(1 << len(free_bits)):
            state_num = 0
            for i_bit, val in fixed_bits.items():
                state_num |= (val << i_bit)
            for j, bit_pos in enumerate(free_bits):
                if (mask >> j) & 1:
                    state_num |= (1 << bit_pos)
            states.add(state_num)

    # Filter to valid vertices
    valid_states = set()
    for s in states:
        s_bdd = _state_bdd(bdd, s, spg.n_bits, spg.var_indices, spg.state_vars)
        if bdd.AND(s_bdd, spg.vertices)._id != bdd.FALSE._id:
            valid_states.add(s)

    return valid_states


def extract_winning_sets(spg: SymbolicParityGame,
                         sol: SymbolicSolution) -> Tuple[Set[int], Set[int]]:
    """Extract concrete winning sets from symbolic solution."""
    we = _extract_states(spg, sol.win_even)
    wo = _extract_states(spg, sol.win_odd)
    return we, wo


def extract_strategy(spg: SymbolicParityGame,
                     strategy_bdd: BDDNode) -> Dict[int, int]:
    """Extract concrete strategy from symbolic strategy BDD."""
    bdd = spg.bdd
    if strategy_bdd._id == bdd.FALSE._id:
        return {}

    strategy = {}
    # Get all vertices that have a strategy edge
    next_idxs = [spg.next_indices[nv] for nv in spg.next_vars]
    has_strat = bdd.exists_multi(next_idxs, strategy_bdd)
    vert_ids = _extract_states(spg, has_strat)

    for v in vert_ids:
        v_bdd = _state_bdd(bdd, v, spg.n_bits, spg.var_indices, spg.state_vars)
        conj = bdd.AND(strategy_bdd, v_bdd)
        curr_idxs = [spg.var_indices[sv] for sv in spg.state_vars]
        projected = bdd.exists_multi(curr_idxs, conj)
        succ_bdd = _rename_to_curr(spg, projected)
        succs = _extract_states(spg, succ_bdd)
        if succs:
            strategy[v] = min(succs)  # Pick smallest successor deterministically

    return strategy


# --- Parametric Game Constructors ---

def make_symbolic_chain_game(n: int) -> SymbolicParityGame:
    """Create a chain game: 0 -> 1 -> 2 -> ... -> n-1 -> 0.

    Even owns even vertices, Odd owns odd vertices.
    Priority = vertex id mod 3.
    """
    game = ParityGame()
    for i in range(n):
        player = Player.EVEN if i % 2 == 0 else Player.ODD
        game.add_vertex(i, player, i % 3)
    for i in range(n):
        game.add_edge(i, (i + 1) % n)
    return explicit_to_symbolic(game)


def make_symbolic_ladder_game(n: int) -> SymbolicParityGame:
    """Create a ladder game: two parallel chains with cross-edges.

    Top row: vertices 0..n-1 (Even-owned)
    Bottom row: vertices n..2n-1 (Odd-owned)
    Top chain: 0->1->...->n-1->0
    Bottom chain: n->n+1->...->2n-1->n
    Cross-edges: i <-> i+n
    Priorities: top row = 0 (good for Even), bottom row = 1 (good for Odd)
    """
    game = ParityGame()
    for i in range(n):
        game.add_vertex(i, Player.EVEN, 0)
        game.add_vertex(i + n, Player.ODD, 1)
    for i in range(n):
        game.add_edge(i, (i + 1) % n)          # top chain
        game.add_edge(i + n, ((i + 1) % n) + n)  # bottom chain
        game.add_edge(i, i + n)                  # cross down
        game.add_edge(i + n, i)                  # cross up
    return explicit_to_symbolic(game)


def make_symbolic_safety_game(n_states: int, bad: Set[int],
                              even_states: Set[int],
                              transitions: List[Tuple[int, int]]) -> SymbolicParityGame:
    """Create a safety game symbolically.

    Even wins iff the play never visits bad states.
    Priority 0 for safe states, priority 1 for bad states.
    """
    game = ParityGame()
    for s in range(n_states):
        player = Player.EVEN if s in even_states else Player.ODD
        prio = 1 if s in bad else 0
        game.add_vertex(s, player, prio)
    for u, v in transitions:
        game.add_edge(u, v)
    return explicit_to_symbolic(game)


def make_symbolic_reachability_game(n_states: int, target: Set[int],
                                    even_states: Set[int],
                                    transitions: List[Tuple[int, int]]) -> SymbolicParityGame:
    """Create a reachability game symbolically.

    Even wins iff the play reaches a target state.
    Priority 1 for non-target, priority 2 for target.
    Target states get self-loops.
    """
    game = ParityGame()
    for s in range(n_states):
        player = Player.EVEN if s in even_states else Player.ODD
        prio = 2 if s in target else 1
        game.add_vertex(s, player, prio)
    for u, v in transitions:
        game.add_edge(u, v)
    for t in target:
        game.add_edge(t, t)
    return explicit_to_symbolic(game)


def make_symbolic_buchi_game(n_states: int, accepting: Set[int],
                             even_states: Set[int],
                             transitions: List[Tuple[int, int]]) -> SymbolicParityGame:
    """Create a Buchi game symbolically.

    Even wins iff accepting states visited infinitely often.
    Priority 1 for non-accepting, priority 2 for accepting.
    """
    game = ParityGame()
    for s in range(n_states):
        player = Player.EVEN if s in even_states else Player.ODD
        prio = 2 if s in accepting else 1
        game.add_vertex(s, player, prio)
    for u, v in transitions:
        game.add_edge(u, v)
    return explicit_to_symbolic(game)


# --- Verification ---

def verify_symbolic_solution(spg: SymbolicParityGame,
                             sol: SymbolicSolution) -> Tuple[bool, List[str]]:
    """Verify a symbolic solution by converting to explicit and using V156 verifier."""
    game = symbolic_to_explicit(spg)
    we, wo = extract_winning_sets(spg, sol)

    explicit_sol = Solution(
        win_even=we, win_odd=wo,
        strategy_even=extract_strategy(spg, sol.strategy_even),
        strategy_odd=extract_strategy(spg, sol.strategy_odd)
    )
    return verify_solution(game, explicit_sol)


# --- Comparison ---

def compare_explicit_vs_symbolic(game: ParityGame) -> Dict:
    """Compare explicit V156 solving with symbolic V159 solving."""
    from parity_games import zielonka

    # Explicit
    explicit_sol = zielonka(game)
    valid_e, errors_e = verify_solution(game, explicit_sol)

    # Symbolic
    spg = explicit_to_symbolic(game)
    sym_sol = solve_symbolic(spg)
    we_sym, wo_sym = extract_winning_sets(spg, sym_sol)
    valid_s, errors_s = verify_symbolic_solution(spg, sym_sol)

    agree = (explicit_sol.win_even == we_sym and explicit_sol.win_odd == wo_sym)

    return {
        'explicit': {
            'win_even': explicit_sol.win_even,
            'win_odd': explicit_sol.win_odd,
            'valid': valid_e,
            'errors': errors_e,
        },
        'symbolic': {
            'win_even': we_sym,
            'win_odd': wo_sym,
            'valid': valid_s,
            'errors': errors_s,
        },
        'agree': agree,
        'n_bits': spg.n_bits,
        'bdd_nodes': spg.bdd.node_count(spg.edges),
    }


# --- High-level APIs ---

def solve_parity_game(game: ParityGame) -> Dict:
    """Solve a parity game symbolically. Returns dict with winning sets, strategies, validity."""
    spg = explicit_to_symbolic(game)
    sol = solve_symbolic(spg)
    we, wo = extract_winning_sets(spg, sol)
    valid, errors = verify_symbolic_solution(spg, sol)

    return {
        'win_even': we,
        'win_odd': wo,
        'strategy_even': extract_strategy(spg, sol.strategy_even),
        'strategy_odd': extract_strategy(spg, sol.strategy_odd),
        'valid': valid,
        'errors': errors,
        'n_bits': spg.n_bits,
    }


def symbolic_game_stats(spg: SymbolicParityGame) -> Dict:
    """Compute statistics about a symbolic parity game."""
    bdd = spg.bdd
    n_verts = len(_extract_states(spg, spg.vertices))
    prio_dist = {}
    for p, p_bdd in spg.priority_bdds.items():
        count = len(_extract_states(spg, bdd.AND(p_bdd, spg.vertices)))
        if count > 0:
            prio_dist[p] = count

    even_count = len(_extract_states(spg, bdd.AND(spg.owner_even, spg.vertices)))

    return {
        'vertices': n_verts,
        'n_bits': spg.n_bits,
        'max_priority': spg.max_priority,
        'num_priorities': len(prio_dist),
        'even_vertices': even_count,
        'odd_vertices': n_verts - even_count,
        'priority_distribution': prio_dist,
        'edge_bdd_nodes': bdd.node_count(spg.edges),
        'vertex_bdd_nodes': bdd.node_count(spg.vertices),
    }


def batch_solve(games: List[ParityGame]) -> List[Dict]:
    """Solve multiple parity games symbolically."""
    return [solve_parity_game(g) for g in games]
