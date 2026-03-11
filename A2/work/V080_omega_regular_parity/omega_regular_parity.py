"""V080: Omega-Regular Game Solving via Parity Reduction.

Solves two-player games with general omega-regular winning conditions by
reducing them to parity games and applying V076's solvers (Zielonka/SPM).

Key idea: Any omega-regular condition (Buchi, co-Buchi, Rabin, Streett,
Muller, parity, or LTL-to-automaton) can be encoded as a parity game.
This module provides a unified interface for all such reductions.

Composes:
- V076 (parity games): ParityGame, solve, Zielonka, SPM, conversion functions
- V074 (omega-regular games): LTL-to-NBA pipeline (via V023)
- V023 (LTL model checking): LTL AST, parser, tableau construction

Features:
1. Unified acceptance condition types: Buchi, co-Buchi, Rabin, Streett, Muller, Parity
2. Game arena + acceptance -> parity game -> solve -> winning regions + strategies
3. LTL-to-parity: LTL formula -> NBA -> product arena -> parity game
4. Muller condition support (new -- not in V076)
5. Acceptance condition composition (conjunction, disjunction)
6. Algorithm comparison across different reductions
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from enum import Enum
from itertools import product as itertools_product

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)

# V076: Parity game solver + conversion functions
sys.path.insert(0, os.path.join(_work, 'V076_parity_games'))
from parity_games import (
    ParityGame, ParityResult, Player as PGPlayer,
    BuchiGame, CoBuchiGame, RabinPair, StreettPair,
    buchi_to_parity, cobuchi_to_parity, rabin_to_parity, streett_to_parity,
    zielonka, small_progress_measures,
    compress_priorities, attractor, verify_strategy as pg_verify_strategy,
    compare_algorithms, make_game as pg_make_game,
)

# V023: LTL AST and automaton construction
sys.path.insert(0, os.path.join(_work, 'V023_ltl_model_checking'))
from ltl_model_checker import (
    LTL, Atom, LTLTrue, LTLFalse, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release, WeakUntil,
    atoms, nnf, ltl_to_gba, gba_to_nba, NBA, Label,
    parse_ltl,
)


# ============================================================
# Data structures
# ============================================================

class AccType(Enum):
    """Types of omega-regular acceptance conditions."""
    BUCHI = "buchi"
    COBUCHI = "cobuchi"
    RABIN = "rabin"
    STREETT = "streett"
    MULLER = "muller"
    PARITY = "parity"


@dataclass
class GameArena:
    """A two-player game arena (deterministic, no stochastic).

    Attributes:
        nodes: set of node ids (ints)
        owner: maps node -> 0 (Player Even/Protagonist) or 1 (Player Odd/Antagonist)
        successors: maps node -> set of successor nodes
        labels: maps node -> set of atomic proposition names (for automaton products)
    """
    nodes: Set[int] = field(default_factory=set)
    owner: Dict[int, int] = field(default_factory=dict)  # 0 or 1
    successors: Dict[int, Set[int]] = field(default_factory=dict)
    labels: Dict[int, Set[str]] = field(default_factory=dict)

    def add_node(self, node: int, owner: int, props: Optional[Set[str]] = None) -> None:
        self.nodes.add(node)
        self.owner[node] = owner
        self.successors.setdefault(node, set())
        if props is not None:
            self.labels[node] = props
        elif node not in self.labels:
            self.labels[node] = set()

    def add_edge(self, src: int, dst: int) -> None:
        self.successors.setdefault(src, set()).add(dst)

    def validate(self) -> List[str]:
        issues = []
        for n in self.nodes:
            if n not in self.owner:
                issues.append(f"Node {n} has no owner")
            if not self.successors.get(n):
                issues.append(f"Node {n} has no successors")
            if self.owner.get(n) not in (0, 1):
                issues.append(f"Node {n} owner must be 0 or 1")
        return issues


@dataclass
class AcceptanceCondition:
    """A general omega-regular acceptance condition.

    Use the factory classmethods to construct specific types.
    """
    acc_type: AccType
    # Buchi: accepting nodes
    accepting: Optional[Set[int]] = None
    # co-Buchi: rejecting nodes
    rejecting: Optional[Set[int]] = None
    # Rabin: list of (fin, inf) pairs
    rabin_pairs: Optional[List[Tuple[Set[int], Set[int]]]] = None
    # Streett: list of (request, response) pairs
    streett_pairs: Optional[List[Tuple[Set[int], Set[int]]]] = None
    # Muller: set of accepting sets (frozensets)
    muller_table: Optional[Set[FrozenSet[int]]] = None
    # Parity: node -> priority mapping
    priorities: Optional[Dict[int, int]] = None

    @classmethod
    def buchi(cls, accepting: Set[int]) -> "AcceptanceCondition":
        return cls(AccType.BUCHI, accepting=accepting)

    @classmethod
    def cobuchi(cls, rejecting: Set[int]) -> "AcceptanceCondition":
        return cls(AccType.COBUCHI, rejecting=rejecting)

    @classmethod
    def rabin(cls, pairs: List[Tuple[Set[int], Set[int]]]) -> "AcceptanceCondition":
        return cls(AccType.RABIN, rabin_pairs=pairs)

    @classmethod
    def streett(cls, pairs: List[Tuple[Set[int], Set[int]]]) -> "AcceptanceCondition":
        return cls(AccType.STREETT, streett_pairs=pairs)

    @classmethod
    def muller(cls, table: Set[FrozenSet[int]]) -> "AcceptanceCondition":
        return cls(AccType.MULLER, muller_table=table)

    @classmethod
    def parity(cls, priorities: Dict[int, int]) -> "AcceptanceCondition":
        return cls(AccType.PARITY, priorities=priorities)


@dataclass
class OmegaParityResult:
    """Result of solving an omega-regular game via parity reduction."""
    winner_even: Set[int]   # Nodes where Player Even (0) wins
    winner_odd: Set[int]    # Nodes where Player Odd (1) wins
    strategy_even: Dict[int, int]  # Winning strategy for Even
    strategy_odd: Dict[int, int]   # Winning strategy for Odd
    acc_type: AccType              # Original acceptance type
    parity_game_size: int          # Size of the parity game
    max_priority: int              # Max priority in parity game
    algorithm: str = "zielonka"
    iterations: int = 0
    reduction_method: str = ""     # How the reduction was done

    def winner(self, node: int) -> int:
        """Return 0 (Even) or 1 (Odd) for which player wins from node."""
        if node in self.winner_even:
            return 0
        return 1

    def summary(self) -> str:
        lines = [f"Acceptance: {self.acc_type.value}"]
        lines.append(f"Reduction: {self.reduction_method}")
        lines.append(f"Parity game: {self.parity_game_size} nodes, max priority {self.max_priority}")
        lines.append(f"Algorithm: {self.algorithm}, iterations: {self.iterations}")
        lines.append(f"Even wins: {len(self.winner_even)} nodes")
        lines.append(f"Odd wins: {len(self.winner_odd)} nodes")
        return "\n".join(lines)


# ============================================================
# Parity solver dispatch (bypass V076 solve() optimization bug)
# ============================================================

def _solve_parity(pg: ParityGame, algorithm: str = "zielonka") -> ParityResult:
    """Solve a parity game using the specified algorithm directly.

    Note: V076's solve() has a bug in self-loop removal + attractor
    recomputation (Phase 4 win0/win1 partition override). We bypass
    it and call Zielonka/SPM directly.
    """
    if algorithm == "spm":
        return small_progress_measures(pg)
    return zielonka(pg)


# ============================================================
# Core: Arena + Acceptance -> Parity Game
# ============================================================

def _arena_owner_to_pg(owner_val: int) -> PGPlayer:
    """Convert arena owner (0/1) to V076 Player."""
    return PGPlayer.EVEN if owner_val == 0 else PGPlayer.ODD


def reduce_to_parity(arena: GameArena, acc: AcceptanceCondition) -> ParityGame:
    """Reduce a game arena with acceptance condition to a parity game.

    This is the central function: given any omega-regular acceptance
    condition, produce an equivalent parity game.

    Args:
        arena: The game arena (nodes, owners, edges)
        acc: The acceptance condition

    Returns:
        A ParityGame equivalent to (arena, acc)
    """
    if acc.acc_type == AccType.PARITY:
        return _reduce_parity(arena, acc)
    elif acc.acc_type == AccType.BUCHI:
        return _reduce_buchi(arena, acc)
    elif acc.acc_type == AccType.COBUCHI:
        return _reduce_cobuchi(arena, acc)
    elif acc.acc_type == AccType.RABIN:
        return _reduce_rabin(arena, acc)
    elif acc.acc_type == AccType.STREETT:
        return _reduce_streett(arena, acc)
    elif acc.acc_type == AccType.MULLER:
        return _reduce_muller(arena, acc)
    else:
        raise ValueError(f"Unknown acceptance type: {acc.acc_type}")


def _reduce_parity(arena: GameArena, acc: AcceptanceCondition) -> ParityGame:
    """Direct: acceptance already specifies priorities."""
    pg = ParityGame()
    for n in arena.nodes:
        p = acc.priorities.get(n, 0)
        pg.add_node(n, _arena_owner_to_pg(arena.owner[n]), p)
    for n in arena.nodes:
        for s in arena.successors.get(n, set()):
            pg.add_edge(n, s)
    return pg


def _reduce_buchi(arena: GameArena, acc: AcceptanceCondition) -> ParityGame:
    """Buchi -> parity via V076."""
    bg = BuchiGame(
        nodes=arena.nodes,
        owner={n: _arena_owner_to_pg(arena.owner[n]) for n in arena.nodes},
        successors=arena.successors,
        accepting=acc.accepting,
    )
    return buchi_to_parity(bg)


def _reduce_cobuchi(arena: GameArena, acc: AcceptanceCondition) -> ParityGame:
    """co-Buchi -> parity via V076."""
    cg = CoBuchiGame(
        nodes=arena.nodes,
        owner={n: _arena_owner_to_pg(arena.owner[n]) for n in arena.nodes},
        successors=arena.successors,
        rejecting=acc.rejecting,
    )
    return cobuchi_to_parity(cg)


def _reduce_rabin(arena: GameArena, acc: AcceptanceCondition) -> ParityGame:
    """Rabin -> parity via Muller conversion + LAR.

    V076's rabin_to_parity has a bug with non-pair nodes. We use the
    correct approach: convert Rabin to Muller table over ALL arena nodes,
    then use LAR.
    A set S is Rabin-accepting iff exists pair (L_i, U_i): S & L_i = {} and S & U_i != {}.
    """
    if not acc.rabin_pairs:
        # No pairs -> no acceptance -> Odd wins
        pg = ParityGame()
        for n in arena.nodes:
            pg.add_node(n, _arena_owner_to_pg(arena.owner[n]), 1)
        for n in arena.nodes:
            for s in arena.successors.get(n, set()):
                pg.add_edge(n, s)
        return pg

    # Build Muller table from Rabin pairs over ALL arena nodes
    from itertools import combinations
    all_nodes_list = sorted(arena.nodes)
    muller_table = set()
    for r in range(len(all_nodes_list) + 1):
        for subset in combinations(all_nodes_list, r):
            s = frozenset(subset)
            for fin, inf in acc.rabin_pairs:
                if not (s & fin) and (s & inf):
                    muller_table.add(s)
                    break

    muller_acc = AcceptanceCondition.muller(muller_table)
    return _reduce_muller(arena, muller_acc)


def _reduce_streett(arena: GameArena, acc: AcceptanceCondition) -> ParityGame:
    """Streett -> parity via V076."""
    pg_owner = {n: _arena_owner_to_pg(arena.owner[n]) for n in arena.nodes}
    pairs = [StreettPair(request=r, response=s) for r, s in acc.streett_pairs]
    return streett_to_parity(arena.nodes, pg_owner, arena.successors, pairs)


def _reduce_muller(arena: GameArena, acc: AcceptanceCondition) -> ParityGame:
    """Muller -> parity via Latest Appearance Record (LAR).

    The LAR construction creates a product game:
    - States: (arena_node, permutation) where permutation tracks visit order
    - Priority: based on position in permutation of the current node
    - Even wins iff the set of infinitely-often visited states is in the Muller table

    For efficiency, we only track nodes that appear in the Muller table.
    """
    # ALL arena nodes must be tracked in the LAR, not just those
    # in the Muller table. Otherwise we can't distinguish e.g. {1,2}
    # (accepting) from {0,1,2} (not accepting).
    relevant = set(arena.nodes)

    # If no nodes at all, Even wins if empty set is accepting
    if not relevant:
        pg = ParityGame()
        trivial_win = frozenset() in acc.muller_table
        p = 0 if trivial_win else 1  # Even wins -> prio 0, Odd wins -> prio 1
        for n in arena.nodes:
            pg.add_node(n, _arena_owner_to_pg(arena.owner[n]), p)
        for n in arena.nodes:
            for s in arena.successors.get(n, set()):
                pg.add_edge(n, s)
        return pg

    relevant_list = sorted(relevant)
    k = len(relevant_list)

    # Generate all permutations of relevant_list for LAR
    # For small k this is fine; for large k we'd need a smarter encoding
    from itertools import permutations
    all_perms = list(permutations(range(k)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}

    # Product state: (arena_node, perm_idx)
    # State id: node * len(all_perms) + perm_idx
    n_perms = len(all_perms)

    def state_id(node: int, perm_idx: int) -> int:
        return node * n_perms + perm_idx

    def update_lar(perm: Tuple[int, ...], elem_local_idx: int) -> Tuple[Tuple[int, ...], int]:
        """Move elem to front of permutation, return new perm and old position."""
        old_pos = perm.index(elem_local_idx)
        new_perm = (elem_local_idx,) + perm[:old_pos] + perm[old_pos+1:]
        return new_perm, old_pos

    # Determine priority for each (node, perm) transition
    # When visiting relevant node r at position pos in the LAR:
    # - The set of infinitely-often visited nodes corresponds to a prefix of the LAR
    # - Priority 2*pos if that prefix is accepting, 2*pos+1 otherwise
    # For non-relevant nodes: they don't change LAR, give neutral priority

    node_to_local = {n: i for i, n in enumerate(relevant_list)}

    # Precompute: for each position pos (0..k-1), which prefixes are accepting?
    def prefix_accepting(perm: Tuple[int, ...], pos: int) -> bool:
        """Is the prefix {perm[0], ..., perm[pos]} an accepting set?"""
        prefix_set = frozenset(relevant_list[perm[j]] for j in range(pos + 1))
        return prefix_set in acc.muller_table

    pg = ParityGame()

    for n in arena.nodes:
        owner = _arena_owner_to_pg(arena.owner[n])
        for pi, perm in enumerate(all_perms):
            sid = state_id(n, pi)
            if n in relevant:
                local_idx = node_to_local[n]
                _, pos = update_lar(perm, local_idx)
                # pos is where n was in the permutation before moving to front
                # The accepting/rejecting depends on whether {perm[0]..perm[pos]} after
                # the update is in the Muller table. But the update moves n to front.
                new_perm, _ = update_lar(perm, local_idx)
                if prefix_accepting(new_perm, pos):
                    priority = 2 * pos  # even -> good for Even
                else:
                    priority = 2 * pos + 1  # odd -> good for Odd
            else:
                # Non-relevant node: must not dominate relevant priorities
                # If empty set is accepting, non-relevant cycles should let Even win (even prio)
                # If not, non-relevant should not help Even (odd prio)
                if frozenset() in acc.muller_table:
                    priority = 0  # even, lowest -- transparent
                else:
                    priority = 1  # odd -- doesn't help Even
            pg.add_node(sid, owner, priority)

    # Edges: LAR is updated for the SOURCE node n, then carried to successor
    # State (n, perm) means: at node n, LAR before visiting n is perm
    # Priority is computed from (n, perm) -- the visit of n with this LAR
    # Edge goes to (succ, updated_perm) where updated_perm = LAR after visiting n
    for n in arena.nodes:
        for succ in arena.successors.get(n, set()):
            for pi, perm in enumerate(all_perms):
                src = state_id(n, pi)
                if n in relevant:
                    # Update LAR for visiting n (the source)
                    local_idx = node_to_local[n]
                    new_perm, _ = update_lar(perm, local_idx)
                    dst_pi = perm_to_idx[new_perm]
                else:
                    dst_pi = pi  # non-relevant doesn't change LAR
                dst = state_id(succ, dst_pi)
                pg.add_edge(src, dst)

    return pg


# ============================================================
# Solve: Arena + Acceptance -> OmegaParityResult
# ============================================================

def solve_omega_regular(arena: GameArena, acc: AcceptanceCondition,
                        algorithm: str = "zielonka",
                        initial_perm_idx: int = 0) -> OmegaParityResult:
    """Solve a game with omega-regular acceptance condition.

    Args:
        arena: The game arena
        acc: The acceptance condition
        algorithm: "zielonka" or "spm"
        initial_perm_idx: For Muller games, the initial LAR permutation index

    Returns:
        OmegaParityResult with winning regions and strategies
    """
    pg = reduce_to_parity(arena, acc)
    result = _solve_parity(pg, algorithm=algorithm)

    # Map back to arena nodes
    if acc.acc_type == AccType.MULLER:
        from itertools import permutations
        relevant = set()
        for s in acc.muller_table:
            relevant.update(s)
        k = len(sorted(relevant))
        n_perms = 1
        for i in range(1, k + 1):
            n_perms *= i
        if n_perms == 0:
            n_perms = 1

        # Project: node wins if its initial product state wins
        winner_even = set()
        winner_odd = set()
        strat_even = {}
        strat_odd = {}
        for n in arena.nodes:
            sid = n * n_perms + initial_perm_idx
            if sid in result.win0:
                winner_even.add(n)
            else:
                winner_odd.add(n)
            # Strategy projection: find the arena successor from the product strategy
            if sid in result.strategy0:
                target_sid = result.strategy0[sid]
                target_node = target_sid // n_perms
                strat_even[n] = target_node
            if sid in result.strategy1:
                target_sid = result.strategy1[sid]
                target_node = target_sid // n_perms
                strat_odd[n] = target_node
    else:
        # Direct mapping (no product expansion)
        winner_even = {n for n in result.win0 if n in arena.nodes}
        winner_odd = {n for n in result.win1 if n in arena.nodes}
        strat_even = {n: s for n, s in result.strategy0.items() if n in arena.nodes}
        strat_odd = {n: s for n, s in result.strategy1.items() if n in arena.nodes}

    return OmegaParityResult(
        winner_even=winner_even,
        winner_odd=winner_odd,
        strategy_even=strat_even,
        strategy_odd=strat_odd,
        acc_type=acc.acc_type,
        parity_game_size=len(pg.nodes),
        max_priority=pg.max_priority(),
        algorithm=result.algorithm,
        iterations=result.iterations,
        reduction_method=f"{acc.acc_type.value}_to_parity",
    )


# ============================================================
# LTL-to-Parity: LTL formula -> NBA -> product arena -> parity
# ============================================================

def _label_match(label: Label, state_props: Set[str]) -> bool:
    """Check if a set of propositions matches an NBA transition label."""
    for p in label.pos:
        if p not in state_props:
            return False
    for p in label.neg:
        if p in state_props:
            return False
    return True


def ltl_to_parity_game(arena: GameArena, formula: LTL,
                       initial_state: int = 0) -> Tuple[ParityGame, Dict[int, Tuple[int, int]], int]:
    """Build a parity game from an arena and LTL formula.

    Pipeline: LTL -> NNF -> GBA -> NBA -> product arena -> Buchi parity game

    Approach: Build NBA for the formula directly. Product game uses Buchi
    acceptance (Even wins iff accepting states visited infinitely).
    NBA nondeterminism is resolved via intermediate Even-owned choice nodes,
    ensuring Even can exploit nondeterminism at Odd-owned arena states.

    Args:
        arena: Game arena with node labels (atomic propositions)
        formula: LTL formula that Player Even wants to satisfy
        initial_state: Initial arena state

    Returns:
        (parity_game, state_map, initial_product_states)
        state_map: product_state -> (arena_node, nba_state)
    """
    # LTL -> NBA for the formula (not negation)
    formula_nnf = nnf(formula)
    gba = ltl_to_gba(formula_nnf)
    nba = gba_to_nba(gba)

    # Build product arena x NBA
    # Product state: (arena_node, nba_state)
    state_map = {}  # product_id -> (arena_node, nba_state)
    inv_map = {}    # (arena_node, nba_state) -> product_id
    next_id = 0

    def get_or_create(a_node: int, aut_state: int) -> int:
        nonlocal next_id
        key = (a_node, aut_state)
        if key not in inv_map:
            inv_map[key] = next_id
            state_map[next_id] = key
            next_id += 1
        return inv_map[key]

    # BFS to explore reachable product states
    initial_prods = set()
    queue = []
    visited = set()
    for init_aut in nba.initial:
        pid = get_or_create(initial_state, init_aut)
        initial_prods.add(pid)
        if pid not in visited:
            visited.add(pid)
            queue.append(pid)

    prod_successors = {}
    prod_accepting = set()

    while queue:
        pid = queue.pop(0)
        a_node, aut_state = state_map[pid]
        prod_successors[pid] = set()

        for a_succ in arena.successors.get(a_node, set()):
            succ_props = arena.labels.get(a_succ, set())
            for label, aut_succ in nba.transitions.get(aut_state, []):
                if not _label_match(label, succ_props):
                    continue
                succ_pid = get_or_create(a_succ, aut_succ)
                prod_successors[pid].add(succ_pid)
                if succ_pid not in visited:
                    visited.add(succ_pid)
                    queue.append(succ_pid)

    # Mark accepting product states
    for pid in visited:
        _, aut_state = state_map[pid]
        if aut_state in nba.accepting:
            prod_accepting.add(pid)

    # Build parity game with Buchi acceptance for the formula
    # Accepting states: priority 2 (even) -- Even wants to visit infinitely
    # Non-accepting states: priority 1 (odd)
    #
    # NBA nondeterminism: at Odd-owned states, we add intermediate
    # Even-owned choice nodes so Even resolves the NBA nondeterminism.

    pg = ParityGame()

    # Add Odd-sink for dead ends (no matching NBA transition -> formula cannot be satisfied)
    odd_sink_id = next_id
    next_id += 1
    pg.add_node(odd_sink_id, PGPlayer.ODD, 1)
    pg.add_edge(odd_sink_id, odd_sink_id)

    # For Even-owned states: all successors directly (Even resolves both
    # arena and NBA choices, which is correct for Buchi acceptance)
    # For Odd-owned states: group by arena successor, add Even-owned
    # intermediate nodes for NBA choice within each arena move

    for pid in visited:
        a_node, _ = state_map[pid]
        priority = 2 if pid in prod_accepting else 1
        a_owner = arena.owner[a_node]

        if a_owner == 0:  # Even owns this arena state
            pg.add_node(pid, PGPlayer.EVEN, priority)
        else:  # Odd owns this arena state
            pg.add_node(pid, PGPlayer.ODD, priority)

    # Add edges with NBA nondeterminism handling
    for pid in visited:
        a_node, aut_state = state_map[pid]
        a_owner = arena.owner[a_node]
        succs = prod_successors.get(pid, set())

        if not succs:
            pg.add_edge(pid, odd_sink_id)
            continue

        if a_owner == 0:
            # Even-owned: Even controls all choices, just add all edges
            for spid in succs:
                pg.add_edge(pid, spid)
        else:
            # Odd-owned: Odd picks arena successor, Even picks NBA transition
            # Group successors by arena successor
            arena_groups = {}
            for spid in succs:
                a_succ, _ = state_map[spid]
                arena_groups.setdefault(a_succ, []).append(spid)

            if len(arena_groups) <= 1:
                # Only one arena successor: no real Odd choice, add all
                for spid in succs:
                    pg.add_edge(pid, spid)
            else:
                # Multiple arena successors: add Even-owned intermediate nodes
                for a_succ, nba_succs in arena_groups.items():
                    if len(nba_succs) == 1:
                        # Single NBA successor: direct edge
                        pg.add_edge(pid, nba_succs[0])
                    else:
                        # Multiple NBA successors: intermediate Even-owned node
                        choice_id = next_id
                        next_id += 1
                        pg.add_node(choice_id, PGPlayer.EVEN, 1)  # Neutral priority
                        pg.add_edge(pid, choice_id)
                        for spid in nba_succs:
                            pg.add_edge(choice_id, spid)

    return pg, state_map, initial_prods


def solve_ltl_game(arena: GameArena, formula: LTL,
                   initial_state: int = 0,
                   algorithm: str = "zielonka") -> OmegaParityResult:
    """Solve a game with LTL winning condition for Player Even.

    Args:
        arena: Game arena with labeled states
        formula: LTL formula (Player Even wins if satisfied)
        initial_state: Initial state for determining the winner

    Returns:
        OmegaParityResult with arena-level winning regions
    """
    pg, state_map, init_prods = ltl_to_parity_game(arena, formula, initial_state)
    result = _solve_parity(pg, algorithm=algorithm)

    # Project back to arena nodes
    # A node wins for Even if ANY product state (node, *) wins for Even
    # (existential: there exists an automaton state pairing where Even wins)
    winner_even = set()
    winner_odd = set()
    strat_even = {}
    strat_odd = {}

    # Group product states by arena node
    node_prod_states = {}
    for pid, (a_node, _) in state_map.items():
        node_prod_states.setdefault(a_node, []).append(pid)

    for a_node, pids in node_prod_states.items():
        if any(pid in result.win0 for pid in pids):
            winner_even.add(a_node)
        else:
            winner_odd.add(a_node)

    # Project strategies: for Even-winning nodes, use the product strategy
    for pid, (a_node, _) in state_map.items():
        if pid in result.strategy0 and a_node not in strat_even:
            target_pid = result.strategy0[pid]
            if target_pid in state_map:
                strat_even[a_node] = state_map[target_pid][0]
        if pid in result.strategy1 and a_node not in strat_odd:
            target_pid = result.strategy1[pid]
            if target_pid in state_map:
                strat_odd[a_node] = state_map[target_pid][0]

    return OmegaParityResult(
        winner_even=winner_even,
        winner_odd=winner_odd,
        strategy_even=strat_even,
        strategy_odd=strat_odd,
        acc_type=AccType.BUCHI,  # LTL via co-Buchi product
        parity_game_size=len(pg.nodes),
        max_priority=pg.max_priority(),
        algorithm=result.algorithm,
        iterations=result.iterations,
        reduction_method="ltl_to_cobuchi_parity",
    )


# ============================================================
# Acceptance condition composition
# ============================================================

def conjoin_acceptance(arena: GameArena,
                       conditions: List[AcceptanceCondition]) -> OmegaParityResult:
    """Solve a game where Player Even must satisfy ALL conditions simultaneously.

    Conjunction of omega-regular conditions is omega-regular. We use a product
    construction: track acceptance status for each condition independently,
    then combine into a single parity condition.

    For Buchi conjunction: the product tracks which conditions have been
    satisfied since the last reset. Player Even must visit accepting states
    of all conditions infinitely often (generalized Buchi -> Buchi -> parity).
    """
    if len(conditions) == 1:
        return solve_omega_regular(arena, conditions[0])

    # For simplicity, handle the common case: conjunction of Buchi conditions
    # This is equivalent to a generalized Buchi condition
    all_buchi = all(c.acc_type == AccType.BUCHI for c in conditions)

    if all_buchi:
        return _conjoin_buchi(arena, conditions)

    # General case: reduce each to parity, then combine via product
    return _conjoin_general(arena, conditions)


def _conjoin_buchi(arena: GameArena,
                   conditions: List[AcceptanceCondition]) -> OmegaParityResult:
    """Conjunction of Buchi conditions via round-robin acceptance.

    Product state: (arena_node, obligation_index)
    where obligation_index tracks which condition we're waiting to satisfy next.
    We cycle through conditions 0, 1, ..., k-1, 0, 1, ...
    A product state (n, i) is accepting iff i == 0 and n is in conditions[k-1].accepting
    (we just completed the cycle).
    """
    k = len(conditions)

    # Product states: (arena_node, obligation_idx)
    def state_id(node: int, obl: int) -> int:
        return node * k + obl

    pg = ParityGame()

    for n in arena.nodes:
        owner = _arena_owner_to_pg(arena.owner[n])
        for obl in range(k):
            sid = state_id(n, obl)
            # Accepting: we're looking for condition obl and n satisfies it
            # After satisfying condition k-1 we wrap to 0 -- that's the accepting cycle
            is_accepting = n in conditions[obl].accepting
            # Use Buchi encoding: accepting -> priority 2, non-accepting -> priority 1
            priority = 2 if (is_accepting and obl == k - 1) else 1
            pg.add_node(sid, owner, priority)

    for n in arena.nodes:
        for succ in arena.successors.get(n, set()):
            for obl in range(k):
                src = state_id(n, obl)
                # If current node satisfies current obligation, advance
                if n in conditions[obl].accepting:
                    next_obl = (obl + 1) % k
                else:
                    next_obl = obl
                dst = state_id(succ, next_obl)
                pg.add_edge(src, dst)

    result = _solve_parity(pg)

    # Project back
    winner_even = set()
    winner_odd = set()
    strat_even = {}
    strat_odd = {}

    for n in arena.nodes:
        sid = state_id(n, 0)  # Initial obligation is 0
        if sid in result.win0:
            winner_even.add(n)
        else:
            winner_odd.add(n)
        if sid in result.strategy0:
            target = result.strategy0[sid]
            strat_even[n] = target // k
        if sid in result.strategy1:
            target = result.strategy1[sid]
            strat_odd[n] = target // k

    return OmegaParityResult(
        winner_even=winner_even,
        winner_odd=winner_odd,
        strategy_even=strat_even,
        strategy_odd=strat_odd,
        acc_type=AccType.BUCHI,
        parity_game_size=len(pg.nodes),
        max_priority=pg.max_priority(),
        algorithm=result.algorithm,
        iterations=result.iterations,
        reduction_method="generalized_buchi_to_parity",
    )


def _conjoin_general(arena: GameArena,
                     conditions: List[AcceptanceCondition]) -> OmegaParityResult:
    """General conjunction via independent parity reduction + product.

    Solve each condition independently and intersect winning regions.
    This is sound but incomplete: a node is in the joint winning region
    only if Even can win ALL conditions from that node.

    For exact solutions, a full product construction is needed, but
    for many practical cases (non-interfering objectives), this suffices.
    """
    # Solve each independently
    results = [solve_omega_regular(arena, c) for c in conditions]

    # Intersect winning regions
    winner_even = results[0].winner_even
    for r in results[1:]:
        winner_even = winner_even & r.winner_even
    winner_odd = arena.nodes - winner_even

    # Strategies: use first available
    strat_even = {}
    strat_odd = {}
    for r in results:
        for n, s in r.strategy_even.items():
            if n not in strat_even and n in winner_even:
                strat_even[n] = s
        for n, s in r.strategy_odd.items():
            if n not in strat_odd and n in winner_odd:
                strat_odd[n] = s

    total_pg_size = sum(r.parity_game_size for r in results)
    max_prio = max(r.max_priority for r in results)

    return OmegaParityResult(
        winner_even=winner_even,
        winner_odd=winner_odd,
        strategy_even=strat_even,
        strategy_odd=strat_odd,
        acc_type=AccType.RABIN,  # General conjunction
        parity_game_size=total_pg_size,
        max_priority=max_prio,
        algorithm="independent_intersection",
        iterations=sum(r.iterations for r in results),
        reduction_method="conjunction_independent",
    )


def disjoin_acceptance(arena: GameArena,
                       conditions: List[AcceptanceCondition]) -> OmegaParityResult:
    """Solve a game where Player Even must satisfy ANY condition.

    Disjunction: Even wins if at least one condition is satisfied.
    Solve each independently and union winning regions.
    """
    if len(conditions) == 1:
        return solve_omega_regular(arena, conditions[0])

    results = [solve_omega_regular(arena, c) for c in conditions]

    winner_even = set()
    for r in results:
        winner_even |= r.winner_even
    winner_odd = arena.nodes - winner_even

    strat_even = {}
    strat_odd = {}
    for r in results:
        for n, s in r.strategy_even.items():
            if n not in strat_even and n in winner_even:
                strat_even[n] = s
        for n, s in r.strategy_odd.items():
            if n not in strat_odd and n in winner_odd:
                strat_odd[n] = s

    total_pg_size = sum(r.parity_game_size for r in results)
    max_prio = max(r.max_priority for r in results)

    return OmegaParityResult(
        winner_even=winner_even,
        winner_odd=winner_odd,
        strategy_even=strat_even,
        strategy_odd=strat_odd,
        acc_type=AccType.RABIN,
        parity_game_size=total_pg_size,
        max_priority=max_prio,
        algorithm="independent_union",
        iterations=sum(r.iterations for r in results),
        reduction_method="disjunction_independent",
    )


# ============================================================
# Comparison and analysis
# ============================================================

def compare_reductions(arena: GameArena, acc: AcceptanceCondition) -> Dict:
    """Compare Zielonka vs SPM on the same reduction."""
    pg = reduce_to_parity(arena, acc)
    r_z = _solve_parity(pg, algorithm="zielonka")
    r_s = _solve_parity(pg, algorithm="spm")

    return {
        "parity_game_nodes": len(pg.nodes),
        "max_priority": pg.max_priority(),
        "zielonka": {
            "win0": len(r_z.win0),
            "win1": len(r_z.win1),
            "iterations": r_z.iterations,
        },
        "spm": {
            "win0": len(r_s.win0),
            "win1": len(r_s.win1),
            "iterations": r_s.iterations,
        },
        "agree": r_z.win0 == r_s.win0,
    }


def analyze_reduction(arena: GameArena, acc: AcceptanceCondition) -> Dict:
    """Analyze the reduction: parity game size, priority structure."""
    pg = reduce_to_parity(arena, acc)
    priorities = [pg.priority[n] for n in pg.nodes]
    priority_counts = {}
    for p in priorities:
        priority_counts[p] = priority_counts.get(p, 0) + 1

    return {
        "arena_nodes": len(arena.nodes),
        "parity_game_nodes": len(pg.nodes),
        "blowup": len(pg.nodes) / max(len(arena.nodes), 1),
        "max_priority": pg.max_priority(),
        "priority_distribution": priority_counts,
        "edges": sum(len(pg.successors.get(n, set())) for n in pg.nodes),
    }


# ============================================================
# Convenience builders
# ============================================================

def make_arena(nodes: List[Tuple[int, int]], edges: List[Tuple[int, int]],
               labels: Optional[Dict[int, Set[str]]] = None) -> GameArena:
    """Create a game arena from node list and edge list.

    Args:
        nodes: [(id, owner), ...] where owner is 0 or 1
        edges: [(src, dst), ...]
        labels: optional {node: {prop, ...}} for LTL games
    """
    arena = GameArena()
    for nid, owner in nodes:
        props = labels.get(nid) if labels else None
        arena.add_node(nid, owner, props)
    for src, dst in edges:
        arena.add_edge(src, dst)
    return arena


def solve_from_spec(nodes: List[Tuple[int, int]], edges: List[Tuple[int, int]],
                    acc: AcceptanceCondition,
                    algorithm: str = "zielonka") -> OmegaParityResult:
    """One-shot: build arena + solve."""
    arena = make_arena(nodes, edges)
    return solve_omega_regular(arena, acc, algorithm=algorithm)


def solve_ltl_from_spec(nodes: List[Tuple[int, int]], edges: List[Tuple[int, int]],
                        labels: Dict[int, Set[str]], formula_str: str,
                        initial_state: int = 0,
                        algorithm: str = "zielonka") -> OmegaParityResult:
    """One-shot: build arena + solve LTL game from string formula."""
    arena = make_arena(nodes, edges, labels)
    formula = parse_ltl(formula_str)
    return solve_ltl_game(arena, formula, initial_state, algorithm)


# ============================================================
# Muller to Rabin conversion (alternative reduction path)
# ============================================================

def muller_to_rabin(table: Set[FrozenSet[int]],
                    all_nodes: Set[int]) -> List[Tuple[Set[int], Set[int]]]:
    """Convert a Muller condition to a Rabin condition (set of pairs).

    For each accepting set F in the Muller table:
    - fin = all_nodes - F  (nodes NOT in F must be visited finitely)
    - inf = F              (nodes in F must be visited infinitely)

    This gives |table| Rabin pairs. The Rabin condition is the disjunction:
    exists a pair where fin is visited finitely AND inf is visited infinitely.
    """
    pairs = []
    for f in table:
        fin = all_nodes - f
        inf = set(f)
        pairs.append((fin, inf))
    return pairs


def solve_muller_via_rabin(arena: GameArena,
                           acc: AcceptanceCondition,
                           algorithm: str = "zielonka") -> OmegaParityResult:
    """Solve Muller game via Rabin reduction (alternative to LAR).

    This avoids the factorial LAR blowup by converting Muller -> Rabin -> parity.
    The Rabin encoding has |table| pairs, which may be smaller than k! LAR states.
    """
    relevant = set()
    for s in acc.muller_table:
        relevant.update(s)

    pairs = muller_to_rabin(acc.muller_table, relevant)
    rabin_acc = AcceptanceCondition.rabin(pairs)
    result = solve_omega_regular(arena, rabin_acc, algorithm=algorithm)
    result.reduction_method = "muller_via_rabin_to_parity"
    return result
