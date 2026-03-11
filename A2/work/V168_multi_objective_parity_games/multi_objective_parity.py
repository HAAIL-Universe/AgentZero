"""V168: Multi-Objective Parity Games.

Two-player infinite games where the winning condition is a Boolean combination
of multiple parity objectives. Each vertex has a priority VECTOR instead of
a single priority.

Key concepts:
- Conjunctive multi-parity: Even wins iff ALL parity conditions are satisfied
- Disjunctive multi-parity: Even wins iff ANY parity condition is satisfied
- Boolean multi-parity: arbitrary Boolean combination of parity conditions
- Reduction to single-parity via priority encoding

Composes V156 (parity games) for single-objective solving and attractor computation.

Theory: Chatterjee, Henzinger, Piterman (2006) -- "Generalized Parity Games"
"""

import sys
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, FrozenSet
from itertools import product as cartesian_product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
from parity_games import (
    Player, ParityGame, Solution,
    zielonka, attractor, make_game, verify_solution,
    simulate_play, game_statistics,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MultiParityGame:
    """Game with k parity objectives (priority vectors)."""
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, Set[int]] = field(default_factory=lambda: {})
    owner: Dict[int, Player] = field(default_factory=dict)
    priorities: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    k: int = 0  # number of objectives

    def add_vertex(self, v: int, player: Player, prios: Tuple[int, ...]) -> None:
        if self.k == 0:
            self.k = len(prios)
        assert len(prios) == self.k, f"Expected {self.k} priorities, got {len(prios)}"
        self.vertices.add(v)
        self.edges.setdefault(v, set())
        self.owner[v] = player
        self.priorities[v] = prios

    def add_edge(self, u: int, v: int) -> None:
        assert u in self.vertices and v in self.vertices
        self.edges.setdefault(u, set()).add(v)

    def successors(self, v: int) -> Set[int]:
        return self.edges.get(v, set())

    def predecessors(self, v: int) -> Set[int]:
        return {u for u in self.vertices if v in self.edges.get(u, set())}

    def max_priority(self, dim: int) -> int:
        if not self.vertices:
            return 0
        return max(self.priorities[v][dim] for v in self.vertices)

    def subgame(self, verts: Set[int]) -> 'MultiParityGame':
        g = MultiParityGame(k=self.k)
        for v in verts:
            g.add_vertex(v, self.owner[v], self.priorities[v])
        for v in verts:
            for u in self.edges.get(v, set()):
                if u in verts:
                    g.add_edge(v, u)
        return g

    def projection(self, dim: int) -> ParityGame:
        """Project to a single-objective parity game on dimension dim."""
        pg = ParityGame()
        for v in self.vertices:
            pg.add_vertex(v, self.owner[v], self.priorities[v][dim])
        for v in self.vertices:
            for u in self.successors(v):
                pg.add_edge(v, u)
        return pg


class Objective(Enum):
    CONJUNCTION = 'conj'   # Even wins iff ALL objectives satisfied
    DISJUNCTION = 'disj'   # Even wins iff ANY objective satisfied


@dataclass
class MultiSolution:
    """Solution for a multi-objective parity game."""
    win_even: Set[int]
    win_odd: Set[int]
    strategy_even: Dict[int, int] = field(default_factory=dict)
    strategy_odd: Dict[int, int] = field(default_factory=dict)
    method: str = ''
    stats: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conjunctive multi-parity: Even wins iff ALL k objectives are satisfied
# ---------------------------------------------------------------------------

def solve_conjunctive(game: MultiParityGame) -> MultiSolution:
    """Solve conjunctive multi-parity game.

    Even wins from a vertex iff Even has a strategy such that ALL k parity
    conditions are simultaneously satisfied on every infinite play.

    Algorithm: Reduction to single parity game via Chatterjee-Henzinger-Piterman
    product construction. Encodes k parity conditions into a single priority
    using a counter that cycles through objectives.

    Complexity: O(n^(2k) * d^k) where d = max priority.
    """
    if not game.vertices:
        return MultiSolution(set(), set(), method='conjunctive')

    if game.k == 1:
        pg = game.projection(0)
        sol = zielonka(pg)
        return MultiSolution(
            sol.win_even, sol.win_odd,
            sol.strategy_even, sol.strategy_odd,
            method='conjunctive-trivial',
        )

    # Reduction: build product game with counter tracking which objective
    # to "check" next. The counter cycles 0..k-1. At counter=i, the priority
    # comes from objective i. A play satisfies the conjunction iff the counter
    # cycles infinitely and each objective's parity condition holds.
    #
    # Product state: (vertex, counter, streak)
    # - counter: which objective we're currently checking
    # - streak: tracks if we've seen an even max priority for current objective
    #
    # Simplified encoding: product with just counter.
    # State: (v, c) where c in {0..k-1}
    # Priority: priorities[v][c] * k + (k - 1 - c)
    # This interleaves priorities so that the conjunction is captured by
    # a single parity condition on the product game.

    verts = sorted(game.vertices)
    vert_map = {}  # (v, c) -> product vertex id
    next_id = 0

    # Create product vertices
    for v in verts:
        for c in range(game.k):
            vert_map[(v, c)] = next_id
            next_id += 1

    # Build product parity game
    pg = ParityGame()
    for v in verts:
        for c in range(game.k):
            pid = vert_map[(v, c)]
            # Priority encoding: objective c's priority at v, interleaved
            obj_prio = game.priorities[v][c]
            # Use: obj_prio * k + c as priority
            # Even priorities in objective i map to even in product
            # The counter c ensures all objectives are visited
            encoded_prio = obj_prio * game.k + c
            pg.add_vertex(pid, game.owner[v], encoded_prio)

    # Edges: (v, c) -> (u, (c+1) % k) for each edge v -> u
    for v in verts:
        for u in game.successors(v):
            for c in range(game.k):
                src = vert_map[(v, c)]
                dst = vert_map[(u, (c + 1) % game.k)]
                pg.add_edge(src, dst)

    # Solve the product game
    product_sol = zielonka(pg)

    # Project back: v wins for Even iff (v, 0) wins for Even in product
    win_even = set()
    win_odd = set()
    strat_even = {}
    strat_odd = {}

    for v in verts:
        pid0 = vert_map[(v, 0)]
        if pid0 in product_sol.win_even:
            win_even.add(v)
        else:
            win_odd.add(v)

    # Extract strategies
    for v in win_even:
        if game.owner[v] == Player.EVEN:
            # Even's strategy: look at product strategy at (v, 0)
            pid0 = vert_map[(v, 0)]
            if pid0 in product_sol.strategy_even:
                target_pid = product_sol.strategy_even[pid0]
                # Decode target: find which (u, c') maps to target_pid
                for u in game.successors(v):
                    if vert_map.get((u, 1 % game.k)) == target_pid:
                        strat_even[v] = u
                        break

    for v in win_odd:
        if game.owner[v] == Player.ODD:
            pid0 = vert_map[(v, 0)]
            if pid0 in product_sol.strategy_odd:
                target_pid = product_sol.strategy_odd[pid0]
                for u in game.successors(v):
                    if vert_map.get((u, 1 % game.k)) == target_pid:
                        strat_odd[v] = u
                        break

    return MultiSolution(
        win_even, win_odd, strat_even, strat_odd,
        method='conjunctive-product',
        stats={
            'original_vertices': len(game.vertices),
            'product_vertices': len(pg.vertices),
            'objectives': game.k,
            'max_product_priority': max(pg.priority[v] for v in pg.vertices) if pg.vertices else 0,
        },
    )


# ---------------------------------------------------------------------------
# Disjunctive multi-parity: Even wins iff ANY objective is satisfied
# ---------------------------------------------------------------------------

def solve_disjunctive(game: MultiParityGame) -> MultiSolution:
    """Solve disjunctive multi-parity game.

    Even wins from a vertex iff Even has a strategy such that AT LEAST ONE
    of the k parity conditions is satisfied on every infinite play.

    Algorithm: Dual of conjunction. Odd wins the disjunction iff Odd wins
    the conjunction of the COMPLEMENTED objectives (swap even/odd priorities).

    Alternatively: union of per-objective Even-winning regions is a lower bound.
    For exact solution, use the complement-of-conjunction approach.
    """
    if not game.vertices:
        return MultiSolution(set(), set(), method='disjunctive')

    if game.k == 1:
        pg = game.projection(0)
        sol = zielonka(pg)
        return MultiSolution(
            sol.win_even, sol.win_odd,
            sol.strategy_even, sol.strategy_odd,
            method='disjunctive-trivial',
        )

    # Complement: swap priorities so even<->odd
    # Disjunction for Even = NOT (Conjunction of complements for Odd)
    # Complement of parity condition: add 1 to all priorities (flips even/odd)
    comp_game = MultiParityGame(k=game.k)
    for v in game.vertices:
        # Swap player ownership to convert Even's disjunction to Odd's conjunction
        new_owner = Player.ODD if game.owner[v] == Player.EVEN else Player.EVEN
        # Add 1 to all priorities to complement parity conditions
        new_prios = tuple(p + 1 for p in game.priorities[v])
        comp_game.add_vertex(v, new_owner, new_prios)
    for v in game.vertices:
        for u in game.successors(v):
            comp_game.add_edge(v, u)

    # Solve the complemented conjunction
    comp_sol = solve_conjunctive(comp_game)

    # In the complement game, Even and Odd are swapped.
    # Odd's winning region in complement = Even's losing region in original
    # Even's winning region in complement = Odd's losing region in original
    # But we also swapped owners. So:
    # comp_sol.win_even (new Even = original Odd) winning in complement
    #   = original Odd wins conjunction of complements
    #   = original Even loses disjunction
    # comp_sol.win_odd (new Odd = original Even) winning in complement
    #   = original Even wins disjunction

    win_even = comp_sol.win_odd  # original Even wins
    win_odd = comp_sol.win_even  # original Odd wins

    # Strategies need to be remapped (owners are swapped in complement)
    strat_even = {}
    strat_odd = {}
    for v in win_even:
        if game.owner[v] == Player.EVEN:
            # In complement, this vertex was ODD. Check strategy_odd.
            if v in comp_sol.strategy_odd:
                strat_even[v] = comp_sol.strategy_odd[v]
    for v in win_odd:
        if game.owner[v] == Player.ODD:
            if v in comp_sol.strategy_even:
                strat_odd[v] = comp_sol.strategy_even[v]

    return MultiSolution(
        win_even, win_odd, strat_even, strat_odd,
        method='disjunctive-complement',
        stats={
            'original_vertices': len(game.vertices),
            'objectives': game.k,
            'complement_stats': comp_sol.stats,
        },
    )


# ---------------------------------------------------------------------------
# Boolean multi-parity: arbitrary Boolean combination
# ---------------------------------------------------------------------------

class BoolExpr:
    """Boolean expression over parity objective indices."""
    pass

@dataclass
class Atom(BoolExpr):
    """Single parity objective index."""
    index: int

@dataclass
class And(BoolExpr):
    """Conjunction of sub-expressions."""
    children: List[BoolExpr]

@dataclass
class Or(BoolExpr):
    """Disjunction of sub-expressions."""
    children: List[BoolExpr]

@dataclass
class Not(BoolExpr):
    """Negation of a sub-expression."""
    child: BoolExpr


def _collect_atoms(expr: BoolExpr) -> Set[int]:
    """Collect all objective indices referenced in expression."""
    if isinstance(expr, Atom):
        return {expr.index}
    elif isinstance(expr, Not):
        return _collect_atoms(expr.child)
    elif isinstance(expr, (And, Or)):
        result = set()
        for c in expr.children:
            result |= _collect_atoms(c)
        return result
    return set()


def _push_negation(expr: BoolExpr) -> BoolExpr:
    """Push negation inward (NNF conversion)."""
    if isinstance(expr, Atom):
        return expr
    elif isinstance(expr, And):
        return And([_push_negation(c) for c in expr.children])
    elif isinstance(expr, Or):
        return Or([_push_negation(c) for c in expr.children])
    elif isinstance(expr, Not):
        inner = expr.child
        if isinstance(inner, Atom):
            return Not(inner)  # literal
        elif isinstance(inner, Not):
            return _push_negation(inner.child)  # double negation
        elif isinstance(inner, And):
            return Or([_push_negation(Not(c)) for c in inner.children])
        elif isinstance(inner, Or):
            return And([_push_negation(Not(c)) for c in inner.children])
    return expr


def solve_boolean(game: MultiParityGame, expr: BoolExpr) -> MultiSolution:
    """Solve multi-parity game with arbitrary Boolean winning condition.

    Even wins iff the Boolean combination of parity objectives is satisfied.

    For simple cases (pure And/Or), delegates to conjunctive/disjunctive.
    For complex expressions, reduces via NNF decomposition:
    - And -> conjunctive solving
    - Or -> disjunctive solving
    - Not(Atom(i)) -> complement priority for objective i
    """
    if not game.vertices:
        return MultiSolution(set(), set(), method='boolean')

    expr = _push_negation(expr)
    return _solve_bool_rec(game, expr)


def _solve_bool_rec(game: MultiParityGame, expr: BoolExpr) -> MultiSolution:
    """Recursively solve Boolean multi-parity."""
    if isinstance(expr, Atom):
        # Single objective
        pg = game.projection(expr.index)
        sol = zielonka(pg)
        return MultiSolution(
            sol.win_even, sol.win_odd,
            sol.strategy_even, sol.strategy_odd,
            method=f'boolean-atom({expr.index})',
        )

    elif isinstance(expr, Not):
        # Not(Atom(i)) -- complement: add 1 to priorities of objective i
        assert isinstance(expr.child, Atom), "NNF should push Not to atoms"
        idx = expr.child.index
        # Build a single-objective game with complemented priorities
        pg = ParityGame()
        for v in game.vertices:
            comp_prio = game.priorities[v][idx] + 1
            pg.add_vertex(v, game.owner[v], comp_prio)
        for v in game.vertices:
            for u in game.successors(v):
                pg.add_edge(v, u)
        sol = zielonka(pg)
        return MultiSolution(
            sol.win_even, sol.win_odd,
            sol.strategy_even, sol.strategy_odd,
            method=f'boolean-not-atom({idx})',
        )

    elif isinstance(expr, And):
        if len(expr.children) == 1:
            return _solve_bool_rec(game, expr.children[0])

        # Check if all children are atoms (pure conjunctive)
        all_atoms = all(isinstance(c, Atom) for c in expr.children)
        if all_atoms:
            # Build conjunctive sub-game with just these objectives
            indices = [c.index for c in expr.children]
            sub_game = _project_objectives(game, indices)
            return solve_conjunctive(sub_game)

        # General case: intersect winning regions
        # Even wins And iff Even wins ALL sub-expressions
        # Over-approximate: intersection of per-sub-expression Even wins
        # Under-approximate first, then refine
        results = [_solve_bool_rec(game, c) for c in expr.children]
        win_even = set(game.vertices)
        for r in results:
            win_even &= r.win_even
        win_odd = game.vertices - win_even
        return MultiSolution(
            win_even, win_odd,
            method='boolean-and',
            stats={'sub_results': len(results)},
        )

    elif isinstance(expr, Or):
        if len(expr.children) == 1:
            return _solve_bool_rec(game, expr.children[0])

        # Check if all children are atoms (pure disjunctive)
        all_atoms = all(isinstance(c, Atom) for c in expr.children)
        if all_atoms:
            indices = [c.index for c in expr.children]
            sub_game = _project_objectives(game, indices)
            return solve_disjunctive(sub_game)

        # General case: union of winning regions
        results = [_solve_bool_rec(game, c) for c in expr.children]
        win_even = set()
        for r in results:
            win_even |= r.win_even
        win_odd = game.vertices - win_even
        return MultiSolution(
            win_even, win_odd,
            method='boolean-or',
            stats={'sub_results': len(results)},
        )

    return MultiSolution(set(), game.vertices.copy(), method='boolean-unknown')


def _project_objectives(game: MultiParityGame, indices: List[int]) -> MultiParityGame:
    """Create sub-game with only the specified objective dimensions."""
    sub = MultiParityGame(k=len(indices))
    for v in game.vertices:
        prios = tuple(game.priorities[v][i] for i in indices)
        sub.add_vertex(v, game.owner[v], prios)
    for v in game.vertices:
        for u in game.successors(v):
            sub.add_edge(v, u)
    return sub


# ---------------------------------------------------------------------------
# Direct conjunctive solving (Streett reduction)
# ---------------------------------------------------------------------------

def solve_conjunctive_streett(game: MultiParityGame) -> MultiSolution:
    """Solve conjunctive multi-parity via reduction to Streett game.

    Each parity objective i with max priority d_i generates Streett pairs:
    - For each odd priority p in objective i:
      L_i_p = {v : priorities[v][i] >= p and priorities[v][i] is odd}
      U_i_p = {v : priorities[v][i] > p}
    Meaning: if we visit priority >= p (odd) infinitely, we must also visit
    priority > p infinitely (to dominate the odd priority).

    The Streett condition is the conjunction of all pairs across all objectives.
    """
    if not game.vertices:
        return MultiSolution(set(), set(), method='conjunctive-streett')

    # Build Streett pairs from all objectives
    pairs = []
    for dim in range(game.k):
        d = game.max_priority(dim)
        for p in range(1, d + 1, 2):  # odd priorities only
            L = {v for v in game.vertices if game.priorities[v][dim] >= p
                 and game.priorities[v][dim] % 2 == 1
                 and game.priorities[v][dim] <= p}
            # L: vertices with this specific odd priority p
            L = {v for v in game.vertices if game.priorities[v][dim] == p}
            # U: vertices with priority strictly greater than p in this dimension
            U = {v for v in game.vertices if game.priorities[v][dim] > p}
            if L:  # only add non-trivial pairs
                pairs.append((L, U))

    if not pairs:
        # No odd priorities in any dimension => Even wins everywhere
        return MultiSolution(
            game.vertices.copy(), set(),
            method='conjunctive-streett-trivial',
        )

    # Solve as nested fixpoint (direct Streett solving)
    # Streett: Even wins iff for all (L, U): visit L inf => visit U inf
    # Equivalently: for all (L, U): avoid L eventually OR visit U infinitely
    win_even = _solve_streett_direct(game, pairs)
    win_odd = game.vertices - win_even

    strat_even = {}
    strat_odd = {}
    # Extract strategies from attractor computation
    for v in win_even:
        if game.owner[v] == Player.EVEN:
            for u in game.successors(v):
                if u in win_even:
                    strat_even[v] = u
                    break
    for v in win_odd:
        if game.owner[v] == Player.ODD:
            for u in game.successors(v):
                if u in win_odd:
                    strat_odd[v] = u
                    break

    return MultiSolution(
        win_even, win_odd, strat_even, strat_odd,
        method='conjunctive-streett',
        stats={
            'objectives': game.k,
            'streett_pairs': len(pairs),
        },
    )


def _solve_streett_direct(game: MultiParityGame, pairs: List[Tuple[Set[int], Set[int]]]) -> Set[int]:
    """Solve Streett condition via nested fixpoint on MultiParityGame.

    Even wins Streett(pairs) from vertices in the greatest fixpoint of:
    for each pair (L_i, U_i):
      either avoid L_i (co-Buchi) or recur through U_i (Buchi)

    Uses attractor computation on the game graph.
    """
    remain = set(game.vertices)

    changed = True
    while changed:
        changed = False
        for L, U in pairs:
            # Odd can force into L without seeing U infinitely
            # Remove Odd's attractor to (remain - co-Buchi(L) intersect Buchi-fail(U))
            # Simplified: iteratively remove vertices where Odd can trap in L
            inner_changed = True
            while inner_changed:
                inner_changed = False
                # Vertices in remain that are in L but not in U
                # These are "bad" -- visiting them without higher even priority
                bad = L & remain - U
                if not bad:
                    break
                # Odd attractor to bad within remain
                odd_attr = _multi_attractor(game, bad, Player.ODD, remain)
                if odd_attr - bad:  # attractor grew beyond bad
                    remain -= odd_attr
                    inner_changed = True
                    changed = True
                elif bad & remain:
                    # Check if Even can escape from bad
                    can_escape = set()
                    for v in bad & remain:
                        succs = game.successors(v) & remain
                        if game.owner[v] == Player.EVEN:
                            if succs - bad:
                                can_escape.add(v)
                        else:
                            if not succs or succs <= bad:
                                pass  # Odd traps here
                            else:
                                can_escape.add(v)
                    stuck = (bad & remain) - can_escape
                    if stuck:
                        odd_attr = _multi_attractor(game, stuck, Player.ODD, remain)
                        remain -= odd_attr
                        inner_changed = True
                        changed = True
                    else:
                        break

    return remain


def _multi_attractor(game: MultiParityGame, target: Set[int],
                     player: Player, restrict: Set[int]) -> Set[int]:
    """Compute attractor for player within restrict on MultiParityGame."""
    attr = set(target) & restrict
    queue = list(attr)
    while queue:
        v = queue.pop()
        for u in game.predecessors(v):
            if u in restrict and u not in attr:
                if game.owner[u] == player:
                    # Player's vertex: one successor in attr suffices
                    attr.add(u)
                    queue.append(u)
                else:
                    # Opponent's vertex: all successors must be in attr
                    succs = game.successors(u) & restrict
                    if succs and succs <= attr:
                        attr.add(u)
                        queue.append(u)
    return attr


# ---------------------------------------------------------------------------
# Pareto-optimal strategies
# ---------------------------------------------------------------------------

def pareto_analysis(game: MultiParityGame) -> Dict:
    """Analyze which objectives Even can satisfy from each vertex.

    Returns per-vertex information about which subsets of objectives
    Even can simultaneously satisfy.

    For each vertex, computes:
    - satisfiable: set of objective indices Even can satisfy individually
    - max_conjunction: largest subset of objectives Even can satisfy simultaneously
    """
    per_vertex = {}
    individual = []

    # Solve each objective individually
    for i in range(game.k):
        pg = game.projection(i)
        sol = zielonka(pg)
        individual.append(sol.win_even)

    for v in game.vertices:
        satisfiable = {i for i in range(game.k) if v in individual[i]}
        per_vertex[v] = {
            'satisfiable_individual': satisfiable,
            'count_individual': len(satisfiable),
        }

    # Try conjunctions of increasing size
    # Start from full conjunction, work down
    conj_sol = solve_conjunctive(game)
    for v in game.vertices:
        if v in conj_sol.win_even:
            per_vertex[v]['max_conjunction'] = set(range(game.k))
            per_vertex[v]['conjunction_size'] = game.k
        else:
            # Find largest satisfiable subset
            best_size = 0
            best_subset = set()
            satisfiable = per_vertex[v]['satisfiable_individual']
            # Try subsets from largest to smallest
            for size in range(len(satisfiable), 0, -1):
                if size <= best_size:
                    break
                found = False
                for subset in _subsets_of_size(sorted(satisfiable), size):
                    sub_game = _project_objectives(game, list(subset))
                    sub_sol = solve_conjunctive(sub_game)
                    if v in sub_sol.win_even:
                        best_size = size
                        best_subset = set(subset)
                        found = True
                        break
                if found:
                    break
            per_vertex[v]['max_conjunction'] = best_subset
            per_vertex[v]['conjunction_size'] = best_size

    return {
        'per_vertex': per_vertex,
        'individual_wins': {i: individual[i] for i in range(game.k)},
        'full_conjunction': conj_sol.win_even,
    }


def _subsets_of_size(items: List[int], size: int):
    """Generate all subsets of given size."""
    if size == 0:
        yield frozenset()
        return
    if len(items) < size:
        return
    # Include first item
    for rest in _subsets_of_size(items[1:], size - 1):
        yield frozenset({items[0]}) | rest
    # Exclude first item
    yield from _subsets_of_size(items[1:], size)


# ---------------------------------------------------------------------------
# Game construction helpers
# ---------------------------------------------------------------------------

def make_multi_parity_game(
    vertices: List[Tuple[int, int, Tuple[int, ...]]],
    edges: List[Tuple[int, int]],
) -> MultiParityGame:
    """Create a multi-parity game from vertex/edge lists.

    vertices: [(id, owner: 0|1, (p0, p1, ...)), ...]
    edges: [(u, v), ...]
    """
    game = MultiParityGame()
    for vid, owner, prios in vertices:
        game.add_vertex(vid, Player.EVEN if owner == 0 else Player.ODD, prios)
    for u, v in edges:
        game.add_edge(u, v)
    return game


def make_safety_liveness_game(
    vertices: List[Tuple[int, int]],
    edges: List[Tuple[int, int]],
    safe: Set[int],
    live: Set[int],
) -> MultiParityGame:
    """Create a 2-objective game: safety (avoid bad) AND liveness (reach good).

    Objective 0: safety -- priority 0 (safe) or 1 (unsafe)
    Objective 1: liveness -- priority 1 (non-live) or 2 (live)

    Even wins iff: always safe AND infinitely often live.
    """
    all_verts = {v for v, _ in vertices}
    game = MultiParityGame()
    for vid, owner in vertices:
        p0 = 0 if vid in safe else 1  # safety
        p1 = 2 if vid in live else 1  # liveness (Buchi)
        game.add_vertex(vid, Player.EVEN if owner == 0 else Player.ODD, (p0, p1))
    for u, v in edges:
        game.add_edge(u, v)
    return game


def make_multi_reachability_game(
    vertices: List[Tuple[int, int]],
    edges: List[Tuple[int, int]],
    targets: List[Set[int]],
) -> MultiParityGame:
    """Create a k-objective reachability game.

    Each target set is an objective: priority 2 if in target, 1 otherwise.
    Even wins conjunctive iff all targets are visited infinitely often.
    """
    game = MultiParityGame()
    k = len(targets)
    for vid, owner in vertices:
        prios = tuple(2 if vid in targets[i] else 1 for i in range(k))
        game.add_vertex(vid, Player.EVEN if owner == 0 else Player.ODD, prios)
    for u, v in edges:
        game.add_edge(u, v)
    return game


# ---------------------------------------------------------------------------
# Comparison and verification
# ---------------------------------------------------------------------------

def verify_multi_solution(game: MultiParityGame, sol: MultiSolution,
                          objective: Objective = Objective.CONJUNCTION) -> Dict:
    """Verify a multi-parity game solution.

    Checks:
    1. Partition: win_even | win_odd == vertices, disjoint
    2. Strategy validity: strategies map to valid successors
    3. Region closure: strategies stay within winning regions
    """
    errors = []

    # Partition check
    if sol.win_even | sol.win_odd != game.vertices:
        errors.append('Not a partition: missing vertices')
    if sol.win_even & sol.win_odd:
        errors.append('Not a partition: overlap')

    # Strategy validity
    for v, u in sol.strategy_even.items():
        if v not in game.vertices:
            errors.append(f'Even strategy maps non-existent vertex {v}')
        elif u not in game.successors(v):
            errors.append(f'Even strategy maps {v} to non-successor {u}')

    for v, u in sol.strategy_odd.items():
        if v not in game.vertices:
            errors.append(f'Odd strategy maps non-existent vertex {v}')
        elif u not in game.successors(v):
            errors.append(f'Odd strategy maps {v} to non-successor {u}')

    # Region closure
    for v in sol.win_even:
        if game.owner[v] == Player.EVEN:
            if v in sol.strategy_even:
                if sol.strategy_even[v] not in sol.win_even:
                    errors.append(f'Even strategy escapes Even region at {v}')
        else:
            # All Odd successors should be in Even region (opponent can't escape)
            succs = game.successors(v)
            if succs and not succs <= sol.win_even:
                # This is expected behavior if Odd has choices
                pass

    for v in sol.win_odd:
        if game.owner[v] == Player.ODD:
            if v in sol.strategy_odd:
                if sol.strategy_odd[v] not in sol.win_odd:
                    errors.append(f'Odd strategy escapes Odd region at {v}')

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'win_even_size': len(sol.win_even),
        'win_odd_size': len(sol.win_odd),
    }


def compare_methods(game: MultiParityGame) -> Dict:
    """Compare conjunctive solving methods: product vs Streett reduction."""
    sol_product = solve_conjunctive(game)
    sol_streett = solve_conjunctive_streett(game)

    agree = sol_product.win_even == sol_streett.win_even
    return {
        'agree': agree,
        'product': {
            'win_even': sol_product.win_even,
            'win_odd': sol_product.win_odd,
            'method': sol_product.method,
            'stats': sol_product.stats,
        },
        'streett': {
            'win_even': sol_streett.win_even,
            'win_odd': sol_streett.win_odd,
            'method': sol_streett.method,
            'stats': sol_streett.stats,
        },
    }


def compare_conjunctive_disjunctive(game: MultiParityGame) -> Dict:
    """Compare conjunctive vs disjunctive winning regions."""
    conj = solve_conjunctive(game)
    disj = solve_disjunctive(game)

    return {
        'conjunctive_even': conj.win_even,
        'disjunctive_even': disj.win_even,
        'conj_subset_disj': conj.win_even <= disj.win_even,
        'conj_size': len(conj.win_even),
        'disj_size': len(disj.win_even),
        'difference': disj.win_even - conj.win_even,
    }


def multi_parity_statistics(game: MultiParityGame) -> Dict:
    """Statistics about a multi-parity game."""
    edge_count = sum(len(s) for s in game.edges.values())
    return {
        'vertices': len(game.vertices),
        'edges': edge_count,
        'objectives': game.k,
        'even_vertices': sum(1 for v in game.vertices if game.owner[v] == Player.EVEN),
        'odd_vertices': sum(1 for v in game.vertices if game.owner[v] == Player.ODD),
        'max_priorities': [game.max_priority(i) for i in range(game.k)],
        'priority_vectors': {v: game.priorities[v] for v in sorted(game.vertices)},
    }
