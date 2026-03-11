"""
V156: Parity Games
==================
Infinite-duration two-player games on graphs with parity winning conditions.

Parity games are central to verification theory:
- They decide the modal mu-calculus (which subsumes CTL, LTL, CTL*)
- Key to reactive synthesis and controller design
- Solving them is in NP intersect coNP (believed polynomial but unproven)

Players: Even (player 0) and Odd (player 1).
Each vertex has an owner (who chooses the successor) and a priority (natural number).
A play is an infinite path. The winner is determined by the highest priority
seen infinitely often: Even wins iff that priority is even.

Algorithms implemented:
1. Attractor computation
2. Zielonka's recursive algorithm
3. Small Progress Measures (Jurdzinski)
4. Priority Promotion (Benerecetti-Dell'Erba-Mogavero)
5. Strategy extraction and verification
6. Reductions: mu-calculus model checking, LTL/safety games
"""

from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math


class Player(Enum):
    EVEN = 0  # Player 0 -- wants highest inf-often priority to be even
    ODD = 1   # Player 1 -- wants highest inf-often priority to be odd

    @property
    def opponent(self):
        return Player.ODD if self == Player.EVEN else Player.EVEN


@dataclass
class ParityGame:
    """A parity game arena with vertices, edges, owners, and priorities."""
    vertices: Set[int] = field(default_factory=set)
    edges: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    owner: Dict[int, Player] = field(default_factory=dict)
    priority: Dict[int, int] = field(default_factory=dict)

    def add_vertex(self, v: int, player: Player, prio: int):
        self.vertices.add(v)
        self.owner[v] = player
        self.priority[v] = prio
        if v not in self.edges:
            self.edges[v] = set()

    def add_edge(self, u: int, v: int):
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges[u].add(v)

    def successors(self, v: int) -> Set[int]:
        return self.edges.get(v, set())

    def predecessors(self, v: int) -> Set[int]:
        return {u for u in self.vertices if v in self.edges.get(u, set())}

    def max_priority(self) -> int:
        if not self.priority:
            return 0
        return max(self.priority.values())

    def vertices_with_priority(self, p: int) -> Set[int]:
        return {v for v in self.vertices if self.priority.get(v) == p}

    def subgame(self, verts: Set[int]) -> 'ParityGame':
        """Create a subgame restricted to the given vertex set."""
        g = ParityGame()
        for v in verts:
            if v in self.owner and v in self.priority:
                g.add_vertex(v, self.owner[v], self.priority[v])
        for v in verts:
            for u in self.edges.get(v, set()):
                if u in verts:
                    g.add_edge(v, u)
        return g

    def has_dead_ends(self, verts: Optional[Set[int]] = None) -> Set[int]:
        """Return vertices with no successors within the given set."""
        if verts is None:
            verts = self.vertices
        return {v for v in verts if not (self.edges.get(v, set()) & verts)}


@dataclass
class Solution:
    """Solution to a parity game: winning regions + strategies."""
    win_even: Set[int] = field(default_factory=set)  # W0 -- vertices won by Even
    win_odd: Set[int] = field(default_factory=set)   # W1 -- vertices won by Odd
    strategy_even: Dict[int, int] = field(default_factory=dict)  # sigma0
    strategy_odd: Dict[int, int] = field(default_factory=dict)   # sigma1


# =============================================================================
# Attractor Computation
# =============================================================================

def attractor(game: ParityGame, target: Set[int], player: Player,
              restrict: Optional[Set[int]] = None) -> Set[int]:
    """
    Compute the attractor of target for player within restrict.

    Attr_p(T) = vertices from which player p can force reaching T.
    - Player p's vertices: if ANY successor is in the attractor
    - Opponent's vertices: if ALL successors are in the attractor
    """
    if restrict is None:
        restrict = game.vertices
    attr = set(v for v in target if v in restrict)
    queue = deque(attr)

    while queue:
        v = queue.popleft()
        for u in restrict:
            if u in attr:
                continue
            if v not in game.edges.get(u, set()):
                continue
            # u -> v is an edge, u is in restrict, u not yet in attractor
            if game.owner.get(u) == player:
                # Player's vertex: one successor in attr suffices
                attr.add(u)
                queue.append(u)
            else:
                # Opponent's vertex: ALL successors must be in attr
                succs = game.edges.get(u, set()) & restrict
                if succs and succs <= attr:
                    attr.add(u)
                    queue.append(u)

    return attr


# =============================================================================
# Zielonka's Recursive Algorithm
# =============================================================================

def zielonka(game: ParityGame) -> Solution:
    """
    Zielonka's recursive algorithm for solving parity games.

    Complexity: O(n^d) where d = number of priorities (exponential worst case,
    but excellent in practice).

    Algorithm:
    1. Handle dead ends: vertices with no successors lose for their owner.
    2. Find max priority p. Let i = p % 2 (the player who "likes" this priority).
    3. Compute Attr_i(vertices with priority p).
    4. Recursively solve the subgame without the attractor.
    5. If opponent wins nothing in the subgame -> i wins everything.
    6. Otherwise, compute Attr_{1-i}(opponent's winning region) and recurse on the rest.
    """
    sol = Solution()
    _zielonka_rec(game, game.vertices, sol)
    _compute_strategies(game, sol)
    return sol


def _zielonka_rec(game: ParityGame, verts: Set[int], sol: Solution):
    if not verts:
        return

    sub = game.subgame(verts)

    # Handle dead ends first: a dead-end vertex is lost by its owner
    # (the player who must move but can't). The opponent wins it.
    dead_ends_even = {v for v in verts
                      if game.owner.get(v) == Player.EVEN
                      and not (game.edges.get(v, set()) & verts)}
    dead_ends_odd = {v for v in verts
                     if game.owner.get(v) == Player.ODD
                     and not (game.edges.get(v, set()) & verts)}

    if dead_ends_even or dead_ends_odd:
        # Dead-end Even vertices: Odd wins them and their attractor
        if dead_ends_even:
            odd_attr = attractor(sub, dead_ends_even, Player.ODD, verts)
            sol.win_odd |= odd_attr
            remaining = verts - odd_attr
            _zielonka_rec(game, remaining, sol)
            return
        # Dead-end Odd vertices: Even wins them and their attractor
        if dead_ends_odd:
            even_attr = attractor(sub, dead_ends_odd, Player.EVEN, verts)
            sol.win_even |= even_attr
            remaining = verts - even_attr
            _zielonka_rec(game, remaining, sol)
            return

    # Find max priority
    max_p = max(sub.priority[v] for v in verts if v in sub.priority)
    # Player who benefits from this priority
    player = Player.EVEN if max_p % 2 == 0 else Player.ODD
    opponent = player.opponent

    # Vertices with max priority
    target = {v for v in verts if sub.priority.get(v) == max_p}

    # Compute attractor for the benefiting player
    attr = attractor(sub, target, player, verts)

    # Recursively solve subgame without attractor
    remaining = verts - attr
    sub_sol = Solution()
    _zielonka_rec(game, remaining, sub_sol)

    # Check opponent's winning region in the subgame
    opp_win = sub_sol.win_odd if player == Player.EVEN else sub_sol.win_even

    if not opp_win:
        # Player wins everything in this subgame
        if player == Player.EVEN:
            sol.win_even |= verts
            sol.strategy_even.update(sub_sol.strategy_even)
        else:
            sol.win_odd |= verts
            sol.strategy_odd.update(sub_sol.strategy_odd)
    else:
        # Opponent wins some region -- expand it with attractor
        opp_attr = attractor(sub, opp_win, opponent, verts)

        # Opponent wins opp_attr
        if opponent == Player.EVEN:
            sol.win_even |= opp_attr
            sol.strategy_even.update(
                {v: s for v, s in sub_sol.strategy_even.items() if v in opp_win}
            )
        else:
            sol.win_odd |= opp_attr
            sol.strategy_odd.update(
                {v: s for v, s in sub_sol.strategy_odd.items() if v in opp_win}
            )

        # Recurse on the rest
        rest = verts - opp_attr
        _zielonka_rec(game, rest, sol)


# =============================================================================
# Small Progress Measures (Jurdzinski)
# =============================================================================

def small_progress_measures(game: ParityGame) -> Solution:
    """
    Jurdzinski's Small Progress Measures algorithm.

    A progress measure mu: V -> M_d maps vertices to tuples.
    mu(v) = TOP means v is won by Odd.
    The least fixpoint of the Prog operator gives Even's winning region.

    For Even vertices: mu(v) = min_{v->w} Prog(v, mu(w))
    For Odd vertices: mu(v) = max_{v->w} Prog(v, mu(w))
    """
    if not game.vertices:
        return Solution()

    max_p = game.max_priority()
    # Only odd priorities matter for the measure
    odd_priorities = sorted(set(p for p in game.priority.values() if p % 2 == 1))

    if not odd_priorities:
        # All priorities are even -> Even wins everything
        sol = Solution(win_even=set(game.vertices))
        _compute_strategies(game, sol)
        return sol

    # Measure dimension: one component per odd priority
    dim = len(odd_priorities)
    prio_to_idx = {p: i for i, p in enumerate(odd_priorities)}

    # For each odd priority p, bound = number of vertices with that exact priority
    bounds = []
    for p in odd_priorities:
        count = sum(1 for v in game.vertices if game.priority[v] == p)
        bounds.append(count)

    TOP = "TOP"

    def is_top(m):
        return m == TOP

    def zero():
        return tuple(0 for _ in range(dim))

    def lt(a, b):
        """Strict lexicographic less-than."""
        if is_top(a):
            return False
        if is_top(b):
            return True
        return a < b

    def le(a, b):
        if a == b:
            return True
        return lt(a, b)

    def prog(v, m):
        """
        Prog(v, m) for MAX-PARITY games.

        In max-parity, higher priorities dominate lower ones. An even priority p
        "resets" the worry about odd priorities BELOW p (because the even priority
        will be the max-inf-often, not the lower odd ones).

        For even p: zero out odd components for priorities < p (dominated by p).
        For odd p: zero out odd components for priorities < p, then increment at p.
        """
        if is_top(m):
            return TOP
        p = game.priority[v]

        # Truncate: zero out components for odd priorities BELOW p
        # (In max-parity, higher even priority resets lower odd worries)
        result = list(m)
        for i, op in enumerate(odd_priorities):
            if op < p:
                result[i] = 0

        if p % 2 == 0:
            # Even priority: just truncate (no increment needed)
            return tuple(result)
        else:
            # Odd priority: increment at this priority's position
            idx = prio_to_idx[p]
            result[idx] += 1
            # Handle overflow: cascade to HIGHER odd priority positions
            if result[idx] > bounds[idx]:
                result[idx] = 0
                carry = True
                for i in range(idx + 1, dim):
                    if carry:
                        result[i] += 1
                        if result[i] > bounds[i]:
                            result[i] = 0
                        else:
                            carry = False
                            break
                if carry:
                    return TOP
            return tuple(result)

    # Initialize all measures to zero
    mu = {v: zero() for v in game.vertices}

    # Iterative lifting until fixpoint
    changed = True
    while changed:
        changed = False
        for v in game.vertices:
            succs = game.successors(v)
            if not succs:
                if game.owner[v] == Player.EVEN:
                    new_val = TOP  # Even can't move -> loses
                else:
                    new_val = zero()  # Odd can't move -> loses
            elif game.owner[v] == Player.EVEN:
                # Even chooses the minimum prog value
                new_val = TOP
                for w in succs:
                    lifted = prog(v, mu[w])
                    if lt(lifted, new_val):
                        new_val = lifted
            else:
                # Odd chooses the maximum prog value
                new_val = zero()
                for w in succs:
                    lifted = prog(v, mu[w])
                    if lt(new_val, lifted):
                        new_val = lifted

            # Only update if new value is strictly greater (monotone lifting)
            if mu[v] != new_val and lt(mu[v], new_val):
                mu[v] = new_val
                changed = True

    # Partition: TOP -> Odd wins, otherwise Even wins
    sol = Solution()
    for v in game.vertices:
        if is_top(mu[v]):
            sol.win_odd.add(v)
        else:
            sol.win_even.add(v)

    _compute_strategies(game, sol)
    return sol


# =============================================================================
# Priority Promotion
# =============================================================================

def priority_promotion(game: ParityGame) -> Solution:
    """
    McNaughton-Zielonka style iterative algorithm.

    An alternative iterative formulation of parity game solving:
    Process priorities from highest to lowest. For the current max priority p,
    compute the attractor for player(p) of all vertices with priority p.
    Recursively solve the remainder. If the opponent wins nothing -> player(p) wins all.
    Otherwise remove the opponent's winning attractor and restart.

    This is equivalent to Zielonka but expressed iteratively with a worklist,
    which can be more efficient when dominions are found early.
    """
    if not game.vertices:
        return Solution()

    sol = Solution()
    remaining = set(game.vertices)

    while remaining:
        sub = game.subgame(remaining)

        # Handle dead ends
        dead_even = {v for v in remaining
                     if game.owner.get(v) == Player.EVEN
                     and not (game.edges.get(v, set()) & remaining)}
        dead_odd = {v for v in remaining
                    if game.owner.get(v) == Player.ODD
                    and not (game.edges.get(v, set()) & remaining)}
        if dead_even:
            attr_set = attractor(sub, dead_even, Player.ODD, remaining)
            sol.win_odd |= attr_set
            remaining -= attr_set
            continue
        if dead_odd:
            attr_set = attractor(sub, dead_odd, Player.EVEN, remaining)
            sol.win_even |= attr_set
            remaining -= attr_set
            continue

        # Find max priority in remaining
        max_p = max(game.priority[v] for v in remaining)
        player = Player.EVEN if max_p % 2 == 0 else Player.ODD
        opponent = player.opponent

        # Attractor of max-priority vertices
        target = {v for v in remaining if game.priority[v] == max_p}
        attr_set = attractor(sub, target, player, remaining)

        # Solve remainder
        rest = remaining - attr_set
        rest_sol = Solution()
        if rest:
            rest_sub = game.subgame(rest)
            # Recursive call via zielonka on remainder
            _zielonka_rec(game, rest, rest_sol)

        opp_win = rest_sol.win_odd if player == Player.EVEN else rest_sol.win_even

        if not opp_win:
            # Player wins everything
            if player == Player.EVEN:
                sol.win_even |= remaining
            else:
                sol.win_odd |= remaining
            remaining = set()
        else:
            # Opponent wins some -- expand with attractor
            opp_attr = attractor(sub, opp_win, opponent, remaining)
            if opponent == Player.EVEN:
                sol.win_even |= opp_attr
            else:
                sol.win_odd |= opp_attr
            remaining -= opp_attr

    _compute_strategies(game, sol)
    return sol


# =============================================================================
# Strategy Computation and Verification
# =============================================================================

def _compute_strategies(game: ParityGame, sol: Solution):
    """Compute winning strategies from winning regions."""
    for v in sol.win_even:
        if game.owner.get(v) == Player.EVEN:
            # Even picks a successor in win_even
            for w in game.successors(v):
                if w in sol.win_even:
                    sol.strategy_even[v] = w
                    break
    for v in sol.win_odd:
        if game.owner.get(v) == Player.ODD:
            # Odd picks a successor in win_odd
            for w in game.successors(v):
                if w in sol.win_odd:
                    sol.strategy_odd[v] = w
                    break


def verify_solution(game: ParityGame, sol: Solution) -> Tuple[bool, List[str]]:
    """
    Verify that a solution is correct:
    1. Winning regions partition the vertex set
    2. Winning regions are closed under the respective strategies
    3. Strategies are well-defined (every owned vertex has a chosen successor)
    """
    errors = []

    # Check partition
    if sol.win_even | sol.win_odd != game.vertices:
        missing = game.vertices - (sol.win_even | sol.win_odd)
        errors.append(f"Vertices not assigned: {missing}")

    overlap = sol.win_even & sol.win_odd
    if overlap:
        errors.append(f"Vertices in both regions: {overlap}")

    # Check strategies are valid
    for v in sol.win_even:
        if game.owner.get(v) == Player.EVEN:
            if v not in sol.strategy_even:
                succs = game.successors(v)
                if succs:  # Only error if vertex has successors
                    errors.append(f"Even vertex {v} in W0 has no strategy")
            elif sol.strategy_even[v] not in game.successors(v):
                errors.append(f"Even strategy at {v}: {sol.strategy_even[v]} not a successor")
            elif sol.strategy_even[v] not in sol.win_even:
                errors.append(f"Even strategy at {v} leads to W1")

    for v in sol.win_odd:
        if game.owner.get(v) == Player.ODD:
            if v not in sol.strategy_odd:
                succs = game.successors(v)
                if succs:
                    errors.append(f"Odd vertex {v} in W1 has no strategy")
            elif sol.strategy_odd[v] not in game.successors(v):
                errors.append(f"Odd strategy at {v}: {sol.strategy_odd[v]} not a successor")
            elif sol.strategy_odd[v] not in sol.win_odd:
                errors.append(f"Odd strategy at {v} leads to W0")

    # Check opponent vertices: all successors must stay in winning region
    for v in sol.win_even:
        if game.owner.get(v) == Player.ODD:
            for w in game.successors(v):
                if w not in sol.win_even:
                    errors.append(f"Odd vertex {v} in W0 can escape to {w} in W1")

    for v in sol.win_odd:
        if game.owner.get(v) == Player.EVEN:
            for w in game.successors(v):
                if w not in sol.win_odd:
                    errors.append(f"Even vertex {v} in W1 can escape to {w} in W0")

    return len(errors) == 0, errors


def simulate_play(game: ParityGame, sol: Solution, start: int,
                  max_steps: int = 100) -> List[Tuple[int, int]]:
    """
    Simulate a play from start using the winning strategies.
    Returns list of (vertex, priority) pairs.
    """
    play = []
    v = start
    visited = {}

    for step in range(max_steps):
        play.append((v, game.priority.get(v, 0)))

        if v in visited:
            # Found a cycle -- this determines the winner
            break
        visited[v] = step

        succs = game.successors(v)
        if not succs:
            break

        if game.owner.get(v) == Player.EVEN:
            if v in sol.strategy_even:
                v = sol.strategy_even[v]
            else:
                v = min(succs)  # fallback
        else:
            if v in sol.strategy_odd:
                v = sol.strategy_odd[v]
            else:
                v = min(succs)

    return play


# =============================================================================
# Game Construction Helpers
# =============================================================================

def make_game(vertices: List[Tuple[int, int, int]],
              edges: List[Tuple[int, int]]) -> ParityGame:
    """
    Construct a parity game.
    vertices: list of (id, owner: 0 or 1, priority)
    edges: list of (from, to)
    """
    g = ParityGame()
    for vid, own, prio in vertices:
        g.add_vertex(vid, Player.EVEN if own == 0 else Player.ODD, prio)
    for u, v in edges:
        g.add_edge(u, v)
    return g


def make_safety_game(states: int, bad: Set[int], player_states: Set[int],
                     transitions: List[Tuple[int, int]]) -> ParityGame:
    """
    Construct a safety game: Even wins iff the play never visits a bad state.
    Priority 0 for safe states (Even likes), priority 1 for bad states (Odd likes).
    """
    g = ParityGame()
    for s in range(states):
        owner = Player.EVEN if s in player_states else Player.ODD
        prio = 1 if s in bad else 0
        g.add_vertex(s, owner, prio)
    for u, v in transitions:
        g.add_edge(u, v)
    return g


def make_reachability_game(states: int, target: Set[int], player_states: Set[int],
                           transitions: List[Tuple[int, int]]) -> ParityGame:
    """
    Construct a reachability game: Even wins iff the play reaches a target state.
    Priority 1 for non-target (Odd likes: staying away), priority 2 for target (Even likes: reaching).
    Target states get self-loops (absorbing).
    """
    g = ParityGame()
    for s in range(states):
        owner = Player.EVEN if s in player_states else Player.ODD
        prio = 2 if s in target else 1
        g.add_vertex(s, owner, prio)
    for u, v in transitions:
        g.add_edge(u, v)
    # Self-loops on targets (so Even wins once reached)
    for t in target:
        g.add_edge(t, t)
    return g


def make_buchi_game(states: int, accepting: Set[int], player_states: Set[int],
                    transitions: List[Tuple[int, int]]) -> ParityGame:
    """
    Construct a Buchi game: Even wins iff accepting states are visited infinitely often.
    Priority 1 for non-accepting, priority 2 for accepting.
    """
    g = ParityGame()
    for s in range(states):
        owner = Player.EVEN if s in player_states else Player.ODD
        prio = 2 if s in accepting else 1
        g.add_vertex(s, owner, prio)
    for u, v in transitions:
        g.add_edge(u, v)
    return g


def make_co_buchi_game(states: int, rejecting: Set[int], player_states: Set[int],
                       transitions: List[Tuple[int, int]]) -> ParityGame:
    """
    Construct a co-Buchi game: Even wins iff rejecting states are visited only finitely often.
    Priority 0 for non-rejecting (good for Even), priority 1 for rejecting (bad).
    """
    g = ParityGame()
    for s in range(states):
        owner = Player.EVEN if s in player_states else Player.ODD
        prio = 1 if s in rejecting else 0
        g.add_vertex(s, owner, prio)
    for u, v in transitions:
        g.add_edge(u, v)
    return g


# =============================================================================
# Analysis and Comparison
# =============================================================================

def solve_all(game: ParityGame) -> Dict[str, Solution]:
    """Solve using all three algorithms and return results."""
    return {
        'zielonka': zielonka(game),
        'spm': small_progress_measures(game),
        'priority_promotion': priority_promotion(game),
    }


def compare_algorithms(game: ParityGame) -> Dict:
    """Compare all algorithms on a game, checking consistency."""
    solutions = solve_all(game)

    results = {}
    for name, sol in solutions.items():
        valid, errors = verify_solution(game, sol)
        results[name] = {
            'win_even': sol.win_even,
            'win_odd': sol.win_odd,
            'valid': valid,
            'errors': errors,
        }

    # Check agreement
    algs = list(solutions.keys())
    agree = True
    for i in range(len(algs)):
        for j in range(i + 1, len(algs)):
            if solutions[algs[i]].win_even != solutions[algs[j]].win_even:
                agree = False

    results['all_agree'] = agree
    return results


def game_statistics(game: ParityGame) -> Dict:
    """Compute statistics about a parity game."""
    return {
        'vertices': len(game.vertices),
        'edges': sum(len(s) for s in game.edges.values()),
        'max_priority': game.max_priority(),
        'num_priorities': len(set(game.priority.values())),
        'even_vertices': sum(1 for v in game.vertices if game.owner.get(v) == Player.EVEN),
        'odd_vertices': sum(1 for v in game.vertices if game.owner.get(v) == Player.ODD),
        'dead_ends': len(game.has_dead_ends()),
        'priority_distribution': {
            p: sum(1 for v in game.vertices if game.priority.get(v) == p)
            for p in sorted(set(game.priority.values()))
        },
    }


def game_summary(game: ParityGame) -> str:
    """Human-readable summary of a parity game and its solution."""
    stats = game_statistics(game)
    sol = zielonka(game)
    valid, errors = verify_solution(game, sol)

    lines = [
        f"Parity Game: {stats['vertices']} vertices, {stats['edges']} edges",
        f"Priorities: {stats['num_priorities']} distinct (max {stats['max_priority']})",
        f"Owners: {stats['even_vertices']} Even, {stats['odd_vertices']} Odd",
        f"Solution: W0={sorted(sol.win_even)}, W1={sorted(sol.win_odd)}",
        f"Valid: {valid}",
    ]
    if errors:
        lines.append(f"Errors: {errors}")
    return "\n".join(lines)
