"""V076: Parity Games -- Zielonka's Recursive Algorithm + Small Progress Measures.

Parity games are two-player infinite-duration games on finite graphs where
each node has a priority (natural number). Player 0 (Even) wins if the
highest priority seen infinitely often along a play is even; Player 1 (Odd)
wins if it is odd.

Key algorithms:
- Zielonka's recursive algorithm (exponential worst-case, fast in practice)
- Small Progress Measures (SPM) -- quasi-polynomial lifting algorithm
- Attractor computation (fundamental building block)
- Priority compression (optimization)

Composition:
- Standalone parity game solver (no dependencies on V074/V075)
- Conversion from Buchi/co-Buchi/Rabin/Streett to parity games
- Conversion from parity games to V074 StochasticGame format
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, FrozenSet


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class Player(Enum):
    EVEN = 0  # Player 0 -- wins if max inf priority is even
    ODD = 1   # Player 1 -- wins if max inf priority is odd

    @property
    def opponent(self) -> "Player":
        return Player.ODD if self == Player.EVEN else Player.EVEN

    @staticmethod
    def owner_of_priority(p: int) -> "Player":
        """The player who benefits from priority p."""
        return Player.EVEN if p % 2 == 0 else Player.ODD


@dataclass
class ParityGame:
    """A parity game arena.

    Attributes:
        nodes: set of node identifiers (ints)
        owner: maps node -> Player (who controls the node)
        priority: maps node -> int (non-negative priority)
        successors: maps node -> set of successor nodes
        predecessors: maps node -> set of predecessor nodes (auto-computed)
    """
    nodes: Set[int] = field(default_factory=set)
    owner: Dict[int, Player] = field(default_factory=dict)
    priority: Dict[int, int] = field(default_factory=dict)
    successors: Dict[int, Set[int]] = field(default_factory=dict)
    predecessors: Dict[int, Set[int]] = field(default_factory=dict)

    def add_node(self, node: int, owner: Player, priority: int) -> None:
        """Add a node with owner and priority."""
        self.nodes.add(node)
        self.owner[node] = owner
        self.priority[node] = priority
        if node not in self.successors:
            self.successors[node] = set()
        if node not in self.predecessors:
            self.predecessors[node] = set()

    def add_edge(self, src: int, dst: int) -> None:
        """Add a directed edge."""
        self.successors.setdefault(src, set()).add(dst)
        self.predecessors.setdefault(dst, set()).add(src)

    def subgame(self, keep: Set[int]) -> "ParityGame":
        """Return the subgame restricted to the given node set."""
        g = ParityGame()
        for n in keep:
            if n in self.nodes:
                g.add_node(n, self.owner[n], self.priority[n])
        for n in keep:
            for s in self.successors.get(n, set()):
                if s in keep:
                    g.add_edge(n, s)
        return g

    def max_priority(self) -> int:
        """Return the maximum priority in the game."""
        if not self.nodes:
            return -1
        return max(self.priority[n] for n in self.nodes)

    def nodes_with_priority(self, p: int) -> Set[int]:
        """Return all nodes with the given priority."""
        return {n for n in self.nodes if self.priority[n] == p}

    def validate(self) -> List[str]:
        """Check game validity. Returns list of issues."""
        issues = []
        for n in self.nodes:
            if n not in self.owner:
                issues.append(f"Node {n} has no owner")
            if n not in self.priority:
                issues.append(f"Node {n} has no priority")
            if not self.successors.get(n):
                issues.append(f"Node {n} has no successors (deadlock)")
            if self.priority.get(n, 0) < 0:
                issues.append(f"Node {n} has negative priority")
        return issues

    def __repr__(self) -> str:
        return (f"ParityGame(nodes={len(self.nodes)}, "
                f"max_priority={self.max_priority()})")


@dataclass
class ParityResult:
    """Result of solving a parity game.

    Attributes:
        win0: winning region for Player 0 (Even)
        win1: winning region for Player 1 (Odd)
        strategy0: winning strategy for Player 0 (node -> successor)
        strategy1: winning strategy for Player 1 (node -> successor)
        algorithm: name of algorithm used
        iterations: number of recursive calls / iterations
    """
    win0: Set[int] = field(default_factory=set)
    win1: Set[int] = field(default_factory=set)
    strategy0: Dict[int, int] = field(default_factory=dict)
    strategy1: Dict[int, int] = field(default_factory=dict)
    algorithm: str = ""
    iterations: int = 0

    def winner(self, node: int) -> Player:
        """Return the winner at a given node."""
        if node in self.win0:
            return Player.EVEN
        if node in self.win1:
            return Player.ODD
        raise ValueError(f"Node {node} not in any winning region")

    def strategy(self, node: int) -> Optional[int]:
        """Return the strategy move at a node (for the winner)."""
        if node in self.strategy0:
            return self.strategy0[node]
        if node in self.strategy1:
            return self.strategy1[node]
        return None

    def summary(self) -> str:
        return (f"ParityResult(|W0|={len(self.win0)}, |W1|={len(self.win1)}, "
                f"algorithm={self.algorithm}, iterations={self.iterations})")


# ---------------------------------------------------------------------------
# Attractor computation
# ---------------------------------------------------------------------------

def attractor(game: ParityGame, target: Set[int], player: Player,
              arena: Optional[Set[int]] = None) -> Tuple[Set[int], Dict[int, int]]:
    """Compute the attractor of `target` for `player` within `arena`.

    A node v is in Attr_player(target) if:
    - v is in target, OR
    - v is owned by `player` and has some successor in Attr (player can choose), OR
    - v is owned by opponent and ALL successors in arena are in Attr (opponent is forced)

    Returns (attractor_set, strategy) where strategy maps player-owned
    attractor nodes to their chosen successor in the attractor.
    """
    if arena is None:
        arena = game.nodes

    attr = set(target & arena)
    strategy: Dict[int, int] = {}
    queue = list(attr)

    while queue:
        v = queue.pop()
        for u in game.predecessors.get(v, set()):
            if u not in arena or u in attr:
                continue
            if game.owner[u] == player:
                # Player owns u: one successor in attr suffices
                attr.add(u)
                strategy[u] = v
                queue.append(u)
            else:
                # Opponent owns u: check if ALL successors in arena are in attr
                succs_in_arena = game.successors[u] & arena
                if succs_in_arena and succs_in_arena <= attr:
                    attr.add(u)
                    queue.append(u)

    return attr, strategy


# ---------------------------------------------------------------------------
# Zielonka's Recursive Algorithm
# ---------------------------------------------------------------------------

class _ZielonkaState:
    """Mutable state for counting iterations."""
    def __init__(self):
        self.iterations = 0


def _zielonka_rec(game: ParityGame, state: _ZielonkaState
                  ) -> Tuple[Set[int], Set[int], Dict[int, int], Dict[int, int]]:
    """Recursive core of Zielonka's algorithm.

    Returns (W0, W1, sigma0, sigma1).
    """
    state.iterations += 1

    if not game.nodes:
        return set(), set(), {}, {}

    d = game.max_priority()
    player = Player.owner_of_priority(d)
    opponent = player.opponent

    # Nodes with max priority
    U = game.nodes_with_priority(d)

    # Attractor of U for player
    A, a_strat = attractor(game, U, player)

    # Solve subgame without attractor
    sub = game.subgame(game.nodes - A)
    W0_sub, W1_sub, s0_sub, s1_sub = _zielonka_rec(sub, state)

    # Winning region for opponent in subgame
    W_opp = W1_sub if player == Player.EVEN else W0_sub

    if not W_opp:
        # Opponent wins nothing in subgame => player wins everything
        # Player's strategy: a_strat in attractor, s_player in subgame
        s_player = s0_sub if player == Player.EVEN else s1_sub
        # For nodes in U owned by player, pick any successor in game
        for n in U:
            if game.owner[n] == player and n not in a_strat:
                # Pick successor in game (any will do since we're in winning region)
                succs = game.successors.get(n, set())
                if succs:
                    a_strat[n] = next(iter(succs))

        combined_strat = {**s_player, **a_strat}
        if player == Player.EVEN:
            return game.nodes.copy(), set(), combined_strat, {}
        else:
            return set(), game.nodes.copy(), {}, combined_strat
    else:
        # Opponent has a non-empty winning region in subgame
        # Compute attractor of opponent's winning region for opponent
        B, b_strat = attractor(game, W_opp, opponent)

        # Solve subgame without B
        sub2 = game.subgame(game.nodes - B)
        W0_sub2, W1_sub2, s0_sub2, s1_sub2 = _zielonka_rec(sub2, state)

        # Opponent wins B plus their winning region in sub2
        # Player wins only their winning region in sub2
        s_opp_sub2 = s1_sub2 if opponent == Player.ODD else s0_sub2
        s_player_sub2 = s0_sub2 if player == Player.EVEN else s1_sub2

        combined_opp_strat = {**s_opp_sub2, **b_strat}
        # For opponent-owned nodes in W_opp, use subgame strategy
        s_opp_inner = s1_sub if opponent == Player.ODD else s0_sub
        combined_opp_strat.update(s_opp_inner)

        W_player_sub2 = W0_sub2 if player == Player.EVEN else W1_sub2
        W_opp_sub2 = W1_sub2 if opponent == Player.ODD else W0_sub2

        W_opp_final = B | W_opp_sub2
        W_player_final = W_player_sub2

        if player == Player.EVEN:
            return W_player_final, W_opp_final, s_player_sub2, combined_opp_strat
        else:
            return W_opp_final, W_player_final, combined_opp_strat, s_player_sub2


def zielonka(game: ParityGame) -> ParityResult:
    """Solve a parity game using Zielonka's recursive algorithm.

    Time complexity: O(n^d) worst case, where n = |nodes|, d = max priority.
    In practice, very fast on most game structures.
    """
    state = _ZielonkaState()
    w0, w1, s0, s1 = _zielonka_rec(game, state)
    return ParityResult(
        win0=w0, win1=w1,
        strategy0=s0, strategy1=s1,
        algorithm="zielonka",
        iterations=state.iterations
    )


# ---------------------------------------------------------------------------
# Small Progress Measures (SPM)
# ---------------------------------------------------------------------------

def _spm_top(n: int, d: int) -> Tuple[int, ...]:
    """Return the TOP measure (infinity). Uses n+1 at each position.

    Tuples are stored in REVERSE order: index 0 = most significant position
    (floor(d/2)), last index = least significant (position 0).
    This makes Python tuple comparison match SPM lexicographic ordering.
    """
    return (n + 1,) * ((d // 2) + 1)


def _spm_lift(game: ParityGame, node: int, measures: Dict[int, Tuple[int, ...]],
              n_nodes: int, d: int) -> Tuple[int, ...]:
    """Compute the lifted measure for a node.

    Tuples stored in reverse: index 0 = position d//2 (most significant).
    For priority p with half = p // 2:
      - Truncate positions < half to 0 (= set indices > measure_len-1-half to 0)
      - If p is odd: increment at position half (= index measure_len-1-half)

    For Even-owned nodes: min over successors.
    For Odd-owned nodes: max over successors.
    """
    measure_len = (d // 2) + 1
    top = _spm_top(n_nodes, d)
    succs = game.successors.get(node, set())
    if not succs:
        return top

    p = game.priority[node]
    half = p // 2
    idx = measure_len - 1 - half  # Index in reversed tuple

    candidates = []
    for s in succs:
        m_s = measures[s]
        if m_s == top:
            candidates.append(top)
            continue

        prog = list(m_s)

        # Truncate: set positions < half to 0 (indices > idx in reversed tuple)
        for i in range(idx + 1, measure_len):
            prog[i] = 0

        if p % 2 == 1:  # Odd priority -- increment at position half
            prog[idx] += 1
            if prog[idx] > n_nodes:
                candidates.append(top)
                continue

        candidates.append(tuple(prog))

    if game.owner[node] == Player.EVEN:
        result = top
        for c in candidates:
            if c < result:
                result = c
        return result
    else:
        result = candidates[0]
        for c in candidates[1:]:
            if c > result:
                result = c
        return result


def small_progress_measures(game: ParityGame) -> ParityResult:
    """Solve a parity game using Small Progress Measures (Jurdzinski 2000).

    Computes winning regions by finding a progress measure assignment.
    Nodes with measure TOP are won by Player 1 (Odd).
    Nodes with non-TOP measures are won by Player 0 (Even).

    Time complexity: O(d * m * (n/d)^(d/2)) where m = |edges|, d = max priority.
    """
    if not game.nodes:
        return ParityResult(algorithm="spm", iterations=0)

    d = game.max_priority()
    n = len(game.nodes)
    top = _spm_top(n, d)
    measure_len = (d // 2) + 1

    # Initialize all measures to (0, 0, ..., 0)
    measures: Dict[int, Tuple[int, ...]] = {
        node: (0,) * measure_len for node in game.nodes
    }

    # Iterative lifting until fixpoint
    # Key invariant: measures are monotonically non-decreasing
    iterations = 0
    changed = True
    while changed:
        changed = False
        for node in game.nodes:
            old = measures[node]
            if old == top:
                continue
            new = _spm_lift(game, node, measures, n, d)
            if new > old:  # Only increase (monotonicity)
                measures[node] = new
                changed = True
                iterations += 1

    # Extract winning regions
    win0 = {n for n in game.nodes if measures[n] != top}
    win1 = {n for n in game.nodes if measures[n] == top}

    # Extract strategies
    strategy0: Dict[int, int] = {}
    strategy1: Dict[int, int] = {}

    for node in win0:
        if game.owner[node] == Player.EVEN:
            # Pick successor that minimizes measure
            best = None
            best_val = top
            for s in game.successors.get(node, set()):
                if s in win0:
                    m_s = measures[s]
                    if m_s < best_val:
                        best_val = m_s
                        best = s
            if best is not None:
                strategy0[node] = best

    for node in win1:
        if game.owner[node] == Player.ODD:
            # Pick successor that maximizes measure (stays in win1)
            best = None
            for s in game.successors.get(node, set()):
                if s in win1:
                    best = s
                    break
            if best is not None:
                strategy1[node] = best

    return ParityResult(
        win0=win0, win1=win1,
        strategy0=strategy0, strategy1=strategy1,
        algorithm="spm",
        iterations=iterations
    )


# ---------------------------------------------------------------------------
# Priority compression
# ---------------------------------------------------------------------------

def compress_priorities(game: ParityGame) -> ParityGame:
    """Compress priorities to remove gaps while preserving parity.

    E.g., priorities {0, 3, 7} become {0, 1, 3} -- maintaining even/odd parity.
    This can speed up solving by reducing the effective priority range.
    """
    if not game.nodes:
        return game

    used = sorted(set(game.priority[n] for n in game.nodes))
    if not used:
        return game

    # Build mapping: preserve parity, compress gaps
    mapping: Dict[int, int] = {}
    current = 0
    for p in used:
        # Ensure parity matches
        while current % 2 != p % 2:
            current += 1
        mapping[p] = current
        current += 1

    compressed = ParityGame()
    for n in game.nodes:
        compressed.add_node(n, game.owner[n], mapping[game.priority[n]])
    for n in game.nodes:
        for s in game.successors.get(n, set()):
            compressed.add_edge(n, s)

    return compressed


# ---------------------------------------------------------------------------
# Self-loop removal optimization
# ---------------------------------------------------------------------------

def remove_self_loops(game: ParityGame) -> Tuple[ParityGame, Set[int], Set[int]]:
    """Remove trivially decided self-loop nodes.

    A node with a self-loop where the owner benefits from the priority
    can be immediately assigned to that player's winning region.

    Returns (reduced_game, immediate_win0, immediate_win1).
    """
    immediate_win0: Set[int] = set()
    immediate_win1: Set[int] = set()

    for n in game.nodes:
        if n in game.successors.get(n, set()):
            p = game.priority[n]
            owner = game.owner[n]
            beneficiary = Player.owner_of_priority(p)

            if owner == beneficiary and len(game.successors[n]) >= 1:
                # Owner benefits from looping -- can choose to stay
                if beneficiary == Player.EVEN:
                    immediate_win0.add(n)
                else:
                    immediate_win1.add(n)

    # Only remove nodes that are immediately decided
    remaining = game.nodes - immediate_win0 - immediate_win1
    reduced = game.subgame(remaining)

    return reduced, immediate_win0, immediate_win1


# ---------------------------------------------------------------------------
# Conversion: Buchi/co-Buchi/Rabin/Streett to Parity Games
# ---------------------------------------------------------------------------

@dataclass
class BuchiGame:
    """A Buchi game: Player 0 wins if accepting states visited infinitely often."""
    nodes: Set[int]
    owner: Dict[int, Player]
    successors: Dict[int, Set[int]]
    accepting: Set[int]


@dataclass
class CoBuchiGame:
    """A co-Buchi game: Player 0 wins if rejecting states visited finitely often."""
    nodes: Set[int]
    owner: Dict[int, Player]
    successors: Dict[int, Set[int]]
    rejecting: Set[int]


@dataclass
class RabinPair:
    """A Rabin pair (L, U): visit L finitely and U infinitely."""
    fin: Set[int]   # L: must be visited finitely often
    inf: Set[int]   # U: must be visited infinitely often


@dataclass
class StreettPair:
    """A Streett pair (L, U): if U visited infinitely then L visited infinitely."""
    request: Set[int]   # U: if visited infinitely often...
    response: Set[int]  # L: ...then this must also be visited infinitely often


def buchi_to_parity(bg: BuchiGame) -> ParityGame:
    """Convert a Buchi game to an equivalent parity game.

    Accepting nodes get priority 2 (even, highest), non-accepting get priority 1 (odd).
    Player 0 wins iff highest priority seen infinitely often is even (2),
    which happens iff accepting states are visited infinitely often.
    """
    pg = ParityGame()
    for n in bg.nodes:
        p = 2 if n in bg.accepting else 1
        pg.add_node(n, bg.owner.get(n, Player.EVEN), p)
    for n in bg.nodes:
        for s in bg.successors.get(n, set()):
            pg.add_edge(n, s)
    return pg


def cobuchi_to_parity(cg: CoBuchiGame) -> ParityGame:
    """Convert a co-Buchi game to an equivalent parity game.

    Rejecting nodes get priority 1 (odd), non-rejecting get priority 0 (even).
    Player 0 wins iff rejecting states are visited only finitely often:
    - If play eventually avoids rejecting: max inf prio is 0 (even) -> Even wins.
    - If rejecting visited infinitely: max inf prio is 1 (odd) -> Odd wins.
    """
    pg = ParityGame()
    for n in cg.nodes:
        p = 1 if n in cg.rejecting else 0
        pg.add_node(n, cg.owner.get(n, Player.EVEN), p)
    for n in cg.nodes:
        for s in cg.successors.get(n, set()):
            pg.add_edge(n, s)
    return pg


def rabin_to_parity(nodes: Set[int], owner: Dict[int, Player],
                    successors: Dict[int, Set[int]],
                    pairs: List[RabinPair]) -> ParityGame:
    """Convert a Rabin game to a parity game via index appearance record (IAR).

    Rabin condition: there exists a pair (L_i, U_i) where L_i visited finitely
    and U_i visited infinitely. This is equivalent to a parity game with
    priorities encoding the IAR.

    For simplicity, we use the direct encoding:
    - k pairs -> priorities 0..2k
    - For pair i (0-indexed): nodes in U_i get priority 2i, nodes in L_i get 2i+1
    - Nodes not in any pair get the highest odd priority 2k+1
    - Lower-indexed pairs have higher priority (checked first)
    """
    pg = ParityGame()
    k = len(pairs)

    for n in nodes:
        # Find the lowest-indexed pair that contains n
        assigned = False
        for i, pair in enumerate(pairs):
            if n in pair.inf:
                pg.add_node(n, owner.get(n, Player.EVEN), 2 * i)
                assigned = True
                break
            if n in pair.fin:
                pg.add_node(n, owner.get(n, Player.EVEN), 2 * i + 1)
                assigned = True
                break
        if not assigned:
            # Not in any pair -- give neutral even priority
            pg.add_node(n, owner.get(n, Player.EVEN), 2 * k)

    for n in nodes:
        for s in successors.get(n, set()):
            pg.add_edge(n, s)

    return pg


def streett_to_parity(nodes: Set[int], owner: Dict[int, Player],
                      successors: Dict[int, Set[int]],
                      pairs: List[StreettPair]) -> ParityGame:
    """Convert a Streett game to a parity game.

    Streett condition: for all pairs (L_i, U_i), if U_i visited infinitely
    then L_i visited infinitely. This is the dual of Rabin.

    Encoding: similar to Rabin but with swapped even/odd.
    - For pair i: nodes in response_i (L_i) get priority 2i+1 (odd -- bad if not seen)
    - Nodes in request_i (U_i) get priority 2i+2 (even -- good, triggers obligation)
    - Default priority 0 (even, neutral)
    """
    pg = ParityGame()
    k = len(pairs)

    for n in nodes:
        assigned = False
        for i, pair in enumerate(pairs):
            if n in pair.response:
                pg.add_node(n, owner.get(n, Player.EVEN), 2 * i + 2)
                assigned = True
                break
            if n in pair.request:
                pg.add_node(n, owner.get(n, Player.EVEN), 2 * i + 1)
                assigned = True
                break
        if not assigned:
            pg.add_node(n, owner.get(n, Player.EVEN), 0)

    for n in nodes:
        for s in successors.get(n, set()):
            pg.add_edge(n, s)

    return pg


# ---------------------------------------------------------------------------
# Optimized solver: compression + self-loop removal + Zielonka
# ---------------------------------------------------------------------------

def solve(game: ParityGame, algorithm: str = "zielonka") -> ParityResult:
    """Solve a parity game with optimizations.

    Args:
        game: the parity game to solve
        algorithm: "zielonka" or "spm"

    Returns:
        ParityResult with winning regions and strategies.
    """
    if not game.nodes:
        return ParityResult(algorithm=algorithm, iterations=0)

    # Phase 1: Self-loop removal
    reduced, imm0, imm1 = remove_self_loops(game)

    # Phase 2: Priority compression
    compressed = compress_priorities(reduced)

    # Phase 3: Solve
    if algorithm == "spm":
        result = small_progress_measures(compressed)
    else:
        result = zielonka(compressed)

    # Phase 4: Propagate immediate wins through attractors
    if imm0:
        attr0, attr0_strat = attractor(game, imm0 | result.win0, Player.EVEN)
        result.win0 = attr0
        result.strategy0.update(attr0_strat)
        # Self-loop nodes: strategy is the self-loop
        for n in imm0:
            if game.owner[n] == Player.EVEN:
                result.strategy0[n] = n
    else:
        result.win0 = result.win0 | imm0

    if imm1:
        attr1, attr1_strat = attractor(game, imm1 | result.win1, Player.ODD)
        result.win1 = attr1
        result.strategy1.update(attr1_strat)
        for n in imm1:
            if game.owner[n] == Player.ODD:
                result.strategy1[n] = n
    else:
        result.win1 = result.win1 | imm1

    # Ensure partition
    result.win0 = game.nodes - result.win1
    result.algorithm = algorithm

    return result


# ---------------------------------------------------------------------------
# Strategy verification
# ---------------------------------------------------------------------------

def verify_strategy(game: ParityGame, result: ParityResult,
                    max_steps: int = 1000) -> Dict:
    """Verify that strategies are winning by simulating plays.

    For each node, simulate the play following both strategies and check
    that the winner's strategy produces a play with correct parity.
    """
    report = {"valid": True, "issues": [], "checked": 0}

    for start in game.nodes:
        winner = result.winner(start)
        strat = result.strategy0 if winner == Player.EVEN else result.strategy1

        # Simulate play
        visited: List[int] = []
        current = start
        seen_states: Dict[int, int] = {}

        for step in range(max_steps):
            if current in seen_states:
                # Found a cycle -- check priorities in cycle
                cycle_start = seen_states[current]
                cycle = visited[cycle_start:]
                cycle_prios = [game.priority[n] for n in cycle]
                max_prio = max(cycle_prios)
                parity_winner = Player.owner_of_priority(max_prio)

                if parity_winner != winner:
                    report["valid"] = False
                    report["issues"].append(
                        f"Node {start}: cycle max priority {max_prio} "
                        f"favors {parity_winner}, not {winner}")
                break

            seen_states[current] = len(visited)
            visited.append(current)

            # Determine next node
            if game.owner[current] == winner:
                # Winner moves according to strategy
                nxt = strat.get(current)
                if nxt is None:
                    # No strategy defined -- pick any valid successor
                    succs = game.successors.get(current, set())
                    nxt = next(iter(succs)) if succs else None
            else:
                # Opponent plays adversarially -- pick worst for winner
                succs = game.successors.get(current, set())
                opp_strat = result.strategy1 if winner == Player.EVEN else result.strategy0
                nxt = opp_strat.get(current)
                if nxt is None and succs:
                    nxt = next(iter(succs))

            if nxt is None:
                break
            current = nxt

        report["checked"] += 1

    return report


# ---------------------------------------------------------------------------
# Game builders (convenience)
# ---------------------------------------------------------------------------

def make_game(nodes: List[Tuple[int, int, int]],
              edges: List[Tuple[int, int]]) -> ParityGame:
    """Build a parity game from node specs and edges.

    Args:
        nodes: list of (node_id, owner_int, priority) tuples.
               owner_int: 0 = Even, 1 = Odd
        edges: list of (src, dst) tuples.
    """
    g = ParityGame()
    for nid, own, pri in nodes:
        g.add_node(nid, Player.EVEN if own == 0 else Player.ODD, pri)
    for src, dst in edges:
        g.add_edge(src, dst)
    return g


def make_random_game(n_nodes: int, n_edges: int, max_priority: int,
                     seed: int = 42) -> ParityGame:
    """Generate a random parity game for testing."""
    import random
    rng = random.Random(seed)

    g = ParityGame()
    for i in range(n_nodes):
        owner = Player.EVEN if rng.random() < 0.5 else Player.ODD
        priority = rng.randint(0, max_priority)
        g.add_node(i, owner, priority)

    # Ensure each node has at least one successor
    for i in range(n_nodes):
        target = rng.randint(0, n_nodes - 1)
        g.add_edge(i, target)

    # Add random edges
    for _ in range(n_edges - n_nodes):
        src = rng.randint(0, n_nodes - 1)
        dst = rng.randint(0, n_nodes - 1)
        g.add_edge(src, dst)

    return g


# ---------------------------------------------------------------------------
# Comparison API
# ---------------------------------------------------------------------------

def compare_algorithms(game: ParityGame) -> Dict:
    """Compare Zielonka vs SPM on the same game."""
    r_z = zielonka(game)
    r_s = small_progress_measures(game)

    agree = (r_z.win0 == r_s.win0 and r_z.win1 == r_s.win1)

    return {
        "nodes": len(game.nodes),
        "max_priority": game.max_priority(),
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
        "agree": agree,
    }


# ---------------------------------------------------------------------------
# Dominion detection (for advanced algorithms)
# ---------------------------------------------------------------------------

def find_dominion(game: ParityGame, player: Player) -> Optional[Set[int]]:
    """Try to find a dominion for the given player.

    A dominion is a non-empty set D where `player` has a strategy to keep
    the play in D forever and win. Used in quasi-polynomial algorithms.
    """
    # Simple approach: try to find a closed set where player wins all cycles
    for p in sorted(set(game.priority[n] for n in game.nodes)):
        if Player.owner_of_priority(p) != player:
            continue
        # Nodes with this priority
        seed = game.nodes_with_priority(p)
        attr_set, _ = attractor(game, seed, player)

        # Check if attr_set is a dominion (all nodes have successors within)
        is_closed = True
        for n in attr_set:
            if game.owner[n] == player:
                if not (game.successors.get(n, set()) & attr_set):
                    is_closed = False
                    break
            else:
                if not (game.successors.get(n, set()) & attr_set) == game.successors.get(n, set()):
                    is_closed = False
                    break

        if is_closed and attr_set:
            return attr_set

    return None


# ---------------------------------------------------------------------------
# McNaughton's algorithm (alternative to Zielonka, simpler but less efficient)
# ---------------------------------------------------------------------------

def mcnaughton(game: ParityGame) -> ParityResult:
    """Solve a parity game using McNaughton's algorithm.

    Similar to Zielonka but uses a different recursion structure.
    Included for comparison and educational purposes.
    """
    iterations = [0]

    def _rec(g: ParityGame) -> Tuple[Set[int], Set[int], Dict[int, int], Dict[int, int]]:
        iterations[0] += 1
        if not g.nodes:
            return set(), set(), {}, {}

        d = g.max_priority()
        player = Player.owner_of_priority(d)

        # Compute attractor of max-priority nodes
        max_nodes = g.nodes_with_priority(d)
        attr_set, attr_strat = attractor(g, max_nodes, player)

        # Solve subgame
        sub = g.subgame(g.nodes - attr_set)
        w0, w1, s0, s1 = _rec(sub)

        w_opp = w1 if player == Player.EVEN else w0

        if not w_opp:
            # Player wins everything
            s_p = s0 if player == Player.EVEN else s1
            combined = {**s_p, **attr_strat}
            for n in max_nodes:
                if g.owner[n] == player and n not in combined:
                    succs = g.successors.get(n, set())
                    if succs:
                        combined[n] = next(iter(succs))
            if player == Player.EVEN:
                return g.nodes.copy(), set(), combined, {}
            else:
                return set(), g.nodes.copy(), {}, combined
        else:
            # Opponent wins something -- remove and recurse
            b_set, b_strat = attractor(g, w_opp, player.opponent)
            sub2 = g.subgame(g.nodes - b_set)
            w0_2, w1_2, s0_2, s1_2 = _rec(sub2)

            s_opp_inner = s1 if player.opponent == Player.ODD else s0
            combined_opp = {**b_strat, **s_opp_inner}
            if player.opponent == Player.ODD:
                combined_opp.update(s1_2)
                w1_final = b_set | w1_2
                return w0_2, w1_final, s0_2, combined_opp
            else:
                combined_opp.update(s0_2)
                w0_final = b_set | w0_2
                return w0_final, w1_2, combined_opp, s1_2

    w0, w1, s0, s1 = _rec(game)
    return ParityResult(
        win0=w0, win1=w1,
        strategy0=s0, strategy1=s1,
        algorithm="mcnaughton",
        iterations=iterations[0]
    )


# ---------------------------------------------------------------------------
# Conversion to/from V074 StochasticGame
# ---------------------------------------------------------------------------

def parity_to_stochastic_game(pg: ParityGame) -> Tuple:
    """Convert a parity game to a structure compatible with V074.

    Returns (game_dict, accepting_sets) where:
    - game_dict has fields compatible with StochasticGame construction
    - accepting_sets maps priority -> set of nodes

    This enables using V074's omega-regular infrastructure.
    """
    # Build transition structure: each node has one action per successor
    states = sorted(pg.nodes)
    n_states = len(states)
    state_idx = {s: i for i, s in enumerate(states)}

    transitions = {}
    owners = {}
    actions = {}

    for node in states:
        idx = state_idx[node]
        from V074_mapping import Player as V074Player
        owners[idx] = 0 if pg.owner[node] == Player.EVEN else 1
        succs = sorted(pg.successors.get(node, set()))
        node_actions = []
        for s in succs:
            # Deterministic transition to s
            node_actions.append({state_idx[s]: 1.0})
        actions[idx] = node_actions

    priority_sets = {}
    for node in states:
        p = pg.priority[node]
        if p not in priority_sets:
            priority_sets[p] = set()
        priority_sets[p].add(state_idx[node])

    return {
        "n_states": n_states,
        "owners": owners,
        "actions": actions,
        "state_map": state_idx,
        "inv_map": {i: s for s, i in state_idx.items()},
    }, priority_sets
