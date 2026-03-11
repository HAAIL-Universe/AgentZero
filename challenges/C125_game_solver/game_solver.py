"""
C125: Game Solver
Minimax, Alpha-Beta Pruning, Monte Carlo Tree Search, Iterative Deepening,
Transposition Tables, Move Ordering, and game abstractions.

Standalone challenge -- no dependencies on previous challenges.
"""

import math
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional


# ============================================================
# Game Abstraction
# ============================================================

class GameState(ABC):
    """Abstract base class for game states."""

    @abstractmethod
    def get_moves(self):
        """Return list of legal moves."""
        pass

    @abstractmethod
    def make_move(self, move):
        """Return new GameState after applying move."""
        pass

    @abstractmethod
    def is_terminal(self):
        """Return True if game is over."""
        pass

    @abstractmethod
    def get_utility(self, player):
        """Return utility value for given player. Only valid at terminal states."""
        pass

    @abstractmethod
    def get_current_player(self):
        """Return the player whose turn it is."""
        pass

    def evaluate(self, player):
        """Heuristic evaluation for non-terminal states. Default: 0."""
        return 0


# ============================================================
# Tic-Tac-Toe (for testing)
# ============================================================

class TicTacToe(GameState):
    """Standard 3x3 Tic-Tac-Toe."""

    def __init__(self, board=None, current_player=1):
        self.board = board if board is not None else [0] * 9
        self.current = current_player

    def get_moves(self):
        if self.is_terminal():
            return []
        return [i for i in range(9) if self.board[i] == 0]

    def make_move(self, move):
        new_board = self.board[:]
        new_board[move] = self.current
        return TicTacToe(new_board, 3 - self.current)

    def is_terminal(self):
        return self._winner() is not None or all(c != 0 for c in self.board)

    def _winner(self):
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        ]
        for a, b, c in lines:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def get_utility(self, player):
        w = self._winner()
        if w is None:
            return 0  # draw
        return 1.0 if w == player else -1.0

    def get_current_player(self):
        return self.current

    def evaluate(self, player):
        # Simple heuristic: count lines still winnable
        score = 0
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        ]
        for a, b, c in lines:
            vals = [self.board[a], self.board[b], self.board[c]]
            opp = 3 - player
            if opp not in vals:
                score += 0.1
            if player not in vals:
                score -= 0.1
        return score

    def __hash__(self):
        return hash((tuple(self.board), self.current))

    def __eq__(self, other):
        return isinstance(other, TicTacToe) and self.board == other.board and self.current == other.current


# ============================================================
# Connect Four (for testing deeper games)
# ============================================================

class ConnectFour(GameState):
    """Standard 7x6 Connect Four."""

    ROWS = 6
    COLS = 7

    def __init__(self, board=None, current_player=1):
        if board is not None:
            self.board = [col[:] for col in board]
        else:
            self.board = [[] for _ in range(self.COLS)]
        self.current = current_player

    def get_moves(self):
        if self.is_terminal():
            return []
        return [c for c in range(self.COLS) if len(self.board[c]) < self.ROWS]

    def make_move(self, move):
        new_board = [col[:] for col in self.board]
        new_board[move].append(self.current)
        return ConnectFour(new_board, 3 - self.current)

    def is_terminal(self):
        return self._winner() is not None or all(len(col) >= self.ROWS for col in self.board)

    def _winner(self):
        # Check all directions
        for c in range(self.COLS):
            for r in range(len(self.board[c])):
                p = self.board[c][r]
                if p == 0:
                    continue
                # horizontal
                if c + 3 < self.COLS and all(
                    r < len(self.board[c + d]) and self.board[c + d][r] == p for d in range(4)
                ):
                    return p
                # vertical
                if r + 3 < len(self.board[c]) and all(self.board[c][r + d] == p for d in range(4)):
                    return p
                # diagonal up-right
                if c + 3 < self.COLS and all(
                    (r + d) < len(self.board[c + d]) and self.board[c + d][r + d] == p for d in range(4)
                ):
                    return p
                # diagonal down-right
                if c + 3 < self.COLS and all(
                    (r - d) >= 0 and (r - d) < len(self.board[c + d]) and self.board[c + d][r - d] == p
                    for d in range(4)
                ):
                    return p
        return None

    def get_utility(self, player):
        w = self._winner()
        if w is None:
            return 0.0
        return 1.0 if w == player else -1.0

    def get_current_player(self):
        return self.current

    def evaluate(self, player):
        """Heuristic: count threats (3-in-a-row with open space)."""
        score = 0.0
        opp = 3 - player
        for c in range(self.COLS):
            for r in range(len(self.board[c])):
                p = self.board[c][r]
                if p == 0:
                    continue
                # Check partial lines
                directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
                for dc, dr in directions:
                    count = 0
                    empty = 0
                    for d in range(4):
                        nc, nr = c + dc * d, r + dr * d
                        if 0 <= nc < self.COLS and 0 <= nr < self.ROWS:
                            if nr < len(self.board[nc]):
                                if self.board[nc][nr] == p:
                                    count += 1
                                elif self.board[nc][nr] == 0:
                                    empty += 1
                                else:
                                    break
                            else:
                                empty += 1
                        else:
                            break
                    if count + empty >= 4:
                        weight = count * 0.01
                        if p == player:
                            score += weight
                        else:
                            score -= weight
        return score

    def __hash__(self):
        return hash((tuple(tuple(col) for col in self.board), self.current))

    def __eq__(self, other):
        return isinstance(other, ConnectFour) and self.board == other.board and self.current == other.current


# ============================================================
# Nim (for testing)
# ============================================================

class Nim(GameState):
    """Nim: take 1-3 items from pile. Last to take loses."""

    def __init__(self, pile=10, current_player=1):
        self.pile = pile
        self.current = current_player

    def get_moves(self):
        if self.is_terminal():
            return []
        return [i for i in range(1, min(3, self.pile) + 1)]

    def make_move(self, move):
        return Nim(self.pile - move, 3 - self.current)

    def is_terminal(self):
        return self.pile <= 0

    def get_utility(self, player):
        # Last player to take loses -- current player is the one who faces empty pile
        # So the previous player (who took the last item) wins
        if self.pile <= 0:
            # Current player has no move, so they lose
            return -1.0 if player == self.current else 1.0
        return 0.0

    def get_current_player(self):
        return self.current

    def __hash__(self):
        return hash((self.pile, self.current))

    def __eq__(self, other):
        return isinstance(other, Nim) and self.pile == other.pile and self.current == other.current


# ============================================================
# Minimax
# ============================================================

def minimax(state, player=None, max_depth=None):
    """
    Pure minimax search. Returns (best_value, best_move).
    player: the maximizing player (default: current player).
    max_depth: depth limit (None = unlimited).
    """
    if player is None:
        player = state.get_current_player()

    def _minimax(s, depth):
        if s.is_terminal():
            return s.get_utility(player), None
        if max_depth is not None and depth >= max_depth:
            return s.evaluate(player), None

        moves = s.get_moves()
        is_maximizing = (s.get_current_player() == player)

        best_move = moves[0]
        if is_maximizing:
            best_val = -math.inf
            for m in moves:
                val, _ = _minimax(s.make_move(m), depth + 1)
                if val > best_val:
                    best_val = val
                    best_move = m
        else:
            best_val = math.inf
            for m in moves:
                val, _ = _minimax(s.make_move(m), depth + 1)
                if val < best_val:
                    best_val = val
                    best_move = m

        return best_val, best_move

    return _minimax(state, 0)


# ============================================================
# Alpha-Beta Pruning
# ============================================================

def alpha_beta(state, player=None, max_depth=None):
    """
    Alpha-beta pruning search. Returns (best_value, best_move).
    """
    if player is None:
        player = state.get_current_player()

    def _ab(s, depth, alpha, beta):
        if s.is_terminal():
            return s.get_utility(player), None
        if max_depth is not None and depth >= max_depth:
            return s.evaluate(player), None

        moves = s.get_moves()
        is_maximizing = (s.get_current_player() == player)

        best_move = moves[0]
        if is_maximizing:
            best_val = -math.inf
            for m in moves:
                val, _ = _ab(s.make_move(m), depth + 1, alpha, beta)
                if val > best_val:
                    best_val = val
                    best_move = m
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
        else:
            best_val = math.inf
            for m in moves:
                val, _ = _ab(s.make_move(m), depth + 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_move = m
                beta = min(beta, val)
                if alpha >= beta:
                    break

        return best_val, best_move

    return _ab(state, 0, -math.inf, math.inf)


# ============================================================
# Transposition Table
# ============================================================

class TranspositionTable:
    """Hash table for storing evaluated positions."""

    EXACT = 0
    LOWER = 1  # alpha cutoff
    UPPER = 2  # beta cutoff

    def __init__(self, max_size=1_000_000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.stores = 0

    def lookup(self, key, depth, alpha, beta):
        """
        Look up position. Returns (found, value, best_move).
        Only returns a usable value if the stored depth >= requested depth.
        """
        entry = self.table.get(key)
        if entry is None:
            return False, 0, None

        stored_depth, stored_val, stored_flag, stored_move = entry
        if stored_depth >= depth:
            self.hits += 1
            if stored_flag == self.EXACT:
                return True, stored_val, stored_move
            elif stored_flag == self.LOWER and stored_val >= beta:
                return True, stored_val, stored_move
            elif stored_flag == self.UPPER and stored_val <= alpha:
                return True, stored_val, stored_move

        return False, 0, stored_move  # return move for ordering even if value unusable

    def store(self, key, depth, value, flag, best_move):
        """Store position evaluation."""
        if len(self.table) >= self.max_size:
            # Simple replacement: always replace
            pass
        self.table[key] = (depth, value, flag, best_move)
        self.stores += 1

    def clear(self):
        self.table.clear()
        self.hits = 0
        self.stores = 0


def alpha_beta_tt(state, player=None, max_depth=None, tt=None):
    """
    Alpha-beta with transposition table. Returns (best_value, best_move).
    """
    if player is None:
        player = state.get_current_player()
    if tt is None:
        tt = TranspositionTable()

    def _ab_tt(s, depth, alpha, beta):
        alpha_orig = alpha

        # TT lookup
        key = hash(s)
        found, tt_val, tt_move = tt.lookup(key, max_depth - depth if max_depth else 0, alpha, beta)
        if found:
            return tt_val, tt_move

        if s.is_terminal():
            return s.get_utility(player), None
        if max_depth is not None and depth >= max_depth:
            return s.evaluate(player), None

        moves = s.get_moves()
        # Move ordering: try TT move first
        if tt_move is not None and tt_move in moves:
            moves = [tt_move] + [m for m in moves if m != tt_move]

        is_maximizing = (s.get_current_player() == player)
        best_move = moves[0]

        if is_maximizing:
            best_val = -math.inf
            for m in moves:
                val, _ = _ab_tt(s.make_move(m), depth + 1, alpha, beta)
                if val > best_val:
                    best_val = val
                    best_move = m
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
        else:
            best_val = math.inf
            for m in moves:
                val, _ = _ab_tt(s.make_move(m), depth + 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_move = m
                beta = min(beta, val)
                if alpha >= beta:
                    break

        # Store in TT
        remaining = (max_depth - depth) if max_depth else depth
        if best_val <= alpha_orig:
            flag = TranspositionTable.UPPER
        elif best_val >= beta:
            flag = TranspositionTable.LOWER
        else:
            flag = TranspositionTable.EXACT
        tt.store(key, remaining, best_val, flag, best_move)

        return best_val, best_move

    return _ab_tt(state, 0, -math.inf, math.inf)


# ============================================================
# Iterative Deepening
# ============================================================

def iterative_deepening(state, player=None, max_depth=20, time_limit=None):
    """
    Iterative deepening alpha-beta. Returns (best_value, best_move, depth_reached).
    Searches deeper each iteration, optionally stopping at time_limit seconds.
    """
    if player is None:
        player = state.get_current_player()

    tt = TranspositionTable()
    best_val = 0
    best_move = None
    depth_reached = 0
    start_time = time.time()

    for depth in range(1, max_depth + 1):
        if time_limit and (time.time() - start_time) > time_limit:
            break

        val, move = alpha_beta_tt(state, player=player, max_depth=depth, tt=tt)
        best_val = val
        best_move = move
        depth_reached = depth

        # If we found a winning move, stop
        if abs(val) >= 1.0:
            break

    return best_val, best_move, depth_reached


# ============================================================
# Negamax (alternative formulation)
# ============================================================

def negamax(state, player=None, max_depth=None):
    """
    Negamax formulation. Returns (best_value, best_move).
    Value is from perspective of current player at each node.
    """
    if player is None:
        player = state.get_current_player()

    def _negamax(s, depth, alpha, beta, color):
        if s.is_terminal():
            return color * s.get_utility(player), None
        if max_depth is not None and depth >= max_depth:
            return color * s.evaluate(player), None

        moves = s.get_moves()
        best_move = moves[0]
        best_val = -math.inf

        for m in moves:
            val, _ = _negamax(s.make_move(m), depth + 1, -beta, -alpha, -color)
            val = -val
            if val > best_val:
                best_val = val
                best_move = m
            alpha = max(alpha, val)
            if alpha >= beta:
                break

        return best_val, best_move

    color = 1  # maximizing player
    return _negamax(state, 0, -math.inf, math.inf, color)


# ============================================================
# Monte Carlo Tree Search
# ============================================================

class MCTSNode:
    """Node in Monte Carlo search tree."""

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = list(state.get_moves())
        self.player = state.get_current_player()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.state.is_terminal()

    def ucb1(self, exploration=1.41):
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return math.inf
        exploit = self.wins / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self, exploration=1.41):
        """Select child with highest UCB1."""
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def expand(self):
        """Expand one untried move."""
        move = self.untried_moves.pop()
        new_state = self.state.make_move(move)
        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result):
        """Backpropagate result."""
        self.visits += 1
        self.wins += result


def mcts(state, player=None, iterations=1000, exploration=1.41, time_limit=None):
    """
    Monte Carlo Tree Search. Returns (best_move, root_node).
    """
    if player is None:
        player = state.get_current_player()

    root = MCTSNode(state)
    start_time = time.time()

    for i in range(iterations):
        if time_limit and (time.time() - start_time) > time_limit:
            break

        # Selection
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(exploration)

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # Simulation (random playout)
        sim_state = node.state
        while not sim_state.is_terminal():
            moves = sim_state.get_moves()
            sim_state = sim_state.make_move(random.choice(moves))

        # Backpropagation
        result = sim_state.get_utility(player)
        # Convert: 1.0 win -> 1, 0.0 draw -> 0.5, -1.0 loss -> 0
        result_normalized = (result + 1.0) / 2.0
        while node is not None:
            # Store wins from parent's perspective (so UCB1 selects correctly)
            # node.player is who is TO MOVE; parent chose this node
            # If node.player != player, parent IS player -> store win
            # If node.player == player, parent is opponent -> store loss
            if node.player != player:
                node.update(result_normalized)
            else:
                node.update(1.0 - result_normalized)
            node = node.parent

    # Pick most visited child
    if not root.children:
        return None, root

    best = max(root.children, key=lambda c: c.visits)
    return best.move, root


# ============================================================
# Principal Variation Search (PVS / NegaScout)
# ============================================================

def pvs(state, player=None, max_depth=None):
    """
    Principal Variation Search (NegaScout). Returns (best_value, best_move).
    Assumes first move is likely best (from move ordering / TT).
    """
    if player is None:
        player = state.get_current_player()

    def _pvs(s, depth, alpha, beta):
        if s.is_terminal():
            return s.get_utility(player), None
        if max_depth is not None and depth >= max_depth:
            return s.evaluate(player), None

        moves = s.get_moves()
        is_maximizing = (s.get_current_player() == player)

        best_move = moves[0]

        if is_maximizing:
            # Search first move with full window
            val, _ = _pvs(s.make_move(moves[0]), depth + 1, alpha, beta)
            best_val = val
            alpha = max(alpha, val)

            for m in moves[1:]:
                if alpha >= beta:
                    break
                # Null window search
                val, _ = _pvs(s.make_move(m), depth + 1, alpha, alpha + 0.001)
                if val > alpha and val < beta:
                    # Re-search with full window
                    val, _ = _pvs(s.make_move(m), depth + 1, val, beta)
                if val > best_val:
                    best_val = val
                    best_move = m
                alpha = max(alpha, val)
        else:
            val, _ = _pvs(s.make_move(moves[0]), depth + 1, alpha, beta)
            best_val = val
            beta = min(beta, val)

            for m in moves[1:]:
                if alpha >= beta:
                    break
                val, _ = _pvs(s.make_move(m), depth + 1, beta - 0.001, beta)
                if val < beta and val > alpha:
                    val, _ = _pvs(s.make_move(m), depth + 1, alpha, val)
                if val < best_val:
                    best_val = val
                    best_move = m
                beta = min(beta, val)

        return best_val, best_move

    return _pvs(state, 0, -math.inf, math.inf)


# ============================================================
# Expectimax (for games with chance)
# ============================================================

class ChanceGameState(GameState):
    """Game state that can have chance nodes."""

    def is_chance_node(self):
        """Return True if this is a chance node (dice roll, card draw, etc.)."""
        return False

    def get_chance_outcomes(self):
        """Return list of (probability, move) pairs."""
        return []


def expectimax(state, player=None, max_depth=None):
    """
    Expectimax for games with chance nodes. Returns (expected_value, best_move).
    """
    if player is None:
        player = state.get_current_player()

    def _expectimax(s, depth):
        if s.is_terminal():
            return s.get_utility(player), None
        if max_depth is not None and depth >= max_depth:
            return s.evaluate(player), None

        if isinstance(s, ChanceGameState) and s.is_chance_node():
            # Chance node: compute expected value
            outcomes = s.get_chance_outcomes()
            expected = 0.0
            for prob, move in outcomes:
                val, _ = _expectimax(s.make_move(move), depth + 1)
                expected += prob * val
            return expected, None

        moves = s.get_moves()
        is_maximizing = (s.get_current_player() == player)
        best_move = moves[0]

        if is_maximizing:
            best_val = -math.inf
            for m in moves:
                val, _ = _expectimax(s.make_move(m), depth + 1)
                if val > best_val:
                    best_val = val
                    best_move = m
        else:
            best_val = math.inf
            for m in moves:
                val, _ = _expectimax(s.make_move(m), depth + 1)
                if val < best_val:
                    best_val = val
                    best_move = m

        return best_val, best_move

    return _expectimax(state, 0)


# ============================================================
# Killer Move Heuristic + History Heuristic
# ============================================================

class MoveOrderer:
    """Move ordering with killer moves and history heuristic."""

    def __init__(self):
        self.killers = defaultdict(list)  # depth -> [move, move]
        self.history = defaultdict(int)   # move -> score
        self.max_killers = 2

    def record_killer(self, depth, move):
        """Record a move that caused a beta cutoff."""
        killers = self.killers[depth]
        if move not in killers:
            killers.insert(0, move)
            if len(killers) > self.max_killers:
                killers.pop()

    def record_history(self, move, depth):
        """Record history score for a move that improved alpha."""
        self.history[move] += depth * depth

    def order_moves(self, moves, depth, tt_move=None):
        """Order moves: TT move first, then killers, then by history."""
        scored = []
        for m in moves:
            score = 0
            if m == tt_move:
                score = 1_000_000
            elif m in self.killers.get(depth, []):
                score = 100_000
            else:
                score = self.history.get(m, 0)
            scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]


def alpha_beta_ordered(state, player=None, max_depth=None):
    """
    Alpha-beta with move ordering (killer + history heuristics).
    Returns (best_value, best_move, stats).
    """
    if player is None:
        player = state.get_current_player()

    orderer = MoveOrderer()
    nodes_searched = [0]

    def _ab(s, depth, alpha, beta):
        nodes_searched[0] += 1

        if s.is_terminal():
            return s.get_utility(player), None
        if max_depth is not None and depth >= max_depth:
            return s.evaluate(player), None

        moves = orderer.order_moves(s.get_moves(), depth)
        is_maximizing = (s.get_current_player() == player)
        best_move = moves[0]

        if is_maximizing:
            best_val = -math.inf
            for m in moves:
                val, _ = _ab(s.make_move(m), depth + 1, alpha, beta)
                if val > best_val:
                    best_val = val
                    best_move = m
                    orderer.record_history(m, max_depth - depth if max_depth else 1)
                alpha = max(alpha, val)
                if alpha >= beta:
                    orderer.record_killer(depth, m)
                    break
        else:
            best_val = math.inf
            for m in moves:
                val, _ = _ab(s.make_move(m), depth + 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_move = m
                beta = min(beta, val)
                if alpha >= beta:
                    orderer.record_killer(depth, m)
                    break

        return best_val, best_move

    val, move = _ab(state, 0, -math.inf, math.inf)
    return val, move, {"nodes_searched": nodes_searched[0]}


# ============================================================
# Game Tree Statistics
# ============================================================

def count_nodes(state, max_depth=None):
    """Count total nodes in game tree up to max_depth."""
    count = [0]

    def _count(s, depth):
        count[0] += 1
        if s.is_terminal():
            return
        if max_depth is not None and depth >= max_depth:
            return
        for m in s.get_moves():
            _count(s.make_move(m), depth + 1)

    _count(state, 0)
    return count[0]


def solve_game(state, player=None):
    """
    Fully solve a game (find game-theoretic value).
    Returns (value, best_move) with perfect play from both sides.
    """
    return alpha_beta(state, player=player)


# ============================================================
# Multi-player Game Support
# ============================================================

class MultiPlayerGameState(ABC):
    """Abstract base for games with more than 2 players."""

    @abstractmethod
    def get_moves(self):
        pass

    @abstractmethod
    def make_move(self, move):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def get_utilities(self):
        """Return dict mapping player -> utility at terminal state."""
        pass

    @abstractmethod
    def get_current_player(self):
        pass

    @abstractmethod
    def get_players(self):
        """Return list of all players."""
        pass

    def evaluate_all(self):
        """Heuristic evaluation for all players. Returns dict player -> value."""
        return {p: 0 for p in self.get_players()}


def maxn(state, max_depth=None):
    """
    Max^n algorithm for multi-player games.
    Returns (utilities_dict, best_move).
    Each player maximizes their own utility.
    """

    def _maxn(s, depth):
        if s.is_terminal():
            return s.get_utilities(), None
        if max_depth is not None and depth >= max_depth:
            return s.evaluate_all(), None

        current = s.get_current_player()
        moves = s.get_moves()
        best_move = moves[0]
        best_utils = None

        for m in moves:
            utils, _ = _maxn(s.make_move(m), depth + 1)
            if best_utils is None or utils[current] > best_utils[current]:
                best_utils = utils
                best_move = m

        return best_utils, best_move

    return _maxn(state, 0)


# ============================================================
# Paranoid Search (multi-player -> 2-player reduction)
# ============================================================

def paranoid(state, player=None, max_depth=None):
    """
    Paranoid search for multi-player games.
    Assumes all opponents cooperate against the maximizing player.
    Returns (value, best_move).
    """
    if player is None:
        player = state.get_current_player()

    def _paranoid(s, depth, alpha, beta):
        if s.is_terminal():
            return s.get_utilities()[player], None
        if max_depth is not None and depth >= max_depth:
            return s.evaluate_all()[player], None

        moves = s.get_moves()
        is_maximizing = (s.get_current_player() == player)
        best_move = moves[0]

        if is_maximizing:
            best_val = -math.inf
            for m in moves:
                val, _ = _paranoid(s.make_move(m), depth + 1, alpha, beta)
                if val > best_val:
                    best_val = val
                    best_move = m
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
        else:
            best_val = math.inf
            for m in moves:
                val, _ = _paranoid(s.make_move(m), depth + 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_move = m
                beta = min(beta, val)
                if alpha >= beta:
                    break

        return best_val, best_move

    return _paranoid(state, 0, -math.inf, math.inf)


# ============================================================
# Aspiration Windows
# ============================================================

def aspiration_search(state, player=None, max_depth=10, initial_window=0.5):
    """
    Iterative deepening with aspiration windows.
    Returns (best_value, best_move, depth_reached).
    """
    if player is None:
        player = state.get_current_player()

    tt = TranspositionTable()
    best_val = 0
    best_move = None
    depth_reached = 0

    for depth in range(1, max_depth + 1):
        if depth == 1:
            val, move = alpha_beta_tt(state, player=player, max_depth=depth, tt=tt)
        else:
            window = initial_window
            alpha = best_val - window
            beta = best_val + window

            val, move = _aspiration_ab(state, player, depth, alpha, beta, tt)

            # Widen window if search fell outside
            while val <= alpha or val >= beta:
                window *= 2
                if window > 10:
                    # Full window search
                    val, move = alpha_beta_tt(state, player=player, max_depth=depth, tt=tt)
                    break
                alpha = best_val - window
                beta = best_val + window
                val, move = _aspiration_ab(state, player, depth, alpha, beta, tt)

        best_val = val
        best_move = move
        depth_reached = depth

        if abs(val) >= 1.0:
            break

    return best_val, best_move, depth_reached


def _aspiration_ab(state, player, max_depth, alpha, beta, tt):
    """Helper for aspiration window search."""

    def _ab(s, depth, alpha, beta):
        key = hash(s)
        found, tt_val, tt_move = tt.lookup(key, max_depth - depth, alpha, beta)
        if found:
            return tt_val, tt_move

        if s.is_terminal():
            return s.get_utility(player), None
        if depth >= max_depth:
            return s.evaluate(player), None

        alpha_orig = alpha
        moves = s.get_moves()
        if tt_move is not None and tt_move in moves:
            moves = [tt_move] + [m for m in moves if m != tt_move]

        is_maximizing = (s.get_current_player() == player)
        best_move = moves[0]

        if is_maximizing:
            best_val = -math.inf
            for m in moves:
                val, _ = _ab(s.make_move(m), depth + 1, alpha, beta)
                if val > best_val:
                    best_val = val
                    best_move = m
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
        else:
            best_val = math.inf
            for m in moves:
                val, _ = _ab(s.make_move(m), depth + 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_move = m
                beta = min(beta, val)
                if alpha >= beta:
                    break

        remaining = max_depth - depth
        if best_val <= alpha_orig:
            flag = TranspositionTable.UPPER
        elif best_val >= beta:
            flag = TranspositionTable.LOWER
        else:
            flag = TranspositionTable.EXACT
        tt.store(key, remaining, best_val, flag, best_move)

        return best_val, best_move

    return _ab(state, 0, alpha, beta)


# ============================================================
# Proof Number Search
# ============================================================

class PNSNode:
    """Node for Proof Number Search."""

    def __init__(self, state, node_type="OR", parent=None, move=None):
        self.state = state
        self.type = node_type  # OR (max) or AND (min)
        self.parent = parent
        self.move = move
        self.children = None  # None = unexpanded
        self.proof = 1     # proof number
        self.disproof = 1  # disproof number
        self.value = None  # True (proved), False (disproved), None (unknown)

    def is_expanded(self):
        return self.children is not None


def proof_number_search(state, player=None, max_nodes=100000):
    """
    Proof Number Search -- tries to prove win/loss/draw.
    Returns (result, move) where result is True (win), False (loss), or None (unknown).
    """
    if player is None:
        player = state.get_current_player()

    root = PNSNode(state, "OR")
    _pns_evaluate(root, player)
    _pns_set_numbers(root)

    nodes_created = [1]

    while root.proof != 0 and root.disproof != 0:
        if nodes_created[0] >= max_nodes:
            break

        # Select most proving node
        mpn = _pns_select(root)
        if mpn is None:
            break

        # Expand
        _pns_expand(mpn, player, nodes_created)

        # Update ancestors
        node = mpn
        while node is not None:
            old_proof, old_disproof = node.proof, node.disproof
            _pns_set_numbers(node)
            if node.proof == old_proof and node.disproof == old_disproof:
                break
            node = node.parent

    if root.proof == 0:
        # Proved: find winning move (child with proof == 0)
        for child in (root.children or []):
            if child.proof == 0:
                return True, child.move
        return True, None
    elif root.disproof == 0:
        return False, None
    else:
        return None, None


def _pns_evaluate(node, player):
    """Evaluate a leaf node."""
    if node.state.is_terminal():
        u = node.state.get_utility(player)
        if u > 0:
            node.value = True
        elif u < 0:
            node.value = False
        else:
            node.value = False  # Draw treated as not-winning for proof search


def _pns_set_numbers(node):
    """Set proof/disproof numbers."""
    if node.value is True:
        node.proof = 0
        node.disproof = math.inf
        return
    if node.value is False:
        node.proof = math.inf
        node.disproof = 0
        return

    if node.children is None:
        # Unexpanded
        node.proof = 1
        node.disproof = 1
        return

    if not node.children:
        # No moves (should be terminal)
        node.proof = math.inf
        node.disproof = 0
        return

    if node.type == "OR":
        node.proof = min(c.proof for c in node.children)
        node.disproof = sum(c.disproof for c in node.children)
    else:
        node.proof = sum(c.proof for c in node.children)
        node.disproof = min(c.disproof for c in node.children)


def _pns_select(node):
    """Select most proving node."""
    if node.children is None:
        return node
    if not node.children:
        return None

    if node.type == "OR":
        # Select child with smallest proof number
        best = min(node.children, key=lambda c: c.proof)
    else:
        best = min(node.children, key=lambda c: c.disproof)

    if best.proof == 0 or best.disproof == 0:
        return None

    return _pns_select(best)


def _pns_expand(node, player, nodes_created):
    """Expand a leaf node."""
    child_type = "AND" if node.type == "OR" else "OR"
    node.children = []

    for move in node.state.get_moves():
        child = PNSNode(node.state.make_move(move), child_type, parent=node, move=move)
        _pns_evaluate(child, player)
        _pns_set_numbers(child)
        node.children.append(child)
        nodes_created[0] += 1


# ============================================================
# Game Solver Facade
# ============================================================

class GameSolver:
    """Unified interface for game solving algorithms."""

    def __init__(self, algorithm="alpha_beta", **kwargs):
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.stats = {}

    def solve(self, state, player=None):
        """Solve/evaluate a game position. Returns (value, best_move)."""
        if self.algorithm == "minimax":
            return minimax(state, player=player, **self.kwargs)
        elif self.algorithm == "alpha_beta":
            return alpha_beta(state, player=player, **self.kwargs)
        elif self.algorithm == "negamax":
            return negamax(state, player=player, **self.kwargs)
        elif self.algorithm == "alpha_beta_tt":
            return alpha_beta_tt(state, player=player, **self.kwargs)
        elif self.algorithm == "pvs":
            return pvs(state, player=player, **self.kwargs)
        elif self.algorithm == "mcts":
            move, root = mcts(state, player=player, **self.kwargs)
            self.stats["root"] = root
            return 0, move  # MCTS doesn't return exact value
        elif self.algorithm == "iterative_deepening":
            val, move, depth = iterative_deepening(state, player=player, **self.kwargs)
            self.stats["depth_reached"] = depth
            return val, move
        elif self.algorithm == "aspiration":
            val, move, depth = aspiration_search(state, player=player, **self.kwargs)
            self.stats["depth_reached"] = depth
            return val, move
        elif self.algorithm == "alpha_beta_ordered":
            val, move, stats = alpha_beta_ordered(state, player=player, **self.kwargs)
            self.stats.update(stats)
            return val, move
        elif self.algorithm == "pns":
            result, move = proof_number_search(state, player=player, **self.kwargs)
            self.stats["proved"] = result
            if result is True:
                return 1.0, move
            elif result is False:
                return -1.0, move
            return 0, move
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
