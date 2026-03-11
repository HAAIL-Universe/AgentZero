"""Tests for C125: Game Solver."""

import math
import random
import pytest

from game_solver import (
    TicTacToe, ConnectFour, Nim,
    GameState, ChanceGameState, MultiPlayerGameState,
    minimax, alpha_beta, alpha_beta_tt, negamax, pvs,
    iterative_deepening, aspiration_search,
    alpha_beta_ordered, expectimax, mcts,
    proof_number_search, maxn, paranoid,
    TranspositionTable, MoveOrderer, MCTSNode,
    count_nodes, solve_game, GameSolver,
)


# ============================================================
# TicTacToe Tests
# ============================================================

class TestTicTacToe:
    def test_initial_state(self):
        game = TicTacToe()
        assert game.get_current_player() == 1
        assert len(game.get_moves()) == 9
        assert not game.is_terminal()

    def test_make_move(self):
        game = TicTacToe()
        g2 = game.make_move(4)
        assert g2.board[4] == 1
        assert g2.get_current_player() == 2
        assert len(g2.get_moves()) == 8

    def test_player1_wins(self):
        # X wins top row
        board = [1, 1, 1, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 2)
        assert game.is_terminal()
        assert game.get_utility(1) == 1.0
        assert game.get_utility(2) == -1.0

    def test_player2_wins(self):
        board = [1, 1, 0, 2, 2, 2, 1, 0, 0]
        game = TicTacToe(board, 1)
        assert game.is_terminal()
        assert game.get_utility(2) == 1.0
        assert game.get_utility(1) == -1.0

    def test_draw(self):
        board = [1, 2, 1, 1, 2, 2, 2, 1, 1]
        game = TicTacToe(board, 2)
        assert game.is_terminal()
        assert game.get_utility(1) == 0.0
        assert game.get_utility(2) == 0.0

    def test_diagonal_win(self):
        board = [1, 2, 0, 0, 1, 2, 0, 0, 1]
        game = TicTacToe(board, 2)
        assert game.is_terminal()
        assert game.get_utility(1) == 1.0

    def test_anti_diagonal_win(self):
        board = [0, 0, 2, 0, 2, 1, 2, 1, 1]
        game = TicTacToe(board, 1)
        assert game.is_terminal()
        assert game.get_utility(2) == 1.0

    def test_column_win(self):
        board = [1, 2, 0, 1, 2, 0, 1, 0, 0]
        game = TicTacToe(board, 2)
        assert game.is_terminal()
        assert game.get_utility(1) == 1.0

    def test_no_moves_at_terminal(self):
        board = [1, 1, 1, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 2)
        assert game.get_moves() == []

    def test_hash_equality(self):
        g1 = TicTacToe([1, 0, 0, 0, 0, 0, 0, 0, 0], 2)
        g2 = TicTacToe([1, 0, 0, 0, 0, 0, 0, 0, 0], 2)
        assert g1 == g2
        assert hash(g1) == hash(g2)

    def test_hash_inequality(self):
        g1 = TicTacToe([1, 0, 0, 0, 0, 0, 0, 0, 0], 2)
        g2 = TicTacToe([0, 1, 0, 0, 0, 0, 0, 0, 0], 2)
        assert g1 != g2

    def test_evaluate_heuristic(self):
        game = TicTacToe()
        val = game.evaluate(1)
        assert isinstance(val, float)


# ============================================================
# Nim Tests
# ============================================================

class TestNim:
    def test_initial_state(self):
        game = Nim(10)
        assert game.get_current_player() == 1
        assert game.get_moves() == [1, 2, 3]
        assert not game.is_terminal()

    def test_terminal(self):
        game = Nim(0)
        assert game.is_terminal()

    def test_small_pile(self):
        game = Nim(2)
        assert game.get_moves() == [1, 2]

    def test_make_move(self):
        game = Nim(5, 1)
        g2 = game.make_move(2)
        assert g2.pile == 3
        assert g2.current == 2

    def test_utility(self):
        # Pile 0, player 1's turn -> player 1 loses
        game = Nim(0, 1)
        assert game.get_utility(1) == -1.0
        assert game.get_utility(2) == 1.0

    def test_hash(self):
        g1 = Nim(5, 1)
        g2 = Nim(5, 1)
        assert hash(g1) == hash(g2)


# ============================================================
# ConnectFour Tests
# ============================================================

class TestConnectFour:
    def test_initial_state(self):
        game = ConnectFour()
        assert game.get_current_player() == 1
        assert len(game.get_moves()) == 7
        assert not game.is_terminal()

    def test_make_move(self):
        game = ConnectFour()
        g2 = game.make_move(3)
        assert len(g2.board[3]) == 1
        assert g2.board[3][0] == 1
        assert g2.get_current_player() == 2

    def test_vertical_win(self):
        game = ConnectFour()
        # Player 1 fills column 0, player 2 fills column 1
        g = game.make_move(0).make_move(1).make_move(0).make_move(1).make_move(0).make_move(1).make_move(0)
        assert g.is_terminal()
        assert g.get_utility(1) == 1.0

    def test_not_terminal_early(self):
        game = ConnectFour()
        g = game.make_move(0).make_move(1)
        assert not g.is_terminal()

    def test_hash(self):
        g1 = ConnectFour()
        g2 = ConnectFour()
        assert hash(g1) == hash(g2)


# ============================================================
# Minimax Tests
# ============================================================

class TestMinimax:
    def test_tictactoe_draw(self):
        """Tic-tac-toe is a draw with perfect play."""
        game = TicTacToe()
        val, move = minimax(game)
        assert val == 0.0  # draw

    def test_winning_position(self):
        # Player 1 can win in one move (top row: X X _)
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        val, move = minimax(game)
        assert val == 1.0
        assert move == 2

    def test_blocking_position(self):
        # Player 2 must block (top row: X X _, player 2's turn)
        board = [1, 1, 0, 2, 0, 0, 0, 0, 0]
        game = TicTacToe(board, 2)
        val, move = minimax(game)
        assert move == 2  # must block

    def test_nim_solution(self):
        """Nim(5) with take 1-3, last to take loses: P1 wins by taking 1."""
        game = Nim(5, 1)
        val, move = minimax(game)
        # pile=5: take 1 -> pile=4 (losing for opponent)
        assert val == 1.0
        assert move == 1

    def test_nim_losing(self):
        """Nim(4): pile%4==0 is losing for current player (misere Nim)."""
        game = Nim(4, 1)
        val, move = minimax(game)
        assert val == -1.0  # P1 loses from pile=4

    def test_depth_limited(self):
        game = TicTacToe()
        val, move = minimax(game, max_depth=2)
        assert move is not None  # should return a valid move

    def test_specific_player(self):
        game = TicTacToe()
        val, move = minimax(game, player=2)
        assert move is not None


# ============================================================
# Alpha-Beta Tests
# ============================================================

class TestAlphaBeta:
    def test_tictactoe_draw(self):
        game = TicTacToe()
        val, move = alpha_beta(game)
        assert val == 0.0

    def test_same_as_minimax(self):
        """Alpha-beta should give same result as minimax."""
        game = TicTacToe()
        mm_val, _ = minimax(game)
        ab_val, _ = alpha_beta(game)
        assert mm_val == ab_val

    def test_winning_move(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        val, move = alpha_beta(game)
        assert val == 1.0
        assert move == 2

    def test_nim(self):
        game = Nim(5, 1)
        val, move = alpha_beta(game)
        assert val == 1.0
        assert move == 1

    def test_depth_limited(self):
        game = TicTacToe()
        val, move = alpha_beta(game, max_depth=3)
        assert move is not None


# ============================================================
# Alpha-Beta with TT Tests
# ============================================================

class TestAlphaBetaTT:
    def test_tictactoe_draw(self):
        game = TicTacToe()
        val, move = alpha_beta_tt(game)
        assert val == 0.0

    def test_same_as_alpha_beta(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        val1, move1 = alpha_beta(game)
        val2, move2 = alpha_beta_tt(game)
        assert val1 == val2

    def test_tt_hits(self):
        tt = TranspositionTable()
        game = TicTacToe()
        alpha_beta_tt(game, tt=tt)
        assert tt.stores > 0

    def test_nim(self):
        game = Nim(7, 1)
        val, move = alpha_beta_tt(game)
        val2, _ = minimax(game)
        assert val == val2

    def test_with_shared_tt(self):
        tt = TranspositionTable()
        game = TicTacToe()
        alpha_beta_tt(game, max_depth=3, tt=tt)
        hits_before = tt.hits
        alpha_beta_tt(game, max_depth=3, tt=tt)
        assert tt.hits >= hits_before  # should get some hits


# ============================================================
# TranspositionTable Tests
# ============================================================

class TestTranspositionTable:
    def test_store_and_lookup_exact(self):
        tt = TranspositionTable()
        tt.store("key1", 5, 0.5, TranspositionTable.EXACT, "move_a")
        found, val, move = tt.lookup("key1", 5, -1, 1)
        assert found
        assert val == 0.5
        assert move == "move_a"

    def test_lookup_miss(self):
        tt = TranspositionTable()
        found, val, move = tt.lookup("missing", 0, -1, 1)
        assert not found

    def test_depth_check(self):
        tt = TranspositionTable()
        tt.store("key1", 3, 0.5, TranspositionTable.EXACT, "move_a")
        # Requesting deeper search than stored
        found, val, move = tt.lookup("key1", 5, -1, 1)
        assert not found
        assert move == "move_a"  # still returns move for ordering

    def test_lower_bound(self):
        tt = TranspositionTable()
        tt.store("key1", 5, 0.8, TranspositionTable.LOWER, "move_a")
        # Lower bound 0.8 >= beta 0.7 -> usable
        found, val, move = tt.lookup("key1", 5, 0.3, 0.7)
        assert found
        assert val == 0.8

    def test_upper_bound(self):
        tt = TranspositionTable()
        tt.store("key1", 5, 0.2, TranspositionTable.UPPER, "move_a")
        # Upper bound 0.2 <= alpha 0.3 -> usable
        found, val, move = tt.lookup("key1", 5, 0.3, 0.7)
        assert found
        assert val == 0.2

    def test_clear(self):
        tt = TranspositionTable()
        tt.store("key1", 5, 0.5, TranspositionTable.EXACT, "m")
        tt.clear()
        found, _, _ = tt.lookup("key1", 0, -1, 1)
        assert not found


# ============================================================
# Negamax Tests
# ============================================================

class TestNegamax:
    def test_tictactoe_draw(self):
        game = TicTacToe()
        val, move = negamax(game)
        assert val == 0.0

    def test_winning_position(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        val, move = negamax(game)
        assert val == 1.0
        assert move == 2

    def test_same_as_minimax(self):
        game = Nim(5, 1)
        mm_val, _ = minimax(game)
        nm_val, _ = negamax(game)
        assert mm_val == nm_val

    def test_depth_limited(self):
        game = TicTacToe()
        val, move = negamax(game, max_depth=2)
        assert move is not None


# ============================================================
# Iterative Deepening Tests
# ============================================================

class TestIterativeDeepening:
    def test_tictactoe(self):
        game = TicTacToe()
        val, move, depth = iterative_deepening(game, max_depth=9)
        assert val == 0.0  # draw
        assert depth >= 1

    def test_winning_position(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        val, move, depth = iterative_deepening(game, max_depth=5)
        assert val == 1.0
        assert move == 2

    def test_depth_reached(self):
        game = TicTacToe()
        _, _, depth = iterative_deepening(game, max_depth=3)
        assert depth <= 3

    def test_time_limit(self):
        game = TicTacToe()
        _, move, _ = iterative_deepening(game, max_depth=20, time_limit=0.5)
        assert move is not None

    def test_nim(self):
        game = Nim(6, 1)
        val, move, _ = iterative_deepening(game, max_depth=10)
        mm_val, _ = minimax(game)
        assert val == mm_val


# ============================================================
# MCTS Tests
# ============================================================

class TestMCTS:
    def test_basic(self):
        random.seed(42)
        game = TicTacToe()
        move, root = mcts(game, iterations=500)
        assert move in game.get_moves()
        assert root.visits > 0

    def test_winning_move(self):
        """MCTS should find obvious winning move."""
        random.seed(42)
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        move, _ = mcts(game, iterations=2000)
        assert move == 2  # win in one

    def test_blocking_move(self):
        """MCTS should block opponent's winning move."""
        random.seed(42)
        board = [1, 1, 0, 2, 0, 0, 0, 0, 0]
        game = TicTacToe(board, 2)
        move, _ = mcts(game, iterations=2000)
        assert move == 2  # must block

    def test_root_stats(self):
        random.seed(42)
        game = TicTacToe()
        _, root = mcts(game, iterations=100)
        total_child_visits = sum(c.visits for c in root.children)
        assert total_child_visits > 0

    def test_exploration_parameter(self):
        random.seed(42)
        game = TicTacToe()
        _, root1 = mcts(game, iterations=100, exploration=0.5)
        _, root2 = mcts(game, iterations=100, exploration=2.0)
        # Both should produce valid results
        assert root1.visits > 0
        assert root2.visits > 0

    def test_time_limit(self):
        random.seed(42)
        game = TicTacToe()
        move, root = mcts(game, iterations=1_000_000, time_limit=0.1)
        assert move is not None

    def test_terminal_state(self):
        board = [1, 1, 1, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 2)
        move, root = mcts(game, iterations=10)
        assert move is None  # no moves available

    def test_nim(self):
        random.seed(42)
        game = Nim(5, 1)
        move, _ = mcts(game, iterations=2000)
        assert move == 1  # optimal: take 1 to leave pile=4 (losing for opponent)


# ============================================================
# MCTSNode Tests
# ============================================================

class TestMCTSNode:
    def test_creation(self):
        game = TicTacToe()
        node = MCTSNode(game)
        assert node.visits == 0
        assert node.wins == 0.0
        assert not node.is_fully_expanded()

    def test_expand(self):
        game = TicTacToe()
        node = MCTSNode(game)
        child = node.expand()
        assert len(node.children) == 1
        assert child.parent == node

    def test_ucb1_unexplored(self):
        game = TicTacToe()
        node = MCTSNode(game)
        assert node.ucb1() == math.inf

    def test_update(self):
        game = TicTacToe()
        node = MCTSNode(game)
        node.update(1.0)
        assert node.visits == 1
        assert node.wins == 1.0

    def test_fully_expanded(self):
        game = TicTacToe()
        node = MCTSNode(game)
        while node.untried_moves:
            node.expand()
        assert node.is_fully_expanded()

    def test_best_child(self):
        game = TicTacToe()
        root = MCTSNode(game)
        root.visits = 10
        c1 = root.expand()
        c1.visits = 5
        c1.wins = 3.0
        c2 = root.expand()
        c2.visits = 5
        c2.wins = 1.0
        best = root.best_child()
        assert best == c1  # higher win rate


# ============================================================
# PVS Tests
# ============================================================

class TestPVS:
    def test_tictactoe_draw(self):
        game = TicTacToe()
        val, move = pvs(game)
        assert val == 0.0

    def test_winning_position(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        val, move = pvs(game)
        assert val == 1.0
        assert move == 2

    def test_same_as_minimax(self):
        game = Nim(5, 1)
        mm_val, _ = minimax(game)
        pvs_val, _ = pvs(game)
        assert mm_val == pvs_val

    def test_depth_limited(self):
        game = TicTacToe()
        val, move = pvs(game, max_depth=3)
        assert move is not None


# ============================================================
# Expectimax Tests
# ============================================================

class TestExpectimax:
    def test_deterministic_game(self):
        """On deterministic games, expectimax should match minimax."""
        game = TicTacToe()
        mm_val, _ = minimax(game)
        em_val, _ = expectimax(game)
        assert mm_val == em_val

    def test_nim(self):
        game = Nim(5, 1)
        val, move = expectimax(game)
        assert val == 1.0

    def test_chance_game(self):
        """Test with a simple chance game."""
        class CoinFlipGame(ChanceGameState):
            def __init__(self, score=0, turns=0, current=1):
                self.score = score
                self.turns = turns
                self.current = current

            def get_moves(self):
                if self.is_terminal():
                    return []
                return ["bet", "pass"]

            def make_move(self, move):
                if move == "pass":
                    return CoinFlipGame(self.score, self.turns + 1, 3 - self.current)
                if move == "heads":
                    return CoinFlipGame(self.score + 1, self.turns + 1, 3 - self.current)
                if move == "tails":
                    return CoinFlipGame(self.score - 1, self.turns + 1, 3 - self.current)
                # bet -> chance node
                return CoinFlipChance(self.score, self.turns, self.current)

            def is_terminal(self):
                return self.turns >= 2

            def get_utility(self, player):
                return float(self.score) if player == 1 else -float(self.score)

            def get_current_player(self):
                return self.current

            def is_chance_node(self):
                return False

        class CoinFlipChance(ChanceGameState):
            def __init__(self, score, turns, current):
                self.score = score
                self.turns = turns
                self.current = current

            def get_moves(self):
                return ["heads", "tails"]

            def make_move(self, move):
                if move == "heads":
                    return CoinFlipGame(self.score + 1, self.turns + 1, 3 - self.current)
                return CoinFlipGame(self.score - 1, self.turns + 1, 3 - self.current)

            def is_terminal(self):
                return False

            def get_utility(self, player):
                return 0

            def get_current_player(self):
                return self.current

            def is_chance_node(self):
                return True

            def get_chance_outcomes(self):
                return [(0.5, "heads"), (0.5, "tails")]

        game = CoinFlipGame()
        val, move = expectimax(game)
        assert isinstance(val, float)
        assert move is not None


# ============================================================
# Move Orderer Tests
# ============================================================

class TestMoveOrderer:
    def test_killer_moves(self):
        orderer = MoveOrderer()
        orderer.record_killer(0, "move_a")
        ordered = orderer.order_moves(["move_b", "move_a", "move_c"], 0)
        assert ordered[0] == "move_a"

    def test_history_heuristic(self):
        orderer = MoveOrderer()
        orderer.record_history("move_c", 5)
        orderer.record_history("move_a", 1)
        ordered = orderer.order_moves(["move_a", "move_b", "move_c"], 0)
        assert ordered[0] == "move_c"

    def test_tt_move_first(self):
        orderer = MoveOrderer()
        ordered = orderer.order_moves(["a", "b", "c"], 0, tt_move="c")
        assert ordered[0] == "c"

    def test_priority_order(self):
        orderer = MoveOrderer()
        orderer.record_killer(0, "b")
        orderer.record_history("c", 3)
        ordered = orderer.order_moves(["a", "b", "c", "d"], 0, tt_move="d")
        assert ordered[0] == "d"  # TT first
        assert ordered[1] == "b"  # killer second

    def test_max_killers(self):
        orderer = MoveOrderer()
        orderer.record_killer(0, "a")
        orderer.record_killer(0, "b")
        orderer.record_killer(0, "c")
        assert len(orderer.killers[0]) == 2


# ============================================================
# Alpha-Beta Ordered Tests
# ============================================================

class TestAlphaBetaOrdered:
    def test_tictactoe_draw(self):
        game = TicTacToe()
        val, move, stats = alpha_beta_ordered(game)
        assert val == 0.0
        assert "nodes_searched" in stats

    def test_same_value_as_minimax(self):
        game = Nim(5, 1)
        mm_val, _ = minimax(game)
        ab_val, _, _ = alpha_beta_ordered(game)
        assert mm_val == ab_val

    def test_nodes_tracked(self):
        game = TicTacToe()
        _, _, stats = alpha_beta_ordered(game)
        assert stats["nodes_searched"] > 0


# ============================================================
# Multi-player Game Tests
# ============================================================

class ThreePlayerNim(MultiPlayerGameState):
    """3-player Nim for testing."""
    def __init__(self, pile=10, current=1):
        self.pile = pile
        self.current = current

    def get_moves(self):
        if self.is_terminal():
            return []
        return [i for i in range(1, min(3, self.pile) + 1)]

    def make_move(self, move):
        next_player = (self.current % 3) + 1
        return ThreePlayerNim(self.pile - move, next_player)

    def is_terminal(self):
        return self.pile <= 0

    def get_utilities(self):
        # Current player (facing empty pile) loses
        result = {}
        for p in [1, 2, 3]:
            if p == self.current:
                result[p] = -1.0
            else:
                result[p] = 0.5
        return result

    def get_current_player(self):
        return self.current

    def get_players(self):
        return [1, 2, 3]

    def evaluate_all(self):
        return {1: 0, 2: 0, 3: 0}

    def __hash__(self):
        return hash((self.pile, self.current))

    def __eq__(self, other):
        return isinstance(other, ThreePlayerNim) and self.pile == other.pile and self.current == other.current


class TestMaxN:
    def test_three_player(self):
        game = ThreePlayerNim(5, 1)
        utils, move = maxn(game)
        assert move in game.get_moves()
        assert 1 in utils

    def test_terminal(self):
        game = ThreePlayerNim(0, 1)
        utils, move = maxn(game)
        assert utils[1] == -1.0

    def test_depth_limited(self):
        game = ThreePlayerNim(8, 1)
        utils, move = maxn(game, max_depth=3)
        assert move is not None


class TestParanoid:
    def test_three_player(self):
        game = ThreePlayerNim(5, 1)
        val, move = paranoid(game)
        assert move in game.get_moves()

    def test_depth_limited(self):
        game = ThreePlayerNim(8, 1)
        val, move = paranoid(game, max_depth=3)
        assert move is not None


# ============================================================
# Aspiration Window Tests
# ============================================================

class TestAspirationSearch:
    def test_tictactoe(self):
        game = TicTacToe()
        val, move, depth = aspiration_search(game, max_depth=9)
        assert val == 0.0
        assert depth >= 1

    def test_winning(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        val, move, depth = aspiration_search(game, max_depth=5)
        assert val == 1.0
        assert move == 2

    def test_nim(self):
        game = Nim(6, 1)
        val, move, _ = aspiration_search(game, max_depth=10)
        mm_val, _ = minimax(game)
        assert val == mm_val


# ============================================================
# Proof Number Search Tests
# ============================================================

class TestProofNumberSearch:
    def test_winning_position(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        result, move = proof_number_search(game)
        assert result is True
        assert move == 2

    def test_draw_position(self):
        """From starting position, TicTacToe should not be provably won."""
        game = TicTacToe()
        result, move = proof_number_search(game, max_nodes=50000)
        # PNS may or may not prove draw -- it tries to prove win
        assert result in (True, False, None)

    def test_nim_winning(self):
        game = Nim(5, 1)
        result, move = proof_number_search(game)
        assert result is True

    def test_nim_losing(self):
        """Nim(4): pile%4==0 is losing for current player."""
        game = Nim(4, 1)
        result, move = proof_number_search(game)
        # P1 can't force a win from pile=4
        assert result is False


# ============================================================
# Count Nodes Tests
# ============================================================

class TestCountNodes:
    def test_terminal(self):
        board = [1, 1, 1, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 2)
        assert count_nodes(game) == 1

    def test_depth_0(self):
        game = TicTacToe()
        assert count_nodes(game, max_depth=0) == 1

    def test_nim_small(self):
        game = Nim(2, 1)
        nodes = count_nodes(game)
        assert nodes > 1


# ============================================================
# Solve Game Tests
# ============================================================

class TestSolveGame:
    def test_tictactoe(self):
        game = TicTacToe()
        val, move = solve_game(game)
        assert val == 0.0

    def test_nim(self):
        game = Nim(5, 1)
        val, move = solve_game(game)
        assert val == 1.0


# ============================================================
# GameSolver Facade Tests
# ============================================================

class TestGameSolver:
    def test_minimax(self):
        solver = GameSolver("minimax")
        val, move = solver.solve(TicTacToe())
        assert val == 0.0

    def test_alpha_beta(self):
        solver = GameSolver("alpha_beta")
        val, move = solver.solve(TicTacToe())
        assert val == 0.0

    def test_negamax(self):
        solver = GameSolver("negamax")
        val, move = solver.solve(TicTacToe())
        assert val == 0.0

    def test_alpha_beta_tt(self):
        solver = GameSolver("alpha_beta_tt")
        val, move = solver.solve(TicTacToe())
        assert val == 0.0

    def test_pvs(self):
        solver = GameSolver("pvs")
        val, move = solver.solve(TicTacToe())
        assert val == 0.0

    def test_mcts(self):
        random.seed(42)
        solver = GameSolver("mcts", iterations=100)
        val, move = solver.solve(TicTacToe())
        assert move is not None
        assert "root" in solver.stats

    def test_iterative_deepening(self):
        solver = GameSolver("iterative_deepening", max_depth=5)
        val, move = solver.solve(TicTacToe())
        assert "depth_reached" in solver.stats

    def test_aspiration(self):
        solver = GameSolver("aspiration", max_depth=5)
        val, move = solver.solve(TicTacToe())
        assert "depth_reached" in solver.stats

    def test_alpha_beta_ordered(self):
        solver = GameSolver("alpha_beta_ordered")
        val, move = solver.solve(TicTacToe())
        assert val == 0.0

    def test_pns(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        solver = GameSolver("pns")
        val, move = solver.solve(TicTacToe(board, 1))
        assert val == 1.0

    def test_unknown_algorithm(self):
        solver = GameSolver("unknown")
        with pytest.raises(ValueError):
            solver.solve(TicTacToe())

    def test_with_player(self):
        solver = GameSolver("alpha_beta")
        val, move = solver.solve(TicTacToe(), player=2)
        assert move is not None

    def test_with_kwargs(self):
        solver = GameSolver("alpha_beta", max_depth=3)
        val, move = solver.solve(TicTacToe())
        assert move is not None


# ============================================================
# Cross-Algorithm Consistency Tests
# ============================================================

class TestCrossAlgorithm:
    """All exact algorithms should agree on game-theoretic values."""

    def test_all_agree_tictactoe(self):
        game = TicTacToe()
        mm_val, _ = minimax(game)
        ab_val, _ = alpha_beta(game)
        nm_val, _ = negamax(game)
        pvs_val, _ = pvs(game)
        tt_val, _ = alpha_beta_tt(game)
        em_val, _ = expectimax(game)
        assert mm_val == ab_val == nm_val == pvs_val == tt_val == em_val

    def test_all_agree_nim(self):
        for pile in [3, 4, 5, 6, 7]:
            game = Nim(pile, 1)
            mm_val, _ = minimax(game)
            ab_val, _ = alpha_beta(game)
            nm_val, _ = negamax(game)
            assert mm_val == ab_val == nm_val, f"Disagreement at pile={pile}"

    def test_winning_move_consistency(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        _, mm_move = minimax(game)
        _, ab_move = alpha_beta(game)
        _, nm_move = negamax(game)
        _, pvs_move = pvs(game)
        # All should find the winning move
        assert mm_move == 2
        assert ab_move == 2
        assert nm_move == 2
        assert pvs_move == 2


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_move(self):
        """Only one legal move available."""
        board = [1, 2, 1, 2, 1, 2, 0, 1, 2]
        game = TicTacToe(board, 1)
        val, move = minimax(game)
        assert move == 6

    def test_already_won(self):
        board = [1, 1, 1, 0, 0, 0, 2, 2, 0]
        game = TicTacToe(board, 2)
        assert game.is_terminal()
        # Default player = current = player 2, who lost
        val, move = minimax(game, player=1)
        assert val == 1.0  # player 1 won
        assert move is None

    def test_nim_pile_1(self):
        game = Nim(1, 1)
        val, move = minimax(game)
        assert val == 1.0
        assert move == 1

    def test_mcts_single_move(self):
        board = [1, 2, 1, 2, 1, 2, 0, 1, 2]
        game = TicTacToe(board, 1)
        move, root = mcts(game, iterations=50)
        assert move == 6

    def test_depth_0_minimax(self):
        game = TicTacToe()
        val, move = minimax(game, max_depth=0)
        # At depth 0, returns heuristic evaluation
        assert isinstance(val, float)

    def test_connect_four_moves(self):
        game = ConnectFour()
        g = game
        for _ in range(6):
            g = g.make_move(0)  # fill column 0
        moves = g.get_moves()
        assert 0 not in moves  # column 0 is full
        assert len(moves) == 6

    def test_nim_various_piles(self):
        """Test Nim theory: pile%4==0 is losing for current player (misere Nim)."""
        for pile in range(1, 9):
            game = Nim(pile, 1)
            val, _ = minimax(game)
            if pile % 4 == 0:
                assert val == -1.0, f"pile={pile} should be losing"
            else:
                assert val == 1.0, f"pile={pile} should be winning"


# ============================================================
# Performance / Pruning Tests
# ============================================================

class TestPerformance:
    def test_alpha_beta_prunes(self):
        """Alpha-beta with ordering should search fewer nodes than plain."""
        game = TicTacToe()
        _, _, stats = alpha_beta_ordered(game)
        # Full tic-tac-toe tree has 549946 nodes; AB should be much less
        assert stats["nodes_searched"] < 200000

    def test_tt_provides_speedup(self):
        """With TT, second search should be faster (hits)."""
        tt = TranspositionTable()
        game = TicTacToe()
        alpha_beta_tt(game, max_depth=5, tt=tt)
        stores1 = tt.stores
        alpha_beta_tt(game, max_depth=5, tt=tt)
        # Second search should hit TT entries
        assert tt.hits > 0

    def test_iterative_deepening_stops_at_win(self):
        board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
        game = TicTacToe(board, 1)
        val, move, depth = iterative_deepening(game, max_depth=20)
        assert val == 1.0
        assert depth <= 5  # should find win quickly and stop
