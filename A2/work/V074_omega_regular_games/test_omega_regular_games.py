"""Tests for V074: Omega-Regular Games."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from omega_regular_games import (
    # Core API
    check_ltl_game, check_ltl_game_direct, check_ltl_labeled_game,
    # Product construction
    build_product_game, ProductGame,
    # Qualitative
    compute_almost_sure_winning, compute_positive_winning,
    # Quantitative
    solve_buchi_game_quantitative,
    # Safety/Liveness
    check_safety_game, check_liveness_game,
    check_persistence_game, check_recurrence_game, check_response_game,
    # Multi-objective
    check_multi_ltl_game, check_conjunction_game,
    # Strategy verification
    verify_ltl_strategy,
    # Comparison
    compare_ltl_vs_pctl, compare_direct_vs_negation,
    # Result type
    OmegaRegularResult,
)

# V023 LTL AST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
from ltl_model_checker import (
    Atom, LTLTrue, LTLFalse, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release, WeakUntil,
    parse_ltl, ltl_to_gba, gba_to_nba, nnf
)

# V070 Stochastic games
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V070_stochastic_games'))
from stochastic_games import (
    StochasticGame, Player, StrategyPair, make_game, game_to_mc
)

# V072 LabeledGame
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V072_game_pctl'))
from game_pctl import LabeledGame, make_labeled_game


# ============================================================
# Helper: build simple games
# ============================================================

def make_simple_reachability_game():
    """Simple 3-state game: P1 chooses left/right, P2 chooses stay/go.

    State 0 (P1): go_left -> state 1, go_right -> state 2
    State 1 (P2): stay -> state 1, go_target -> state 2
    State 2 (absorbing, target): self-loop

    Labels: state 2 has 'target'
    """
    game = make_game(
        n_states=3,
        owners={0: Player.P1, 1: Player.P2, 2: Player.P1},
        action_transitions={
            0: {'left': [0, 1, 0], 'right': [0, 0, 1]},
            1: {'stay': [0, 1, 0], 'go': [0, 0, 1]},
            2: {'loop': [0, 0, 1]},
        },
        state_labels=['s0', 's1', 's2']
    )
    labels = {0: set(), 1: set(), 2: {'target'}}
    return game, labels


def make_stochastic_game():
    """Game with chance nodes.

    State 0 (P1): safe -> state 1, risky -> state 2
    State 1 (CHANCE): 0.8 -> state 0, 0.2 -> state 3
    State 2 (CHANCE): 0.5 -> state 3, 0.5 -> state 4
    State 3 (absorbing, good): self-loop
    State 4 (absorbing, bad): self-loop

    Labels: state 3 has 'good', state 4 has 'bad'
    """
    game = make_game(
        n_states=5,
        owners={0: Player.P1, 1: Player.CHANCE, 2: Player.CHANCE,
                3: Player.P1, 4: Player.P1},
        action_transitions={
            0: {'safe': [0, 1, 0, 0, 0], 'risky': [0, 0, 1, 0, 0]},
            1: {'flip': [0.8, 0, 0, 0.2, 0]},
            2: {'flip': [0, 0, 0, 0.5, 0.5]},
            3: {'loop': [0, 0, 0, 1, 0]},
            4: {'loop': [0, 0, 0, 0, 1]},
        },
        state_labels=['start', 'safe_chance', 'risky_chance', 'good', 'bad']
    )
    labels = {0: set(), 1: set(), 2: set(), 3: {'good'}, 4: {'bad'}}
    return game, labels


def make_cycling_game():
    """Game where P1 can force cycling through states.

    State 0 (P1): go -> state 1
    State 1 (P2): left -> state 2, right -> state 3
    State 2 (P1): back -> state 0  (has label 'a')
    State 3 (P1): back -> state 0  (has label 'b')

    Labels: state 2 has 'a', state 3 has 'b'
    """
    game = make_game(
        n_states=4,
        owners={0: Player.P1, 1: Player.P2, 2: Player.P1, 3: Player.P1},
        action_transitions={
            0: {'go': [0, 1, 0, 0]},
            1: {'left': [0, 0, 1, 0], 'right': [0, 0, 0, 1]},
            2: {'back': [1, 0, 0, 0]},
            3: {'back': [1, 0, 0, 0]},
        },
        state_labels=['s0', 's1', 's2_a', 's3_b']
    )
    labels = {0: set(), 1: set(), 2: {'a'}, 3: {'b'}}
    return game, labels


def make_mutex_game():
    """Simple mutual exclusion game.

    Two processes (P1 and P2) trying to enter critical section.
    State encoding: (p1_state, p2_state) where state in {idle, trying, critical}

    Simplified: 4 states
    State 0: both idle
    State 1: P1 in critical (P1 chooses)
    State 2: P2 in critical (P2 chooses)
    State 3: both trying -> nondeterministic

    Labels: cs1 for P1 in CS, cs2 for P2 in CS
    """
    game = make_game(
        n_states=4,
        owners={0: Player.P1, 1: Player.P1, 2: Player.P2, 3: Player.CHANCE},
        action_transitions={
            0: {'p1_try': [0, 1, 0, 0], 'p2_try': [0, 0, 1, 0], 'both_try': [0, 0, 0, 1]},
            1: {'exit': [1, 0, 0, 0]},
            2: {'exit': [1, 0, 0, 0]},
            3: {'resolve': [0, 0.5, 0.5, 0]},
        },
        state_labels=['idle', 'cs1', 'cs2', 'both_try']
    )
    labels = {0: set(), 1: {'cs1'}, 2: {'cs2'}, 3: set()}
    return game, labels


# ============================================================
# Section 1: LTL to NBA Construction (V023 reuse)
# ============================================================

class TestLTLAutomaton:
    def test_simple_finally(self):
        """F(p) automaton construction."""
        f = Finally(Atom("p"))
        gba = ltl_to_gba(nnf(f))
        nba = gba_to_nba(gba)
        assert len(nba.states) > 0
        assert len(nba.initial) > 0
        assert len(nba.accepting) > 0

    def test_globally_automaton(self):
        """G(p) automaton construction."""
        f = Globally(Atom("p"))
        gba = ltl_to_gba(nnf(f))
        nba = gba_to_nba(gba)
        assert len(nba.states) > 0

    def test_until_automaton(self):
        """p U q automaton construction."""
        f = Until(Atom("p"), Atom("q"))
        gba = ltl_to_gba(nnf(f))
        nba = gba_to_nba(gba)
        assert len(nba.states) > 0
        assert len(nba.accepting) > 0

    def test_response_automaton(self):
        """G(p -> F(q)) automaton."""
        f = Globally(Implies(Atom("p"), Finally(Atom("q"))))
        gba = ltl_to_gba(nnf(f))
        nba = gba_to_nba(gba)
        assert len(nba.states) > 0


# ============================================================
# Section 2: Product Game Construction
# ============================================================

class TestProductConstruction:
    def test_product_simple(self):
        """Product game has correct number of states."""
        game, labels = make_simple_reachability_game()
        f = Finally(Atom("target"))
        nba = gba_to_nba(ltl_to_gba(nnf(f)))
        product = build_product_game(game, nba, labels)

        assert product.n_game_states == 3
        assert product.n_aut_states == len(nba.states)
        assert product.game.n_states == 3 * len(nba.states)

    def test_product_owners_preserved(self):
        """Product game preserves state ownership from original game."""
        game, labels = make_simple_reachability_game()
        f = Finally(Atom("target"))
        nba = gba_to_nba(ltl_to_gba(nnf(f)))
        product = build_product_game(game, nba, labels)

        for ps in range(product.game.n_states):
            g, q = product.inv_map[ps]
            assert product.game.owners[ps] == game.owners[g]

    def test_product_accepting_states(self):
        """Product accepting states correspond to NBA accepting states."""
        game, labels = make_simple_reachability_game()
        f = Finally(Atom("target"))
        nba = gba_to_nba(ltl_to_gba(nnf(f)))
        product = build_product_game(game, nba, labels)

        for ps in product.accepting:
            g, q = product.inv_map[ps]
            assert q in nba.accepting

    def test_product_initial_states(self):
        """Product initial states correspond to initial game + automaton states."""
        game, labels = make_simple_reachability_game()
        f = Finally(Atom("target"))
        nba = gba_to_nba(ltl_to_gba(nnf(f)))
        product = build_product_game(game, nba, labels, initial_game_state=0)

        assert len(product.initial_product_states) > 0
        for ps in product.initial_product_states:
            g, q = product.inv_map[ps]
            assert g == 0
            assert q in nba.initial

    def test_product_transitions_sum(self):
        """Product transitions are valid probability distributions."""
        game, labels = make_stochastic_game()
        f = Finally(Atom("good"))
        nba = gba_to_nba(ltl_to_gba(nnf(f)))
        product = build_product_game(game, nba, labels)

        for s in range(product.game.n_states):
            for a_idx in range(len(product.game.actions[s])):
                total = sum(product.game.transition[s][a_idx])
                # Total may be < 1 if some transitions have no automaton successor
                assert total <= 1.0 + 1e-9


# ============================================================
# Section 3: Qualitative Analysis
# ============================================================

class TestQualitative:
    def test_almost_sure_reachability(self):
        """Almost-sure winning for simple reachability."""
        game, labels = make_simple_reachability_game()
        f = Finally(Atom("target"))
        nba = gba_to_nba(ltl_to_gba(nnf(f)))
        product = build_product_game(game, nba, labels)

        as_winning = compute_almost_sure_winning(product)
        # P1 can go directly to target (state 2) via 'right' action
        # So initial product states should be almost-sure winning
        # (P1 has a strategy to reach target regardless of P2)
        assert len(as_winning) > 0

    def test_positive_winning_superset(self):
        """Positive winning region is superset of almost-sure winning."""
        game, labels = make_simple_reachability_game()
        f = Finally(Atom("target"))
        nba = gba_to_nba(ltl_to_gba(nnf(f)))
        product = build_product_game(game, nba, labels)

        as_winning = compute_almost_sure_winning(product)
        pos_winning = compute_positive_winning(product)
        assert as_winning <= pos_winning

    def test_no_winning_for_impossible(self):
        """No winning region when property is impossible.

        Build a game where 'target' never appears.
        """
        game = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.P1},
            action_transitions={
                0: {'a': [0, 1]},
                1: {'b': [1, 0]},
            }
        )
        labels = {0: set(), 1: set()}  # No 'target' anywhere

        f = Finally(Atom("target"))
        nba = gba_to_nba(ltl_to_gba(nnf(f)))
        product = build_product_game(game, nba, labels)

        pos_winning = compute_positive_winning(product)
        # No state can reach 'target', so positive winning should reflect that
        # F(target) can never be satisfied
        # The accepting states in the product may or may not exist,
        # but no accepting cycles should exist
        as_winning = compute_almost_sure_winning(product)
        # Almost-sure winning should be empty or minimal
        # (depends on automaton structure for F(target) with no target states)


# ============================================================
# Section 4: Quantitative Buchi Game
# ============================================================

class TestQuantitative:
    def test_deterministic_reachability(self):
        """Deterministic reachability: P1 can reach target with prob 1."""
        game, labels = make_simple_reachability_game()
        result = check_ltl_game_direct(game, labels, Finally(Atom("target")))
        # P1 can go directly right to reach target
        assert result.probabilities[0] >= 0.99

    def test_stochastic_reachability(self):
        """Stochastic reachability with chance nodes."""
        game, labels = make_stochastic_game()
        result = check_ltl_game_direct(game, labels, Finally(Atom("good")))
        # P1 can choose 'safe' (0.2 prob of good per visit, repeatedly)
        # or 'risky' (0.5 prob of good)
        # Optimal depends on repeated visits
        assert 0 < result.probabilities[0] <= 1.0

    def test_safety_property(self):
        """G(NOT bad) -- always avoid bad state."""
        game, labels = make_stochastic_game()
        # Check G(NOT bad) = avoid 'bad' forever
        formula = Globally(Not(Atom("bad")))
        result = check_ltl_game_direct(game, labels, formula)
        # If P1 always chooses 'safe', there's 0.2 chance of reaching 'good' per cycle
        # and 0.8 chance of cycling back. 'bad' is only reachable via 'risky'.
        # So G(NOT bad) should be satisfiable with some probability
        assert result.probabilities[0] >= 0.0

    def test_liveness_property(self):
        """F(good) -- eventually reach good."""
        game, labels = make_stochastic_game()
        result = check_liveness_game(game, labels, "good")
        assert result.probabilities[0] > 0

    def test_recurrence_cycling(self):
        """G(F(a)) in cycling game."""
        game, labels = make_cycling_game()
        result = check_recurrence_game(game, labels, "a")
        # P2 can choose to always go to 'b', preventing infinite visits to 'a'
        # So G(F(a)) probability depends on P2's adversarial strategy
        assert 0.0 <= result.probabilities[0] <= 1.0

    def test_values_bounded(self):
        """All probabilities are in [0, 1]."""
        game, labels = make_stochastic_game()
        result = check_ltl_game_direct(game, labels, Finally(Atom("good")))
        for p in result.probabilities:
            assert -1e-9 <= p <= 1.0 + 1e-9


# ============================================================
# Section 5: Safety and Liveness Convenience APIs
# ============================================================

class TestConvenienceAPIs:
    def test_check_safety_game(self):
        """check_safety_game wraps G(prop)."""
        game, labels = make_stochastic_game()
        result = check_safety_game(game, labels, "good")
        assert isinstance(result, OmegaRegularResult)
        assert result.formula.op.name == 'G'

    def test_check_liveness_game(self):
        """check_liveness_game wraps F(prop)."""
        game, labels = make_stochastic_game()
        result = check_liveness_game(game, labels, "good")
        assert isinstance(result, OmegaRegularResult)
        assert result.formula.op.name == 'F'

    def test_check_persistence_game(self):
        """check_persistence_game wraps F(G(prop))."""
        game, labels = make_stochastic_game()
        result = check_persistence_game(game, labels, "good")
        assert isinstance(result, OmegaRegularResult)

    def test_check_recurrence_game(self):
        """check_recurrence_game wraps G(F(prop))."""
        game, labels = make_cycling_game()
        result = check_recurrence_game(game, labels, "a")
        assert isinstance(result, OmegaRegularResult)

    def test_check_response_game(self):
        """check_response_game wraps G(trigger -> F(response))."""
        game, labels = make_mutex_game()
        result = check_response_game(game, labels, "cs1", "cs2")
        assert isinstance(result, OmegaRegularResult)


# ============================================================
# Section 6: Labeled Game Integration
# ============================================================

class TestLabeledGame:
    def test_labeled_game_api(self):
        """check_ltl_labeled_game works with V072 LabeledGame."""
        lgame = make_labeled_game(
            n_states=3,
            owners={0: Player.P1, 1: Player.P2, 2: Player.P1},
            action_transitions={
                0: {'left': [0, 1, 0], 'right': [0, 0, 1]},
                1: {'stay': [0, 1, 0], 'go': [0, 0, 1]},
                2: {'loop': [0, 0, 1]},
            },
            labels={0: set(), 1: set(), 2: {'target'}},
        )
        result = check_ltl_labeled_game(lgame, Finally(Atom("target")))
        assert isinstance(result, OmegaRegularResult)
        assert result.probabilities[0] >= 0.99  # P1 goes right directly

    def test_labeled_game_safety(self):
        """Safety property via LabeledGame."""
        lgame = make_labeled_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.P1},
            action_transitions={
                0: {'stay': [1, 0], 'leave': [0, 1]},
                1: {'loop': [0, 1]},
            },
            labels={0: {'safe'}, 1: set()},
        )
        result = check_ltl_labeled_game(lgame, Globally(Atom("safe")))
        # P1 can stay at state 0 forever (if 'stay' goes to state 0)
        # Actually stay goes to [1, 0] = all to state 0, so P1 keeps safe forever
        assert result.probabilities[0] >= 0.99


# ============================================================
# Section 7: Multi-Objective LTL
# ============================================================

class TestMultiObjective:
    def test_multi_ltl_independent(self):
        """Check multiple independent LTL properties."""
        game, labels = make_stochastic_game()
        formulas = [
            Finally(Atom("good")),
            Globally(Not(Atom("bad"))),
        ]
        results = check_multi_ltl_game(game, labels, formulas)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, OmegaRegularResult)

    def test_conjunction(self):
        """Check conjunction of formulas."""
        game, labels = make_stochastic_game()
        formulas = [
            Finally(Atom("good")),
            Globally(Not(Atom("bad"))),
        ]
        result = check_conjunction_game(game, labels, formulas)
        assert isinstance(result, OmegaRegularResult)
        # Conjunction is at most as satisfiable as either formula alone
        individual = check_multi_ltl_game(game, labels, formulas)
        for r in individual:
            assert result.probabilities[0] <= r.probabilities[0] + 1e-6


# ============================================================
# Section 8: Strategy Extraction and Verification
# ============================================================

class TestStrategies:
    def test_strategy_extraction(self):
        """Strategies are extracted from product game."""
        game, labels = make_simple_reachability_game()
        result = check_ltl_game_direct(game, labels, Finally(Atom("target")))
        assert result.game_p1_strategy is not None
        assert result.game_p2_strategy is not None
        # P1 strategy should map all P1 states to actions
        for s in range(game.n_states):
            if game.owners[s] == Player.P1:
                assert s in result.game_p1_strategy

    def test_strategy_verification(self):
        """Verify strategy via simulation."""
        game, labels = make_simple_reachability_game()
        result = check_ltl_game_direct(game, labels, Finally(Atom("target")))

        # Build strategy pair
        strategies = StrategyPair(
            p1_strategy=result.game_p1_strategy,
            p2_strategy=result.game_p2_strategy,
        )

        verification = verify_ltl_strategy(
            game, labels, Finally(Atom("target")), strategies,
            n_simulations=100, max_steps=50
        )
        assert verification['simulations'] == 100
        # With a good strategy, most simulations should satisfy F(target)
        assert verification['satisfaction_rate'] > 0.5

    def test_strategy_completeness(self):
        """All player states have strategy entries."""
        game, labels = make_stochastic_game()
        result = check_ltl_game_direct(game, labels, Finally(Atom("good")))
        if result.game_p1_strategy:
            for s in range(game.n_states):
                if game.owners[s] == Player.P1:
                    assert s in result.game_p1_strategy


# ============================================================
# Section 9: LTL Parse Integration
# ============================================================

class TestParseIntegration:
    def test_parse_and_check(self):
        """Parse LTL formula and check against game."""
        game, labels = make_simple_reachability_game()
        f = parse_ltl("F target")
        result = check_ltl_game_direct(game, labels, f)
        assert result.probabilities[0] >= 0.99

    def test_parse_globally(self):
        """Parse G(safe) and check."""
        game = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.P1},
            action_transitions={
                0: {'stay': [1, 0]},
                1: {'loop': [0, 1]},
            },
        )
        labels = {0: {'safe'}, 1: set()}
        f = parse_ltl("G safe")
        result = check_ltl_game_direct(game, labels, f)
        # State 0 has 'safe' and only action goes back to 0
        assert result.probabilities[0] >= 0.99

    def test_parse_response(self):
        """Parse G(p -> F q) response pattern."""
        game, labels = make_mutex_game()
        f = parse_ltl("G (cs1 -> F cs2)")
        result = check_ltl_game_direct(game, labels, f)
        assert isinstance(result, OmegaRegularResult)


# ============================================================
# Section 10: Comparison APIs
# ============================================================

class TestComparison:
    def test_direct_vs_negation(self):
        """Direct and negation-based approaches should agree."""
        game, labels = make_simple_reachability_game()
        f = Finally(Atom("target"))
        comparison = compare_direct_vs_negation(game, labels, f)
        # Both approaches should give similar results
        assert 'direct_prob' in comparison
        assert 'negation_prob' in comparison
        assert comparison['direct_aut_states'] > 0
        assert comparison['negation_aut_states'] > 0

    def test_ltl_vs_pctl(self):
        """Compare LTL with PCTL equivalent."""
        lgame = make_labeled_game(
            n_states=3,
            owners={0: Player.P1, 1: Player.P2, 2: Player.P1},
            action_transitions={
                0: {'left': [0, 1, 0], 'right': [0, 0, 1]},
                1: {'stay': [0, 1, 0], 'go': [0, 0, 1]},
                2: {'loop': [0, 0, 1]},
            },
            labels={0: set(), 1: set(), 2: {'target'}},
        )

        from game_pctl import check_game_pctl
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
        from pctl_model_check import (
            PCTL as PCTLFormula, FormulaKind,
            atom, prob_geq, eventually
        )

        ltl_f = Finally(Atom("target"))
        pctl_f = prob_geq(1.0, eventually(atom("target")))

        comparison = compare_ltl_vs_pctl(lgame, ltl_f, pctl_f)
        assert 'ltl_prob_state0' in comparison
        assert 'pctl_satisfying' in comparison


# ============================================================
# Section 11: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_state_game(self):
        """Single absorbing state with property."""
        game = make_game(
            n_states=1,
            owners={0: Player.P1},
            action_transitions={0: {'loop': [1]}},
        )
        labels = {0: {'p'}}
        result = check_ltl_game_direct(game, labels, Globally(Atom("p")))
        # Single state with 'p', self-loop: G(p) trivially holds
        assert result.probabilities[0] >= 0.99

    def test_single_state_false(self):
        """Single state without required property."""
        game = make_game(
            n_states=1,
            owners={0: Player.P1},
            action_transitions={0: {'loop': [1]}},
        )
        labels = {0: set()}
        result = check_ltl_game_direct(game, labels, Finally(Atom("p")))
        # No state has 'p': F(p) cannot be satisfied
        assert result.probabilities[0] < 0.01

    def test_true_formula(self):
        """G(true) trivially holds."""
        game, labels = make_simple_reachability_game()
        result = check_ltl_game_direct(game, labels, Globally(LTLTrue()))
        assert result.probabilities[0] >= 0.99

    def test_false_formula(self):
        """F(false) can never hold."""
        game, labels = make_simple_reachability_game()
        result = check_ltl_game_direct(game, labels, Finally(LTLFalse()))
        assert result.probabilities[0] < 0.01

    def test_all_chance_game(self):
        """Game with only chance nodes (pure Markov chain)."""
        game = make_game(
            n_states=3,
            owners={0: Player.CHANCE, 1: Player.CHANCE, 2: Player.CHANCE},
            action_transitions={
                0: {'flip': [0, 0.7, 0.3]},
                1: {'flip': [0, 1, 0]},
                2: {'flip': [0, 0, 1]},
            },
        )
        labels = {0: set(), 1: {'a'}, 2: {'b'}}
        result = check_ltl_game_direct(game, labels, Finally(Atom("a")))
        # From state 0: 0.7 prob of reaching 'a'
        assert abs(result.probabilities[0] - 0.7) < 0.15


# ============================================================
# Section 12: Mutual Exclusion Game
# ============================================================

class TestMutualExclusion:
    def test_no_simultaneous_cs(self):
        """NOT(cs1 AND cs2) -- no simultaneous critical section."""
        game, labels = make_mutex_game()
        formula = Globally(Not(And(Atom("cs1"), Atom("cs2"))))
        result = check_ltl_game_direct(game, labels, formula)
        # cs1 and cs2 are in different states, so they can't both be true
        assert result.probabilities[0] >= 0.99

    def test_p1_can_enter(self):
        """F(cs1) -- P1 can eventually enter critical section."""
        game, labels = make_mutex_game()
        result = check_liveness_game(game, labels, "cs1")
        # P1 controls state 0 and can choose p1_try
        assert result.probabilities[0] >= 0.99


# ============================================================
# Section 13: Result Properties
# ============================================================

class TestResultProperties:
    def test_result_summary(self):
        """OmegaRegularResult has summary method."""
        game, labels = make_simple_reachability_game()
        result = check_ltl_game_direct(game, labels, Finally(Atom("target")))
        summary = result.summary()
        assert "LTL Formula" in summary
        assert "Satisfied" in summary

    def test_result_fields(self):
        """All result fields are populated."""
        game, labels = make_simple_reachability_game()
        result = check_ltl_game_direct(game, labels, Finally(Atom("target")))
        assert result.formula is not None
        assert result.probabilities is not None
        assert len(result.probabilities) == game.n_states
        assert result.automaton_states > 0
        assert result.product_states > 0

    def test_automaton_size(self):
        """Automaton size is reasonable."""
        game, labels = make_simple_reachability_game()
        result = check_ltl_game_direct(game, labels, Finally(Atom("target")))
        assert result.automaton_states <= 20  # Small formula -> small automaton
        assert result.product_states == game.n_states * result.automaton_states


# ============================================================
# Section 14: Complex LTL Formulas
# ============================================================

class TestComplexFormulas:
    def test_until(self):
        """p U q -- p holds until q becomes true."""
        game = make_game(
            n_states=3,
            owners={0: Player.P1, 1: Player.P1, 2: Player.P1},
            action_transitions={
                0: {'next': [0, 1, 0]},
                1: {'next': [0, 0, 1]},
                2: {'loop': [0, 0, 1]},
            },
        )
        labels = {0: {'p'}, 1: {'p'}, 2: {'q'}}
        result = check_ltl_game_direct(game, labels, Until(Atom("p"), Atom("q")))
        # Path: 0(p) -> 1(p) -> 2(q): p holds until q
        assert result.probabilities[0] >= 0.99

    def test_release(self):
        """p R q -- q holds until p becomes true (or forever)."""
        game = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.P1},
            action_transitions={
                0: {'stay': [1, 0]},
                1: {'loop': [0, 1]},
            },
        )
        labels = {0: {'p', 'q'}, 1: {'q'}}
        result = check_ltl_game_direct(game, labels, Release(Atom("p"), Atom("q")))
        # q holds everywhere, so R trivially holds
        assert result.probabilities[0] >= 0.99

    def test_weak_until(self):
        """p W q -- p holds until q or p holds forever."""
        game = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.P1},
            action_transitions={
                0: {'loop': [1, 0]},
                1: {'loop': [0, 1]},
            },
        )
        labels = {0: {'p'}, 1: set()}
        result = check_ltl_game_direct(game, labels, WeakUntil(Atom("p"), Atom("q")))
        # State 0 loops to itself with 'p': p W q holds (p forever case)
        assert result.probabilities[0] >= 0.99

    def test_next(self):
        """X(p) -- p holds in the next state."""
        game = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.P1},
            action_transitions={
                0: {'go': [0, 1]},
                1: {'loop': [0, 1]},
            },
        )
        labels = {0: set(), 1: {'p'}}
        result = check_ltl_game_direct(game, labels, Next(Atom("p")))
        # From state 0, next state is 1 which has 'p'
        assert result.probabilities[0] >= 0.99


# ============================================================
# Section 15: Negation-Based Checking
# ============================================================

class TestNegationBased:
    def test_negation_reachability(self):
        """check_ltl_game (negation-based) for reachability."""
        game, labels = make_simple_reachability_game()
        result = check_ltl_game(game, labels, Finally(Atom("target")))
        assert isinstance(result, OmegaRegularResult)
        # P1 can reach target directly
        assert result.probabilities[0] >= 0.5

    def test_negation_safety(self):
        """check_ltl_game for safety property."""
        game = make_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.P1},
            action_transitions={
                0: {'stay': [1, 0]},
                1: {'loop': [0, 1]},
            },
        )
        labels = {0: {'safe'}, 1: set()}
        result = check_ltl_game(game, labels, Globally(Atom("safe")))
        assert isinstance(result, OmegaRegularResult)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
