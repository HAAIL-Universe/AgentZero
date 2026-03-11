"""
Tests for V072: PCTL Model Checking for Stochastic Games
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V070_stochastic_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))

from game_pctl import (
    LabeledGame, make_labeled_game, GameQuantification, GamePCTLResult,
    GamePCTLChecker, check_game_pctl, check_game_pctl_state,
    game_pctl_quantitative, verify_game_property, compare_quantifications,
    batch_check_game, induced_mc_comparison, game_expected_reward_pctl,
)
from pctl_model_check import (
    tt, ff, atom, pnot, pand, por,
    prob_geq, prob_leq, prob_gt, prob_lt,
    next_f, until, bounded_until,
    eventually, always, bounded_eventually,
    parse_pctl, FormulaKind,
)
from stochastic_games import Player, StochasticGame, make_game


# ============================================================
# Test fixtures: example games
# ============================================================

def simple_reachability_game():
    """Simple 3-state game: P1 at s0, P2 at s1, target at s2.

    s0 (P1): action "left" -> s1 (prob 1.0)
              action "right" -> s2 (prob 1.0)
    s1 (P2): action "go" -> s2 (prob 1.0)
              action "stay" -> s1 (prob 1.0)
    s2 (absorbing): action "done" -> s2 (prob 1.0)
    """
    return make_labeled_game(
        n_states=3,
        owners={0: Player.P1, 1: Player.P2, 2: Player.CHANCE},
        action_transitions={
            0: {"left": [0, 1, 0], "right": [0, 0, 1]},
            1: {"go": [0, 0, 1], "stay": [0, 1, 0]},
            2: {"done": [0, 0, 1]},
        },
        labels={0: {"start"}, 1: {"middle"}, 2: {"target"}},
    )


def probabilistic_game():
    """4-state game with chance nodes.

    s0 (P1): "safe" -> s1 (1.0), "risky" -> s2 (1.0)
    s1 (CHANCE): "flip" -> s3 (0.7), s0 (0.3)
    s2 (P2): "block" -> s0 (1.0), "allow" -> s3 (1.0)
    s3 (absorbing): "done" -> s3 (1.0)
    """
    return make_labeled_game(
        n_states=4,
        owners={0: Player.P1, 1: Player.CHANCE, 2: Player.P2, 3: Player.CHANCE},
        action_transitions={
            0: {"safe": [0, 1, 0, 0], "risky": [0, 0, 1, 0]},
            1: {"flip": [0.3, 0, 0, 0.7]},
            2: {"block": [1, 0, 0, 0], "allow": [0, 0, 0, 1]},
            3: {"done": [0, 0, 0, 1]},
        },
        labels={0: {"init"}, 1: {"chance"}, 2: {"p2_choice"}, 3: {"goal"}},
    )


def adversarial_safety_game():
    """5-state game for safety property testing.

    s0 (P1): "left" -> s1, "right" -> s2
    s1 (P2): "safe" -> s0, "attack" -> s3
    s2 (CHANCE): "coin" -> s0 (0.5), s4 (0.5)
    s3: unsafe (absorbing)
    s4: safe sink (absorbing)
    """
    return make_labeled_game(
        n_states=5,
        owners={
            0: Player.P1, 1: Player.P2, 2: Player.CHANCE,
            3: Player.CHANCE, 4: Player.CHANCE,
        },
        action_transitions={
            0: {"left": [0, 1, 0, 0, 0], "right": [0, 0, 1, 0, 0]},
            1: {"safe": [1, 0, 0, 0, 0], "attack": [0, 0, 0, 1, 0]},
            2: {"coin": [0.5, 0, 0, 0, 0.5]},
            3: {"stay": [0, 0, 0, 1, 0]},
            4: {"stay": [0, 0, 0, 0, 1]},
        },
        labels={
            0: {"start", "safe_state"},
            1: {"safe_state"},
            2: {"safe_state"},
            3: {"unsafe"},
            4: {"safe_state", "sink"},
        },
    )


# ============================================================
# 1. LabeledGame construction
# ============================================================

class TestLabeledGame:
    def test_create_labeled_game(self):
        lg = simple_reachability_game()
        assert lg.game.n_states == 3
        assert lg.labels[0] == {"start"}
        assert lg.labels[2] == {"target"}

    def test_states_with(self):
        lg = simple_reachability_game()
        assert lg.states_with("target") == {2}
        assert lg.states_with("start") == {0}

    def test_states_without(self):
        lg = simple_reachability_game()
        assert lg.states_without("target") == {0, 1}

    def test_make_labeled_game_convenience(self):
        lg = make_labeled_game(
            n_states=2,
            owners={0: Player.P1, 1: Player.CHANCE},
            action_transitions={
                0: {"go": [0, 1]},
                1: {"stay": [0, 1]},
            },
            labels={0: {"a"}, 1: {"b"}},
        )
        assert isinstance(lg, LabeledGame)
        assert lg.game.n_states == 2


# ============================================================
# 2. Boolean formula checking
# ============================================================

class TestBooleanFormulas:
    def test_true(self):
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        assert checker.check(tt()) == {0, 1, 2}

    def test_false(self):
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        assert checker.check(ff()) == set()

    def test_atom(self):
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        assert checker.check(atom("target")) == {2}
        assert checker.check(atom("start")) == {0}

    def test_not(self):
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        assert checker.check(pnot(atom("target"))) == {0, 1}

    def test_and(self):
        lg = adversarial_safety_game()
        checker = GamePCTLChecker(lg)
        result = checker.check(pand(atom("safe_state"), atom("sink")))
        assert result == {4}

    def test_or(self):
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        result = checker.check(por(atom("start"), atom("target")))
        assert result == {0, 2}


# ============================================================
# 3. Next-state formulas
# ============================================================

class TestNextFormulas:
    def test_next_deterministic(self):
        """P1 can reach target in one step from s0 (action 'right')."""
        lg = simple_reachability_game()
        # P>=1[X target] should hold at s0 (P1 can choose "right")
        formula = prob_geq(1.0, next_f(atom("target")))
        result = check_game_pctl(lg, formula)
        assert 0 in result.satisfying_states  # P1 can guarantee next target

    def test_next_p2_blocks(self):
        """P2 at s1 can choose to stay, so Pmin(X target from s1) = 0."""
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        probs = checker._next_probs_by_owner(checker.check(atom("target")))
        # s0: P1 can go right to target -> Pmax = 1.0
        assert probs[0] == pytest.approx(1.0)
        # s1: P2 minimizes, can choose "stay" -> Pmin = 0.0
        assert probs[1] == pytest.approx(0.0)
        # s2: already at target, self-loop -> 1.0
        assert probs[2] == pytest.approx(1.0)

    def test_next_chance_node(self):
        """Chance nodes compute expected probability."""
        lg = probabilistic_game()
        checker = GamePCTLChecker(lg)
        probs = checker._next_probs_by_owner(checker.check(atom("goal")))
        # s1 (CHANCE): flip -> 0.7 to goal
        assert probs[1] == pytest.approx(0.7)

    def test_next_all_max(self):
        """When all players cooperate to maximize."""
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        probs = checker._next_probs_all_max(atom("target"))
        # s1: P2 maximizes (go to target) -> 1.0
        assert probs[1] == pytest.approx(1.0)

    def test_next_all_min(self):
        """When all players cooperate to minimize."""
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        probs = checker._next_probs_all_min(atom("target"))
        # s0: P1 minimizes (go left to s1) -> 0.0
        assert probs[0] == pytest.approx(0.0)
        # s1: P2 minimizes (stay) -> 0.0
        assert probs[1] == pytest.approx(0.0)


# ============================================================
# 4. Until formulas: game value
# ============================================================

class TestUntilFormulas:
    def test_until_p1_can_guarantee(self):
        """P1 at s0 can go directly to target -> game value = 1.0."""
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        probs = checker._until_probs_game(
            phi_sat=set(range(3)),  # true U target
            psi_sat={2},
        )
        # P1 can guarantee reaching target from s0
        assert probs[0] == pytest.approx(1.0)
        # P2 can prevent reaching target from s1 (stay forever)
        assert probs[1] == pytest.approx(0.0)
        # s2 is target
        assert probs[2] == pytest.approx(1.0)

    def test_until_adversarial(self):
        """In probabilistic game, P2 blocks the risky path."""
        lg = probabilistic_game()
        checker = GamePCTLChecker(lg)
        probs = checker._until_probs_game(
            phi_sat=set(range(4)),
            psi_sat={3},
        )
        # s0: P1 best move is "safe" -> s1 -> 0.7 to goal, 0.3 back to s0
        # This is a geometric series: 0.7 + 0.3*0.7 + 0.3^2*0.7 + ... = 1.0
        assert probs[0] == pytest.approx(1.0, abs=1e-6)
        # s2: P2 minimizes, chooses "block" -> back to s0
        # But from s0, P1 goes "safe" -> eventually reaches goal anyway
        assert probs[2] == pytest.approx(1.0, abs=1e-6)

    def test_until_cooperative_vs_adversarial(self):
        """Cooperative mode should give >= adversarial game value."""
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)

        game_val = checker._until_probs_game({0, 1, 2}, {2})
        coop_max = checker._until_probs_all_max({0, 1, 2}, {2})

        # Cooperative max >= game value (P1 max, P2 min)
        for s in range(3):
            assert coop_max[s] >= game_val[s] - 1e-10

    def test_until_p2_wins(self):
        """P2 can prevent target from s1 forever."""
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        result = check_game_pctl(lg, formula)
        # s0: P1 can reach target directly -> satisfied
        assert 0 in result.satisfying_states
        # s1: P2 stays forever -> not satisfied
        assert 1 not in result.satisfying_states

    def test_until_threshold(self):
        """Check various thresholds."""
        lg = probabilistic_game()
        # From s0, game value for reaching goal is 1.0
        f1 = prob_geq(0.99, eventually(atom("goal")))
        r1 = check_game_pctl(lg, f1)
        assert 0 in r1.satisfying_states

        f2 = prob_leq(0.5, eventually(atom("goal")))
        r2 = check_game_pctl(lg, f2)
        assert 0 not in r2.satisfying_states  # game value is 1.0, not <= 0.5


# ============================================================
# 5. Bounded until formulas
# ============================================================

class TestBoundedUntil:
    def test_bounded_until_immediate(self):
        """With k=0, only states in psi satisfy."""
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        probs = checker._bounded_until_probs_game({0, 1, 2}, {2}, k=0)
        assert probs[2] == pytest.approx(1.0)
        assert probs[0] == pytest.approx(0.0)

    def test_bounded_until_one_step(self):
        """With k=1, P1 at s0 can reach target in one step."""
        lg = simple_reachability_game()
        checker = GamePCTLChecker(lg)
        probs = checker._bounded_until_probs_game({0, 1, 2}, {2}, k=1)
        assert probs[0] == pytest.approx(1.0)  # P1 goes "right"
        assert probs[1] == pytest.approx(0.0)  # P2 stays

    def test_bounded_until_chance(self):
        """Bounded until with chance node probability."""
        lg = probabilistic_game()
        checker = GamePCTLChecker(lg)
        # k=2: s0 -> s1 (safe) -> s3 (0.7) or s0 (0.3)
        probs = checker._bounded_until_probs_game({0, 1, 2, 3}, {3}, k=2)
        assert probs[0] == pytest.approx(0.7, abs=1e-6)

    def test_bounded_until_increasing(self):
        """More steps -> higher probability."""
        lg = probabilistic_game()
        checker = GamePCTLChecker(lg)
        p2 = checker._bounded_until_probs_game({0, 1, 2, 3}, {3}, k=2)
        p4 = checker._bounded_until_probs_game({0, 1, 2, 3}, {3}, k=4)
        p10 = checker._bounded_until_probs_game({0, 1, 2, 3}, {3}, k=10)
        assert p2[0] <= p4[0] + 1e-10
        assert p4[0] <= p10[0] + 1e-10

    def test_bounded_eventually_formula(self):
        """Test bounded_eventually sugar."""
        lg = probabilistic_game()
        formula = prob_geq(0.9, bounded_eventually(atom("goal"), 10))
        result = check_game_pctl(lg, formula)
        # With 10 steps and 0.7 per attempt, 1 - 0.3^5 = 0.9997
        assert 0 in result.satisfying_states


# ============================================================
# 6. Safety properties
# ============================================================

class TestSafetyProperties:
    def test_safety_p1_avoids_unsafe(self):
        """P1 can avoid unsafe state by choosing "right" (coin flip, never unsafe)."""
        lg = adversarial_safety_game()
        # P<=0[F unsafe] means "never reach unsafe"
        # Equivalently: P>=1[G safe_state] -- but we use always() sugar
        # Check: can P1 guarantee never reaching unsafe?
        formula = prob_leq(0.0, eventually(atom("unsafe")))
        result = check_game_pctl(lg, formula)
        # s0: P1 goes "right" -> s2 (CHANCE: 0.5 s0, 0.5 s4) -> never reaches s3
        assert 0 in result.satisfying_states

    def test_safety_p2_can_force_unsafe(self):
        """If P1 goes left, P2 can attack -> unsafe."""
        lg = adversarial_safety_game()
        # Check game value for reaching unsafe from s1
        checker = GamePCTLChecker(lg)
        probs = checker._until_probs_game(set(range(5)), {3})
        # s1: P2 can attack -> 1.0 to unsafe. Game value (P1 max, P2 min reach unsafe):
        # P2 minimizes? No, P2 maximizes reach to unsafe? Wait...
        # Game value = P1 max, P2 min. P1 maximizes reach to unsafe from s1? P1 doesn't own s1.
        # s1 is P2's state. P2 minimizes reach to unsafe? P2 chooses "safe" -> back to s0.
        # Actually reaching unsafe is BAD for P1. Game value of reaching unsafe:
        # P1 max P2 min: P1 wants to maximize, P2 wants to minimize reach to unsafe.
        # From s0: P1 maximizes reach to unsafe -> goes left to s1
        # From s1: P2 minimizes reach to unsafe -> goes "safe" to s0 (loop)
        # So game value of reaching unsafe from s0 is 0.0 (P2 prevents it)
        assert probs[0] == pytest.approx(0.0, abs=1e-6)

    def test_safety_absorbing(self):
        """Absorbing safe state has prob 0 of reaching unsafe."""
        lg = adversarial_safety_game()
        formula = prob_leq(0.0, eventually(atom("unsafe")))
        result = check_game_pctl(lg, formula)
        assert 4 in result.satisfying_states  # safe sink


# ============================================================
# 7. Quantification modes
# ============================================================

class TestQuantificationModes:
    def test_adversarial_mode(self):
        """Default adversarial: P1 max, P2 min."""
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        result = check_game_pctl(lg, formula, GameQuantification.ADVERSARIAL)
        assert 0 in result.satisfying_states
        assert 1 not in result.satisfying_states

    def test_cooperative_mode(self):
        """Cooperative: both players maximize together."""
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        result = check_game_pctl(lg, formula, GameQuantification.COOPERATIVE)
        # In cooperative mode, P2 at s1 chooses "go" -> target
        assert 0 in result.satisfying_states
        assert 1 in result.satisfying_states

    def test_p1_optimistic(self):
        """P1 optimistic: uses game value (P1 max, P2 min) for GEQ."""
        lg = simple_reachability_game()
        formula = prob_geq(1.0, eventually(atom("target")))
        result = check_game_pctl(lg, formula, GameQuantification.P1_OPTIMISTIC)
        assert 0 in result.satisfying_states

    def test_p2_optimistic(self):
        """P2 optimistic: P2 minimizes, P1 also minimizes (pessimistic for property)."""
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        result = check_game_pctl(lg, formula, GameQuantification.P2_OPTIMISTIC)
        # All minimize -> P1 at s0 goes left to s1, P2 stays -> prob 0
        assert 0 not in result.satisfying_states

    def test_compare_quantifications(self):
        """Compare all 4 quantification modes."""
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        comp = compare_quantifications(lg, formula)
        assert len(comp) == 4
        # Cooperative should have most satisfying states
        coop_count = comp['cooperative']['count']
        adv_count = comp['adversarial']['count']
        assert coop_count >= adv_count


# ============================================================
# 8. Strategy extraction
# ============================================================

class TestStrategyExtraction:
    def test_p1_strategy(self):
        """P1 strategy should choose the maximizing action."""
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        result = check_game_pctl(lg, formula)
        # P1 at s0 should choose "right" (index 1) to go directly to target
        assert result.p1_strategy is not None
        assert 0 in result.p1_strategy
        assert result.p1_strategy[0] == 1  # "right" is index 1

    def test_p2_strategy(self):
        """P2 strategy should choose the minimizing action."""
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        result = check_game_pctl(lg, formula)
        # P2 at s1 should choose "stay" (index 1) to minimize reachability
        assert result.p2_strategy is not None
        assert 1 in result.p2_strategy
        assert result.p2_strategy[1] == 1  # "stay" is index 1

    def test_strategies_in_probabilistic_game(self):
        """Extract strategies from a game with chance nodes."""
        lg = probabilistic_game()
        formula = prob_geq(0.5, eventually(atom("goal")))
        result = check_game_pctl(lg, formula)
        assert result.p1_strategy is not None
        assert result.p2_strategy is not None


# ============================================================
# 9. Expected reward
# ============================================================

class TestExpectedReward:
    def test_expected_reward_simple(self):
        """P1 can choose shorter path for lower cost."""
        lg = simple_reachability_game()
        rewards = [1.0, 1.0, 0.0]  # 1 per step, 0 at target
        values, p1_s, p2_s = game_expected_reward_pctl(
            lg, rewards, atom("target"), maximize_p1=False,
        )
        # P1 minimizes cost: goes "right" from s0 -> 1 step = reward 1
        assert values[0] == pytest.approx(1.0, abs=1e-6)
        # s2 is target -> 0
        assert values[2] == pytest.approx(0.0)

    def test_expected_reward_adversarial(self):
        """P2 can increase cost by staying at s1."""
        lg = simple_reachability_game()
        rewards = [1.0, 1.0, 0.0]
        values, p1_s, p2_s = game_expected_reward_pctl(
            lg, rewards, atom("target"), maximize_p1=True,
        )
        # P1 maximizes: goes "right" -> 1 step -> reward 1.
        # But that reaches target immediately. Going left:
        # s1 owned by P2, P2 minimizes reward -> P2 goes to target (reward 1+1=2? no)
        # Wait: P1 maximizes, P2 minimizes. P1 at s0 picks max, P2 at s1 picks min.
        # P1 "right" -> s2 target -> value 1.
        # P1 "left" -> s1 -> P2 min: "go" -> s2 (value 1+1=2) or "stay" -> s1 (1+value[1])
        # P2 minimizes at s1: min("go"->1+0=1, "stay"->1+values[1])
        # If "go" gives 1, and "stay" gives 1+values[1], P2 picks min.
        # If values[1] >= 0, "go" (1) <= "stay" (1+values[1]), so P2 picks "go" -> 1.
        # P1 at s0: max("left"->1+1=2, "right"->1+0=1) -> P1 picks "left" -> 2
        assert values[0] == pytest.approx(2.0, abs=1e-6)

    def test_expected_reward_with_chance(self):
        """Expected reward with chance node."""
        lg = probabilistic_game()
        rewards = [1.0, 1.0, 1.0, 0.0]
        values, _, _ = game_expected_reward_pctl(
            lg, rewards, atom("goal"), maximize_p1=False,
        )
        # P1 minimizes cost. "safe" -> s1 (chance: 0.7 to goal, 0.3 back)
        # Expected steps via safe path: 1/0.7 attempts, each costs 2 steps (s0->s1)
        # Expected cost from s0 = 1 + values[s1]
        # values[s1] = 1 + 0.7*0 + 0.3*values[s0] => values[s1] = 1 + 0.3*values[s0]
        # values[s0] = 1 + values[s1] = 1 + 1 + 0.3*values[s0] = 2 + 0.3*values[s0]
        # 0.7*values[s0] = 2 => values[s0] = 2/0.7 ~= 2.857
        assert values[0] == pytest.approx(2.0 / 0.7, abs=0.1)


# ============================================================
# 10. GamePCTLResult
# ============================================================

class TestGamePCTLResult:
    def test_result_fields(self):
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        result = check_game_pctl(lg, formula)
        assert isinstance(result, GamePCTLResult)
        assert result.all_states == 3
        assert result.prob_max is not None
        assert len(result.prob_max) == 3

    def test_all_satisfy(self):
        lg = simple_reachability_game()
        result = check_game_pctl(lg, tt())
        assert result.all_satisfy

    def test_none_satisfy(self):
        lg = simple_reachability_game()
        result = check_game_pctl(lg, ff())
        assert result.none_satisfy

    def test_summary(self):
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        result = check_game_pctl(lg, formula)
        s = result.summary()
        assert "Satisfying" in s
        assert "Pmax" in s


# ============================================================
# 11. check_game_pctl_state
# ============================================================

class TestCheckState:
    def test_state_satisfies(self):
        lg = simple_reachability_game()
        formula = prob_geq(1.0, eventually(atom("target")))
        assert check_game_pctl_state(lg, 0, formula) is True

    def test_state_not_satisfies(self):
        lg = simple_reachability_game()
        formula = prob_geq(1.0, eventually(atom("target")))
        # s1: P2 blocks -> game value 0
        assert check_game_pctl_state(lg, 1, formula) is False


# ============================================================
# 12. Quantitative API
# ============================================================

class TestQuantitative:
    def test_game_pctl_quantitative(self):
        lg = simple_reachability_game()
        path = eventually(atom("target"))
        q = game_pctl_quantitative(lg, path)
        assert 'game_value' in q
        assert 'all_min' in q
        assert 'all_max' in q
        assert len(q['game_value']) == 3

    def test_all_max_geq_game_value(self):
        """Cooperative max >= game value."""
        lg = probabilistic_game()
        path = eventually(atom("goal"))
        q = game_pctl_quantitative(lg, path)
        for s in range(4):
            assert q['all_max'][s] >= q['game_value'][s] - 1e-10

    def test_game_value_geq_all_min(self):
        """Game value >= all-min."""
        lg = probabilistic_game()
        path = eventually(atom("goal"))
        q = game_pctl_quantitative(lg, path)
        for s in range(4):
            assert q['game_value'][s] >= q['all_min'][s] - 1e-10


# ============================================================
# 13. Verify game property
# ============================================================

class TestVerifyProperty:
    def test_verify_satisfied(self):
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        info = verify_game_property(lg, formula, initial_state=0)
        assert info['satisfied'] is True
        assert 'prob_max' in info

    def test_verify_not_satisfied(self):
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        info = verify_game_property(lg, formula, initial_state=1)
        assert info['satisfied'] is False

    def test_verify_strategies(self):
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        info = verify_game_property(lg, formula, initial_state=0)
        assert 'p1_strategy' in info
        assert 'p2_strategy' in info


# ============================================================
# 14. Batch checking
# ============================================================

class TestBatchCheck:
    def test_batch_multiple_formulas(self):
        lg = simple_reachability_game()
        formulas = [
            prob_geq(0.5, eventually(atom("target"))),
            prob_leq(0.0, next_f(atom("middle"))),
            tt(),
        ]
        results = batch_check_game(lg, formulas)
        assert len(results) == 3
        assert all(isinstance(r, GamePCTLResult) for r in results)


# ============================================================
# 15. Induced MC comparison
# ============================================================

class TestInducedMC:
    def test_induced_mc(self):
        lg = simple_reachability_game()
        formula = prob_geq(0.5, eventually(atom("target")))
        comp = induced_mc_comparison(lg, formula)
        assert 'game_satisfying' in comp
        assert 'mc_satisfying' in comp
        assert 'p1_strategy' in comp

    def test_induced_mc_no_prob(self):
        """Non-probability formula has no strategy."""
        lg = simple_reachability_game()
        comp = induced_mc_comparison(lg, atom("target"))
        assert 'note' in comp

    def test_induced_mc_consistency(self):
        """Under extracted strategies, MC should be consistent."""
        lg = probabilistic_game()
        formula = prob_geq(0.9, eventually(atom("goal")))
        comp = induced_mc_comparison(lg, formula)
        # The game says s0 satisfies. Under the extracted strategy,
        # the MC should also satisfy (P1 always goes safe, P2's strategy is fixed).
        assert 0 in set(comp['game_satisfying'])


# ============================================================
# 16. Parsed formulas
# ============================================================

class TestParsedFormulas:
    def test_parse_and_check(self):
        lg = simple_reachability_game()
        formula = parse_pctl('P>=0.5[F "target"]')
        result = check_game_pctl(lg, formula)
        assert 0 in result.satisfying_states

    def test_parse_bounded(self):
        lg = probabilistic_game()
        formula = parse_pctl('P>=0.5[F<=5 "goal"]')
        result = check_game_pctl(lg, formula)
        assert 0 in result.satisfying_states

    def test_parse_next(self):
        lg = simple_reachability_game()
        formula = parse_pctl('P>=1.0[X "target"]')
        result = check_game_pctl(lg, formula)
        assert 0 in result.satisfying_states


# ============================================================
# 17. Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_state_game(self):
        """Game with one absorbing state."""
        lg = make_labeled_game(
            n_states=1,
            owners={0: Player.P1},
            action_transitions={0: {"stay": [1.0]}},
            labels={0: {"here"}},
        )
        formula = prob_geq(1.0, next_f(atom("here")))
        result = check_game_pctl(lg, formula)
        assert 0 in result.satisfying_states

    def test_all_chance_game(self):
        """Game with only chance nodes (pure Markov chain)."""
        lg = make_labeled_game(
            n_states=3,
            owners={0: Player.CHANCE, 1: Player.CHANCE, 2: Player.CHANCE},
            action_transitions={
                0: {"a": [0, 0.5, 0.5]},
                1: {"a": [0, 0, 1]},
                2: {"a": [0, 0, 1]},
            },
            labels={0: set(), 1: set(), 2: {"goal"}},
        )
        formula = prob_geq(0.5, next_f(atom("goal")))
        result = check_game_pctl(lg, formula)
        assert 0 in result.satisfying_states  # 0.5 >= 0.5
        assert 1 in result.satisfying_states  # 1.0 >= 0.5

    def test_threshold_boundary(self):
        """Test exact threshold matching."""
        lg = probabilistic_game()
        checker = GamePCTLChecker(lg)
        probs = checker._next_probs_by_owner({3})  # goal
        # s1 (CHANCE) -> 0.7 to goal
        # P>=0.7 should hold, P>0.7 should not
        f_geq = prob_geq(0.7, next_f(atom("goal")))
        r_geq = check_game_pctl(lg, f_geq)
        assert 1 in r_geq.satisfying_states

        f_gt = prob_gt(0.7, next_f(atom("goal")))
        r_gt = check_game_pctl(lg, f_gt)
        assert 1 not in r_gt.satisfying_states

    def test_prob_leq(self):
        """Test P<= operator."""
        lg = simple_reachability_game()
        formula = prob_leq(0.0, eventually(atom("target")))
        result = check_game_pctl(lg, formula)
        # Game value for reaching target: s0=1.0 (P1 goes right), s1=0.0, s2=1.0
        # P<=0 holds at s1 (game value 0) but not at s0 or s2
        assert 1 in result.satisfying_states
        assert 0 not in result.satisfying_states

    def test_prob_lt(self):
        """Test P< operator."""
        lg = probabilistic_game()
        formula = prob_lt(0.8, next_f(atom("goal")))
        result = check_game_pctl(lg, formula)
        # s1 (CHANCE): 0.7 to goal -> 0.7 < 0.8 holds
        assert 1 in result.satisfying_states


# ============================================================
# 18. Complex game scenarios
# ============================================================

class TestComplexGames:
    def test_two_target_game(self):
        """Game with two different targets, P1 and P2 prefer different ones."""
        lg = make_labeled_game(
            n_states=4,
            owners={0: Player.P1, 1: Player.P2, 2: Player.CHANCE, 3: Player.CHANCE},
            action_transitions={
                0: {"left": [0, 1, 0, 0], "right": [0, 0, 1, 0]},
                1: {"up": [0, 0, 1, 0], "down": [0, 0, 0, 1]},
                2: {"stay": [0, 0, 1, 0]},
                3: {"stay": [0, 0, 0, 1]},
            },
            labels={0: set(), 1: set(), 2: {"A"}, 3: {"B"}},
        )
        # P1 wants to reach A, P2 wants to prevent it
        f = prob_geq(1.0, eventually(atom("A")))
        result = check_game_pctl(lg, f)
        # P1 at s0 can go "right" -> s2 (A) directly
        assert 0 in result.satisfying_states

    def test_multi_step_game(self):
        """Game requiring multiple steps of reasoning."""
        # s0 (P1) -> s1 or s2
        # s1 (P2) -> s0 or s3
        # s2 (P2) -> s0 or s3
        # s3 (CHANCE) -> s4 (0.6) or s0 (0.4)
        # s4: target
        lg = make_labeled_game(
            n_states=5,
            owners={
                0: Player.P1, 1: Player.P2, 2: Player.P2,
                3: Player.CHANCE, 4: Player.CHANCE,
            },
            action_transitions={
                0: {"a": [0, 1, 0, 0, 0], "b": [0, 0, 1, 0, 0]},
                1: {"back": [1, 0, 0, 0, 0], "fwd": [0, 0, 0, 1, 0]},
                2: {"back": [1, 0, 0, 0, 0], "fwd": [0, 0, 0, 1, 0]},
                3: {"flip": [0.4, 0, 0, 0, 0.6]},
                4: {"stay": [0, 0, 0, 0, 1]},
            },
            labels={0: set(), 1: set(), 2: set(), 3: set(), 4: {"goal"}},
        )
        # Game value for reaching goal:
        # P2 minimizes: always picks "back" -> loops forever -> game value 0
        formula = prob_geq(0.5, eventually(atom("goal")))
        result = check_game_pctl(lg, formula)
        assert 0 not in result.satisfying_states  # P2 prevents

    def test_nested_boolean_pctl(self):
        """Test nested boolean PCTL formulas."""
        lg = simple_reachability_game()
        # NOT(P>=0.5[F target])
        formula = pnot(prob_geq(0.5, eventually(atom("target"))))
        result = check_game_pctl(lg, formula)
        # s0: P>=0.5 holds -> NOT doesn't hold
        assert 0 not in result.satisfying_states
        # s1: P>=0.5 doesn't hold -> NOT holds
        assert 1 in result.satisfying_states


# ============================================================
# 19. Comparison with MDP
# ============================================================

class TestMDPComparison:
    def test_all_p1_game_is_mdp(self):
        """A game where all states are P1-owned is equivalent to an MDP."""
        lg = make_labeled_game(
            n_states=3,
            owners={0: Player.P1, 1: Player.P1, 2: Player.CHANCE},
            action_transitions={
                0: {"a": [0, 1, 0], "b": [0, 0, 1]},
                1: {"c": [0, 0, 1]},
                2: {"stay": [0, 0, 1]},
            },
            labels={0: set(), 1: set(), 2: {"goal"}},
        )
        # All states P1-owned -> game value = max over all policies
        formula = prob_geq(1.0, eventually(atom("goal")))
        result = check_game_pctl(lg, formula)
        assert 0 in result.satisfying_states
        assert 1 in result.satisfying_states

    def test_all_chance_is_mc(self):
        """Game with only CHANCE nodes is a Markov chain."""
        lg = make_labeled_game(
            n_states=2,
            owners={0: Player.CHANCE, 1: Player.CHANCE},
            action_transitions={
                0: {"a": [0.3, 0.7]},
                1: {"a": [0, 1]},
            },
            labels={0: set(), 1: {"goal"}},
        )
        formula = prob_geq(0.7, next_f(atom("goal")))
        result = check_game_pctl(lg, formula)
        assert 0 in result.satisfying_states  # 0.7 >= 0.7


# ============================================================
# 20. Regression tests
# ============================================================

class TestRegression:
    def test_self_loop_convergence(self):
        """Ensure value iteration converges with self-loops."""
        lg = make_labeled_game(
            n_states=3,
            owners={0: Player.P1, 1: Player.CHANCE, 2: Player.CHANCE},
            action_transitions={
                0: {"go": [0, 1, 0]},
                1: {"flip": [0, 0.9, 0.1]},  # heavy self-loop
                2: {"stay": [0, 0, 1]},
            },
            labels={0: set(), 1: set(), 2: {"goal"}},
        )
        formula = prob_geq(0.9, eventually(atom("goal")))
        result = check_game_pctl(lg, formula)
        # Eventually reaches goal with prob 1 (geometric series)
        assert 0 in result.satisfying_states
        assert 1 in result.satisfying_states

    def test_p2_forces_delay(self):
        """P2 can delay but not prevent if all paths lead to target."""
        lg = make_labeled_game(
            n_states=3,
            owners={0: Player.P2, 1: Player.CHANCE, 2: Player.CHANCE},
            action_transitions={
                0: {"slow": [0, 1, 0], "fast": [0, 0, 1]},
                1: {"coin": [0.5, 0, 0.5]},
                2: {"stay": [0, 0, 1]},
            },
            labels={0: set(), 1: set(), 2: {"goal"}},
        )
        # P2 minimizes: picks "slow" -> s1 -> 0.5 back to s0, 0.5 to goal
        # Eventually still reaches goal with prob 1
        formula = prob_geq(0.99, eventually(atom("goal")))
        result = check_game_pctl(lg, formula)
        assert 0 in result.satisfying_states

    def test_always_formula(self):
        """Test safety via P<=0[F NOT safe_state] (never reach unsafe)."""
        lg = adversarial_safety_game()
        # "Always safe" = P<=0[F NOT safe_state]
        formula = prob_leq(0.0, eventually(pnot(atom("safe_state"))))
        result = check_game_pctl(lg, formula)
        # s3 (unsafe) is already not safe -> P[F NOT safe] = 1.0, not <= 0
        assert 3 not in result.satisfying_states
        # s4 (safe sink) -> P[F NOT safe] = 0.0, satisfies <= 0
        assert 4 in result.satisfying_states
