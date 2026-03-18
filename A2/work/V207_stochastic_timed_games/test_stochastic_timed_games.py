"""Tests for V207: Stochastic Timed Games."""

import pytest
import math
from stochastic_timed_games import (
    # DBM / Zone
    Zone, make_zone, make_zero_zone, constrain_zone, apply_guard,
    reset_clocks, future, past, apply_invariant, successor_zone,
    backward_zone, CompOp, ClockConstraint, Guard, true_guard,
    clock_lt, clock_le, clock_gt, clock_ge, clock_eq, guard_and,
    # Game
    StochasticTimedGame, PlayerType, Edge,
    # Solvers
    solve_positive_prob_reachability, solve_almost_sure_reachability,
    solve_stochastic_timed_reachability, solve_stochastic_timed_safety,
    solve_expected_time, solve_qualitative_buchi,
    # Utilities
    explore_reachable, game_statistics, compare_as_pp,
    simulate_play, SimTrace, SimStep,
    StochasticTimedResult, ExpectedTimeResult,
    # Zone helpers
    _zones_include, _zones_union_add, _zones_equal,
    ZoneStore, SymState,
    # Example games
    make_coin_flip_game, make_probabilistic_traffic,
    make_adversarial_random_game, make_retry_game,
    make_two_player_stochastic,
)


# ============================================================
# Zone / DBM Tests
# ============================================================

class TestZoneBasics:
    def test_make_zone(self):
        z = make_zone(["x", "y"])
        assert not z.is_empty()
        assert z.n == 2

    def test_make_zero_zone(self):
        z = make_zero_zone(["x"])
        assert not z.is_empty()
        # x = 0
        assert z.dbm[1][0] == (0, False)
        assert z.dbm[0][1] == (0, False)

    def test_constrain_le(self):
        z = make_zone(["x"])
        z = constrain_zone(z, clock_le("x", 5))
        assert not z.is_empty()
        assert z.dbm[1][0][0] <= 5

    def test_constrain_ge(self):
        z = make_zone(["x"])
        z = constrain_zone(z, clock_ge("x", 2))
        assert not z.is_empty()

    def test_constrain_empty(self):
        z = make_zone(["x"])
        z = constrain_zone(z, clock_ge("x", 5))
        z = constrain_zone(z, clock_le("x", 3))
        assert z.is_empty()

    def test_constrain_eq(self):
        z = make_zone(["x"])
        z = constrain_zone(z, clock_eq("x", 3))
        assert not z.is_empty()

    def test_guard_and(self):
        g = guard_and(clock_ge("x", 1), clock_le("x", 3))
        z = make_zone(["x"])
        z = apply_guard(z, g)
        assert not z.is_empty()

    def test_true_guard(self):
        g = true_guard()
        assert len(g.constraints) == 0

    def test_zone_copy(self):
        z = make_zone(["x"])
        z2 = z.copy()
        assert z == z2
        z2 = constrain_zone(z2, clock_le("x", 5))
        assert z != z2

    def test_zone_includes(self):
        z1 = make_zone(["x"])
        z2 = constrain_zone(make_zone(["x"]), clock_le("x", 5))
        assert z1.includes(z2)
        assert not z2.includes(z1)

    def test_zone_intersect(self):
        z1 = constrain_zone(make_zone(["x"]), clock_le("x", 5))
        z2 = constrain_zone(make_zone(["x"]), clock_ge("x", 3))
        z3 = z1.intersect(z2)
        assert not z3.is_empty()

    def test_zone_intersect_empty(self):
        z1 = constrain_zone(make_zone(["x"]), clock_le("x", 2))
        z2 = constrain_zone(make_zone(["x"]), clock_ge("x", 5))
        z3 = z1.intersect(z2)
        assert z3.is_empty()

    def test_zone_equality(self):
        z1 = constrain_zone(make_zone(["x"]), clock_le("x", 5))
        z2 = constrain_zone(make_zone(["x"]), clock_le("x", 5))
        assert z1 == z2

    def test_zone_hash(self):
        z1 = constrain_zone(make_zone(["x"]), clock_le("x", 5))
        z2 = constrain_zone(make_zone(["x"]), clock_le("x", 5))
        assert hash(z1) == hash(z2)


class TestZoneOperations:
    def test_reset_clocks(self):
        z = constrain_zone(make_zone(["x", "y"]), clock_eq("x", 3))
        z = reset_clocks(z, frozenset({"x"}))
        assert not z.is_empty()
        # x should be 0 after reset
        assert z.dbm[1][0] == (0, False)

    def test_future(self):
        z = make_zero_zone(["x"])
        z = future(z)
        assert not z.is_empty()
        # x can be any non-negative value
        assert z.dbm[1][0] == (float('inf'), False)

    def test_past(self):
        z = constrain_zone(make_zone(["x"]), clock_eq("x", 5))
        z = past(z)
        assert not z.is_empty()

    def test_successor_zone(self):
        z = make_zero_zone(["x"])
        z = future(z)
        g = Guard(constraints=(clock_le("x", 3),))
        inv = Guard(constraints=(clock_le("x", 5),))
        sz = successor_zone(z, g, frozenset({"x"}), inv)
        assert sz is not None
        assert not sz.is_empty()

    def test_successor_zone_infeasible(self):
        z = constrain_zone(make_zone(["x"]), clock_ge("x", 5))
        g = Guard(constraints=(clock_le("x", 2),))
        sz = successor_zone(z, g, frozenset(), true_guard())
        assert sz is None

    def test_backward_zone(self):
        target_z = constrain_zone(make_zone(["x"]), clock_le("x", 3))
        g = Guard(constraints=(clock_le("x", 5),))
        src_inv = Guard(constraints=(clock_le("x", 10),))
        bz = backward_zone(target_z, g, frozenset({"x"}), src_inv, ["x"])
        assert bz is not None
        assert not bz.is_empty()

    def test_apply_invariant(self):
        z = make_zone(["x"])
        inv = Guard(constraints=(clock_le("x", 10),))
        z = apply_invariant(z, inv)
        assert not z.is_empty()

    def test_two_clocks(self):
        z = make_zone(["x", "y"])
        z = constrain_zone(z, clock_le("x", 3))
        z = constrain_zone(z, clock_ge("y", 1))
        assert not z.is_empty()


class TestZoneHelpers:
    def test_zones_include(self):
        z1 = make_zone(["x"])
        z2 = constrain_zone(make_zone(["x"]), clock_le("x", 5))
        assert _zones_include([z1], z2)
        assert not _zones_include([z2], z1)

    def test_zones_union_add(self):
        z1 = constrain_zone(make_zone(["x"]), clock_le("x", 5))
        z2 = constrain_zone(make_zone(["x"]), clock_le("x", 3))
        result = _zones_union_add([], z1)
        assert len(result) == 1
        # z2 subsumed by z1
        result = _zones_union_add(result, z2)
        assert len(result) == 1

    def test_zones_union_add_non_subsumed(self):
        z1 = constrain_zone(make_zone(["x"]), clock_le("x", 3))
        z2 = constrain_zone(make_zone(["x"]), clock_ge("x", 5))
        result = _zones_union_add([z1], z2)
        assert len(result) == 2

    def test_zones_equal(self):
        z1 = make_zone(["x"])
        assert _zones_equal([z1], [z1])
        assert _zones_equal([], [])

    def test_zone_store(self):
        store = ZoneStore()
        z1 = make_zone(["x"])
        z2 = make_zone(["x"])
        id1 = store.add(z1)
        id2 = store.add(z2)
        assert id1 == id2
        assert len(store) == 1


# ============================================================
# Game Construction Tests
# ============================================================

class TestGameConstruction:
    def test_empty_game(self):
        g = StochasticTimedGame()
        assert len(g.locations) == 0

    def test_add_location(self):
        g = StochasticTimedGame()
        g.add_location("s0", PlayerType.MIN)
        assert "s0" in g.locations
        assert g.owner["s0"] == PlayerType.MIN

    def test_add_edge(self):
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        idx = g.add_edge("s0", "s1", guard=Guard(
            constraints=(clock_le("x", 3),)))
        assert idx == 0
        assert len(g.edges) == 1

    def test_add_probabilistic_edge(self):
        g = StochasticTimedGame()
        g.add_location("r", PlayerType.RANDOM)
        g.add_location("a", PlayerType.MIN)
        g.add_location("b", PlayerType.MIN)
        i1 = g.add_edge("r", "a", probability=0.6)
        i2 = g.add_edge("r", "b", probability=0.4)
        assert g.get_probability(i1) == 0.6
        assert g.get_probability(i2) == 0.4

    def test_get_edges_from(self):
        g = StochasticTimedGame()
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        g.add_location("s2", PlayerType.MIN)
        g.add_edge("s0", "s1")
        g.add_edge("s0", "s2")
        g.add_edge("s1", "s2")
        edges = g.get_edges_from("s0")
        assert len(edges) == 2

    def test_get_invariant_default(self):
        g = StochasticTimedGame()
        g.add_location("s0", PlayerType.MIN)
        inv = g.get_invariant("s0")
        assert len(inv.constraints) == 0

    def test_get_invariant_set(self):
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.add_location("s0", PlayerType.MIN, invariant=Guard(
            constraints=(clock_le("x", 5),)))
        inv = g.get_invariant("s0")
        assert len(inv.constraints) == 1

    def test_max_constant(self):
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.add_location("s0", PlayerType.MIN, invariant=Guard(
            constraints=(clock_le("x", 10),)))
        g.add_location("s1", PlayerType.MIN)
        g.add_edge("s0", "s1", guard=Guard(
            constraints=(clock_ge("x", 3),)))
        assert g.max_constant() == 10

    def test_validate_ok(self):
        g = make_coin_flip_game()
        errors = g.validate()
        assert len(errors) == 0

    def test_validate_bad_initial(self):
        g = StochasticTimedGame()
        g.initial = "nonexistent"
        g.add_location("s0", PlayerType.MIN)
        errors = g.validate()
        assert any("Initial" in e for e in errors)

    def test_validate_bad_probabilities(self):
        g = StochasticTimedGame()
        g.add_location("r", PlayerType.RANDOM)
        g.add_location("a", PlayerType.MIN)
        g.add_edge("r", "a", probability=0.5)
        errors = g.validate()
        assert any("sum" in e for e in errors)

    def test_validate_random_no_edges(self):
        g = StochasticTimedGame()
        g.add_location("r", PlayerType.RANDOM)
        errors = g.validate()
        assert any("no outgoing" in e for e in errors)


# ============================================================
# Example Game Tests
# ============================================================

class TestExampleGames:
    def test_coin_flip_valid(self):
        g = make_coin_flip_game()
        assert len(g.validate()) == 0
        stats = game_statistics(g)
        assert stats["random_locations"] == 1

    def test_probabilistic_traffic_valid(self):
        g = make_probabilistic_traffic()
        assert len(g.validate()) == 0
        stats = game_statistics(g)
        assert stats["random_locations"] == 1

    def test_adversarial_random_valid(self):
        g = make_adversarial_random_game()
        assert len(g.validate()) == 0
        stats = game_statistics(g)
        assert stats["random_locations"] == 2

    def test_retry_game_valid(self):
        g = make_retry_game()
        assert len(g.validate()) == 0

    def test_two_player_stochastic_valid(self):
        g = make_two_player_stochastic()
        assert len(g.validate()) == 0
        stats = game_statistics(g)
        assert stats["min_locations"] >= 1
        assert stats["max_locations"] >= 1
        assert stats["random_locations"] >= 1


# ============================================================
# Exploration Tests
# ============================================================

class TestExploration:
    def test_explore_coin_flip(self):
        g = make_coin_flip_game()
        reached = explore_reachable(g)
        assert "start" in reached
        assert reached["start"]
        assert reached["flip"]
        assert reached["win"]

    def test_explore_traffic(self):
        g = make_probabilistic_traffic()
        reached = explore_reachable(g)
        assert reached["idle"]
        assert reached["sense"]

    def test_explore_unreachable(self):
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        # No edge from s0 to s1
        reached = explore_reachable(g)
        assert reached["s0"]
        assert not reached["s1"]


# ============================================================
# Positive-Probability Reachability Tests
# ============================================================

class TestPositiveProbReachability:
    def test_coin_flip_pp(self):
        g = make_coin_flip_game()
        result = solve_positive_prob_reachability(g, {"win"})
        assert "start" in result.winning_locations_pp
        assert "flip" in result.winning_locations_pp
        assert "retry" in result.winning_locations_pp

    def test_simple_reachable(self):
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        g.add_edge("s0", "s1")
        result = solve_positive_prob_reachability(g, {"s1"})
        assert "s0" in result.winning_locations_pp

    def test_unreachable(self):
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        # No edge
        result = solve_positive_prob_reachability(g, {"s1"})
        assert "s0" not in result.winning_locations_pp

    def test_random_one_path(self):
        """RANDOM with one path to target -> positive prob."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.RANDOM)
        g.add_location("win", PlayerType.MIN)
        g.add_location("lose", PlayerType.MIN)
        g.add_edge("s0", "win", probability=0.3)
        g.add_edge("s0", "lose", probability=0.7)
        result = solve_positive_prob_reachability(g, {"win"})
        assert "s0" in result.winning_locations_pp

    def test_adversarial_game_pp(self):
        g = make_adversarial_random_game()
        result = solve_positive_prob_reachability(g, {"win"})
        assert "s0" in result.winning_locations_pp

    def test_traffic_pp(self):
        g = make_probabilistic_traffic()
        result = solve_positive_prob_reachability(g, {"passed"})
        assert "idle" in result.winning_locations_pp


# ============================================================
# Almost-Sure Reachability Tests
# ============================================================

class TestAlmostSureReachability:
    def test_deterministic_reachable(self):
        """Deterministic reachable -> almost-sure."""
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        g.add_edge("s0", "s1")
        result = solve_almost_sure_reachability(g, {"s1"})
        assert "s0" in result.winning_locations_as

    def test_retry_game_as(self):
        """Infinite retries with p>0 -> almost-sure."""
        g = make_retry_game()
        result = solve_almost_sure_reachability(g, {"done"})
        assert "try" in result.winning_locations_as

    def test_random_no_escape_as(self):
        """RANDOM where ALL successors lead to target."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "r"
        g.add_location("r", PlayerType.RANDOM)
        g.add_location("t1", PlayerType.MIN)
        g.add_location("t2", PlayerType.MIN)
        g.add_edge("r", "t1", probability=0.5)
        g.add_edge("r", "t2", probability=0.5)
        result = solve_almost_sure_reachability(g, {"t1", "t2"})
        assert "r" in result.winning_locations_as

    def test_random_with_escape_not_as(self):
        """RANDOM where one successor escapes -> not almost-sure."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "r"
        g.add_location("r", PlayerType.RANDOM)
        g.add_location("win", PlayerType.MIN)
        g.add_location("sink", PlayerType.MIN)
        g.add_edge("r", "win", probability=0.5)
        g.add_edge("r", "sink", probability=0.5)
        # sink has no edges to win
        result = solve_almost_sure_reachability(g, {"win"})
        assert "r" not in result.winning_locations_as

    def test_max_blocks_as(self):
        """MAX can prevent reaching target."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "m"
        g.add_location("m", PlayerType.MAX)
        g.add_location("win", PlayerType.MIN)
        g.add_location("lose", PlayerType.MIN)
        g.add_edge("m", "win")
        g.add_edge("m", "lose")
        result = solve_almost_sure_reachability(g, {"win"})
        # MAX can always choose lose
        assert "m" not in result.winning_locations_as

    def test_coin_flip_as(self):
        """Coin flip with retry is almost-sure winning."""
        g = make_coin_flip_game()
        result = solve_almost_sure_reachability(g, {"win"})
        # With unbounded retries, eventually wins almost-surely
        assert "retry" in result.winning_locations_as


# ============================================================
# Combined Reachability Tests
# ============================================================

class TestCombinedReachability:
    def test_as_subset_of_pp(self):
        """Almost-sure winning is always subset of positive-prob."""
        g = make_adversarial_random_game()
        result = solve_stochastic_timed_reachability(g, {"win"})
        assert result.winning_locations_as.issubset(result.winning_locations_pp)

    def test_compare_as_pp(self):
        g = make_coin_flip_game()
        comp = compare_as_pp(g, {"win"})
        assert "almost_sure" in comp
        assert "positive_prob" in comp

    def test_gap_between_as_and_pp(self):
        """Game where pp > as (some locations positive-prob but not almost-sure)."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.RANDOM)
        g.add_location("win", PlayerType.MIN)
        g.add_location("trap", PlayerType.MIN)
        g.add_edge("s0", "win", probability=0.5)
        g.add_edge("s0", "trap", probability=0.5)
        # trap is absorbing (no edges to win)
        result = solve_stochastic_timed_reachability(g, {"win"})
        assert "s0" in result.winning_locations_pp
        assert "s0" not in result.winning_locations_as


# ============================================================
# Safety Tests
# ============================================================

class TestSafety:
    def test_simple_safe(self):
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "safe"
        g.add_location("safe", PlayerType.MIN)
        g.add_location("unsafe", PlayerType.MIN)
        g.add_location("other", PlayerType.MIN)
        g.add_edge("safe", "other")
        g.add_edge("other", "safe")
        result = solve_stochastic_timed_safety(g, {"unsafe"})
        assert "safe" in result.winning_locations_as

    def test_forced_unsafe(self):
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("unsafe", PlayerType.MIN)
        g.add_edge("s0", "unsafe")
        # Only edge goes to unsafe
        result = solve_stochastic_timed_safety(g, {"unsafe"})
        assert "s0" not in result.winning_locations_as

    def test_max_can_force_unsafe(self):
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.MAX)
        g.add_location("safe", PlayerType.MIN)
        g.add_location("unsafe", PlayerType.MIN)
        g.add_edge("s0", "safe")
        g.add_edge("s0", "unsafe")
        g.add_edge("safe", "safe")
        result = solve_stochastic_timed_safety(g, {"unsafe"})
        # MAX can choose unsafe
        assert "s0" not in result.winning_locations_as

    def test_random_safety(self):
        """RANDOM with any path to unsafe -> not safe."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "r"
        g.add_location("r", PlayerType.RANDOM)
        g.add_location("safe", PlayerType.MIN)
        g.add_location("unsafe", PlayerType.MIN)
        g.add_edge("r", "safe", probability=0.9)
        g.add_edge("r", "unsafe", probability=0.1)
        g.add_edge("safe", "safe")
        result = solve_stochastic_timed_safety(g, {"unsafe"})
        assert "r" not in result.winning_locations_as

    def test_random_all_safe(self):
        """RANDOM where all successors safe -> safe."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "r"
        g.add_location("r", PlayerType.RANDOM)
        g.add_location("s1", PlayerType.MIN)
        g.add_location("s2", PlayerType.MIN)
        g.add_location("unsafe", PlayerType.MIN)
        g.add_edge("r", "s1", probability=0.5)
        g.add_edge("r", "s2", probability=0.5)
        g.add_edge("s1", "s1")
        g.add_edge("s2", "s2")
        result = solve_stochastic_timed_safety(g, {"unsafe"})
        assert "r" in result.winning_locations_as


# ============================================================
# Expected Time Tests
# ============================================================

class TestExpectedTime:
    def test_direct_reachability(self):
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("target", PlayerType.MIN)
        g.add_edge("s0", "target")
        result = solve_expected_time(g, {"target"})
        assert result.reachable
        assert result.expected_time["s0"] == pytest.approx(1.0)
        assert result.expected_time["target"] == 0.0

    def test_two_step(self):
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        g.add_location("target", PlayerType.MIN)
        g.add_edge("s0", "s1")
        g.add_edge("s1", "target")
        result = solve_expected_time(g, {"target"})
        assert result.expected_time["s0"] == pytest.approx(2.0)

    def test_random_expected_time(self):
        """Random with p=0.5: geometric distribution, E[time] = 1/p = 2 steps from flip."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "try"
        g.add_location("try", PlayerType.MIN)
        g.add_location("flip", PlayerType.RANDOM)
        g.add_location("done", PlayerType.MIN)
        g.add_edge("try", "flip")
        g.add_edge("flip", "done", probability=0.5)
        g.add_edge("flip", "try", probability=0.5)
        result = solve_expected_time(g, {"done"})
        assert result.reachable
        # E[try] = 1 + E[flip], E[flip] = 0.5*0 + 0.5*(1+E[try])
        # E[flip] = 0.5 + 0.5 + 0.5*E[flip] -> wrong, let me think
        # E[try] = 1 + E[flip]
        # E[flip] = 1 + 0.5*0 + 0.5*E[try] = 1 + 0.5*E[try]
        # E[try] = 1 + 1 + 0.5*E[try] = 2 + 0.5*E[try]
        # 0.5*E[try] = 2 => E[try] = 4
        assert result.expected_time["try"] == pytest.approx(4.0, rel=0.1)

    def test_unreachable_expected_time(self):
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("target", PlayerType.MIN)
        # No path
        result = solve_expected_time(g, {"target"})
        assert not result.reachable

    def test_min_chooses_shorter_path(self):
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("short", PlayerType.MIN)
        g.add_location("long1", PlayerType.MIN)
        g.add_location("long2", PlayerType.MIN)
        g.add_location("target", PlayerType.MIN)
        g.add_edge("s0", "short")
        g.add_edge("s0", "long1")
        g.add_edge("short", "target")
        g.add_edge("long1", "long2")
        g.add_edge("long2", "target")
        result = solve_expected_time(g, {"target"})
        assert result.expected_time["s0"] == pytest.approx(2.0)


# ============================================================
# Qualitative Buchi Tests
# ============================================================

class TestQualitativeBuchi:
    def test_simple_buchi(self):
        """Cycle through accepting -> pp winning."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.accepting = {"s0"}
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        g.add_edge("s0", "s1")
        g.add_edge("s1", "s0")
        result = solve_qualitative_buchi(g)
        assert "s0" in result.winning_locations_pp
        assert "s1" in result.winning_locations_pp

    def test_no_cycle_to_accepting(self):
        """No cycle through accepting -> not winning."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.accepting = {"s1"}
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        g.add_edge("s0", "s1")
        # s1 is absorbing, can visit once but not infinitely
        result = solve_qualitative_buchi(g)
        # s1 has no outgoing edges, can't visit again
        assert "s0" not in result.winning_locations_pp

    def test_random_buchi(self):
        """RANDOM in cycle -- pp winning."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.accepting = {"s0"}
        g.add_location("s0", PlayerType.MIN)
        g.add_location("r", PlayerType.RANDOM)
        g.add_edge("s0", "r")
        g.add_edge("r", "s0", probability=0.5)
        g.add_edge("r", "s0", probability=0.5)  # both go back
        result = solve_qualitative_buchi(g)
        assert "s0" in result.winning_locations_pp

    def test_max_blocks_buchi(self):
        """MAX can prevent revisiting accepting."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.accepting = {"s0"}
        g.add_location("s0", PlayerType.MIN)
        g.add_location("m", PlayerType.MAX)
        g.add_location("sink", PlayerType.MIN)
        g.add_edge("s0", "m")
        g.add_edge("m", "s0")
        g.add_edge("m", "sink")
        g.add_edge("sink", "sink")
        result = solve_qualitative_buchi(g)
        # MAX can always choose sink
        assert "s0" not in result.winning_locations_pp


# ============================================================
# Simulation Tests
# ============================================================

class TestSimulation:
    def test_deterministic_sim(self):
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.initial = "s0"
        g.accepting = {"s1"}
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        idx = g.add_edge("s0", "s1")
        trace = simulate_play(g, strategy_min={"s0": (1.0, idx)})
        assert trace.reached_target
        assert len(trace.steps) == 1
        assert trace.total_time == 1.0

    def test_random_sim(self):
        g = make_coin_flip_game()
        # MIN strategy: always go/retry immediately
        strat = {"start": (0.0, 0), "retry": (0.0, 3)}
        trace = simulate_play(g, strategy_min=strat, random_seed=42,
                              max_steps=100)
        # Should eventually reach win
        assert trace.reached_target or len(trace.steps) == 100

    def test_sim_records_probability(self):
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "r"
        g.accepting = {"a"}
        g.add_location("r", PlayerType.RANDOM)
        g.add_location("a", PlayerType.MIN)
        g.add_location("b", PlayerType.MIN)
        g.add_edge("r", "a", probability=0.7)
        g.add_edge("r", "b", probability=0.3)
        trace = simulate_play(g, strategy_min={}, random_seed=1)
        assert len(trace.steps) == 1
        assert trace.steps[0].probability in (0.7, 0.3)

    def test_sim_max_steps(self):
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        g.add_edge("s0", "s1")
        g.add_edge("s1", "s0")
        trace = simulate_play(g, strategy_min={"s0": (0, 0), "s1": (0, 1)},
                              max_steps=5)
        assert len(trace.steps) == 5


# ============================================================
# Statistics Tests
# ============================================================

class TestStatistics:
    def test_coin_flip_stats(self):
        g = make_coin_flip_game()
        stats = game_statistics(g)
        assert stats["locations"] == 4
        assert stats["random_locations"] == 1
        assert stats["clocks"] == 1

    def test_adversarial_stats(self):
        g = make_adversarial_random_game()
        stats = game_statistics(g)
        assert stats["min_locations"] >= 1
        assert stats["max_locations"] >= 1
        assert stats["random_locations"] == 2

    def test_empty_stats(self):
        g = StochasticTimedGame()
        stats = game_statistics(g)
        assert stats["locations"] == 0
        assert stats["edges"] == 0


# ============================================================
# Timed Constraint Tests
# ============================================================

class TestTimedConstraints:
    def test_timed_reachability_with_deadline(self):
        """Must reach target within clock constraint."""
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN, invariant=Guard(
            constraints=(clock_le("x", 5),)))
        g.add_location("target", PlayerType.MIN)
        g.add_edge("s0", "target", guard=Guard(
            constraints=(clock_le("x", 3),)))
        result = solve_positive_prob_reachability(g, {"target"})
        assert "s0" in result.winning_locations_pp

    def test_timed_unreachable_too_late(self):
        """Guard requires x >= 10 but invariant limits x <= 5."""
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN, invariant=Guard(
            constraints=(clock_le("x", 5),)))
        g.add_location("target", PlayerType.MIN)
        g.add_edge("s0", "target", guard=Guard(
            constraints=(clock_ge("x", 10),)))
        result = solve_positive_prob_reachability(g, {"target"})
        assert "s0" not in result.winning_locations_pp

    def test_clock_reset_enables_path(self):
        """Reset enables taking a guarded edge again."""
        g = StochasticTimedGame()
        g.clocks = {"x"}
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        g.add_location("target", PlayerType.MIN)
        g.add_edge("s0", "s1", resets=frozenset({"x"}))
        g.add_edge("s1", "target", guard=Guard(
            constraints=(clock_le("x", 2),)))
        result = solve_positive_prob_reachability(g, {"target"})
        assert "s0" in result.winning_locations_pp

    def test_two_clock_game(self):
        g = StochasticTimedGame()
        g.clocks = {"x", "y"}
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN, invariant=Guard(
            constraints=(clock_le("x", 5),)))
        g.add_location("s1", PlayerType.MIN, invariant=Guard(
            constraints=(clock_le("y", 3),)))
        g.add_location("target", PlayerType.MIN)
        g.add_edge("s0", "s1", guard=Guard(
            constraints=(clock_ge("x", 1),)), resets=frozenset({"y"}))
        g.add_edge("s1", "target", guard=Guard(
            constraints=(clock_le("y", 2),)))
        result = solve_positive_prob_reachability(g, {"target"})
        assert "s0" in result.winning_locations_pp


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_coin_flip_full(self):
        """Full analysis of coin flip game."""
        g = make_coin_flip_game()
        result = solve_stochastic_timed_reachability(g, {"win"})
        # Positive-prob: yes (can win on first flip)
        assert "start" in result.winning_locations_pp
        # Almost-sure: yes (infinite retries)
        assert "start" in result.winning_locations_as

    def test_traffic_full(self):
        g = make_probabilistic_traffic()
        result = solve_stochastic_timed_reachability(g, {"passed"})
        assert "idle" in result.winning_locations_pp

    def test_retry_game_full(self):
        g = make_retry_game()
        result = solve_stochastic_timed_reachability(g, {"done"})
        assert "try" in result.winning_locations_pp
        assert "try" in result.winning_locations_as

        et = solve_expected_time(g, {"done"})
        assert et.reachable
        assert et.expected_time["try"] > 0

    def test_safety_and_reachability_dual(self):
        """Safety of complement should relate to reachability."""
        g = StochasticTimedGame()
        g.clocks = set()
        g.initial = "s0"
        g.add_location("s0", PlayerType.MIN)
        g.add_location("s1", PlayerType.MIN)
        g.add_location("target", PlayerType.MIN)
        g.add_edge("s0", "s1")
        g.add_edge("s1", "target")
        g.add_edge("s1", "s0")

        reach = solve_positive_prob_reachability(g, {"target"})
        safety = solve_stochastic_timed_safety(g, {"target"})
        # s0 can reach target, so it's not safe from target
        assert "s0" in reach.winning_locations_pp

    def test_all_example_games_solvable(self):
        """Smoke test: all example games produce valid results."""
        games = [
            (make_coin_flip_game(), {"win"}),
            (make_probabilistic_traffic(), {"passed"}),
            (make_adversarial_random_game(), {"win"}),
            (make_retry_game(), {"done"}),
            (make_two_player_stochastic(), {"goal"}),
        ]
        for g, targets in games:
            result = solve_stochastic_timed_reachability(g, targets)
            assert result.winning_locations_as.issubset(result.winning_locations_pp)

    def test_sym_state_hashable(self):
        s1 = SymState("loc", 0)
        s2 = SymState("loc", 0)
        assert s1 == s2
        assert hash(s1) == hash(s2)
        s = {s1, s2}
        assert len(s) == 1
