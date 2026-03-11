"""Tests for V165: Stochastic Parity Games."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from stochastic_parity import (
    StochasticParityGame, StochasticParityResult, VertexType,
    stochastic_attractor, solve_stochastic_parity, solve_almost_sure,
    solve_positive_prob, simulate_play, verify_strategy,
    make_game, make_simple_stochastic, make_buchi_stochastic,
    make_reachability_stochastic, make_safety_stochastic,
    compare_with_deterministic, stochastic_parity_statistics, batch_solve,
)


# ---- Section 1: Data Structure Tests ----

class TestStochasticParityGame:
    def test_add_vertex_and_edge(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_edge(0, 1)
        assert g.vertices == {0, 1}
        assert g.successors(0) == {1}
        assert g.predecessors(1) == {0}

    def test_random_vertex_probabilities(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 1)
        g.add_vertex(2, VertexType.ODD, 2)
        g.add_edge(0, 1, 0.3)
        g.add_edge(0, 2, 0.7)
        assert g.get_prob(0, 1) == 0.3
        assert g.get_prob(0, 2) == 0.7

    def test_validate_good(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 1)
        g.add_edge(0, 1, 1.0)
        assert g.validate() == []

    def test_validate_bad_sum(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 1)
        g.add_vertex(2, VertexType.EVEN, 1)
        g.add_edge(0, 1, 0.3)
        g.add_edge(0, 2, 0.3)
        errors = g.validate()
        assert len(errors) > 0

    def test_max_priority(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 3)
        g.add_vertex(1, VertexType.ODD, 5)
        g.add_vertex(2, VertexType.RANDOM, 1)
        assert g.max_priority() == 5

    def test_vertices_with_priority(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 2)
        g.add_vertex(2, VertexType.EVEN, 3)
        assert g.vertices_with_priority(2) == {0, 1}

    def test_subgame(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 0)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_vertex(2, VertexType.EVEN, 2)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        sub = g.subgame({0, 1})
        assert sub.vertices == {0, 1}
        assert sub.successors(0) == {1}
        assert sub.successors(1) == set()  # edge to 2 removed

    def test_to_parity_game(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 0)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_vertex(2, VertexType.RANDOM, 2)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0, 1.0)
        pg = g.to_parity_game()
        from parity_games import Player
        assert pg.owner[0] == Player.EVEN
        assert pg.owner[1] == Player.ODD
        assert pg.owner[2] == Player.EVEN  # RANDOM -> EVEN


# ---- Section 2: Attractor Tests ----

class TestStochasticAttractor:
    def test_even_attractor_simple(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 0)
        g.add_vertex(1, VertexType.EVEN, 0)
        g.add_vertex(2, VertexType.EVEN, 2)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        # Even can reach {2} from {0,1,2}
        attr = stochastic_attractor(g, {2}, VertexType.EVEN, g.vertices)
        assert attr == {0, 1, 2}

    def test_odd_blocks_attractor(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 0)
        g.add_vertex(1, VertexType.ODD, 0)
        g.add_vertex(2, VertexType.EVEN, 2)
        g.add_vertex(3, VertexType.EVEN, 0)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 3)
        # Odd at 1 can escape to 3, so Even can't force reaching {2} from 0
        attr = stochastic_attractor(g, {2}, VertexType.EVEN, g.vertices)
        assert 2 in attr
        assert 0 not in attr  # Odd at 1 can escape

    def test_random_almost_sure_attractor(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_vertex(2, VertexType.EVEN, 0)
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 2, 0.5)
        g.add_edge(2, 2)
        # Almost-sure: RANDOM 0 goes to {1} with p=0.5 and {2} with p=0.5
        # 2 is not in target, so 0 is NOT attracted almost-surely
        attr = stochastic_attractor(g, {1}, VertexType.EVEN, g.vertices, 'almost_sure')
        assert 0 not in attr

    def test_random_positive_prob_attractor(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_vertex(2, VertexType.EVEN, 0)
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 2, 0.5)
        g.add_edge(2, 2)
        # Positive-prob: RANDOM 0 has p=0.5 to reach {1}, so IS attracted
        attr = stochastic_attractor(g, {1}, VertexType.EVEN, g.vertices, 'positive_prob')
        assert 0 in attr

    def test_random_all_succs_in_target(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_vertex(2, VertexType.EVEN, 2)
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 2, 0.5)
        # Both successors in target: attracted in both modes
        attr_as = stochastic_attractor(g, {1, 2}, VertexType.EVEN, g.vertices, 'almost_sure')
        attr_pp = stochastic_attractor(g, {1, 2}, VertexType.EVEN, g.vertices, 'positive_prob')
        assert 0 in attr_as
        assert 0 in attr_pp


# ---- Section 3: Deterministic Parity (No Random Vertices) ----

class TestDeterministicParity:
    def test_single_vertex_even_priority(self):
        """Single Even vertex with even priority -> Even wins."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_edge(0, 0)
        r = solve_stochastic_parity(g)
        assert 0 in r.win_even_as
        assert 0 in r.win_even_pp

    def test_single_vertex_odd_priority(self):
        """Single Even vertex with odd priority -> Odd wins."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 1)
        g.add_edge(0, 0)
        r = solve_stochastic_parity(g)
        assert 0 in r.win_odd_as
        assert 0 in r.win_odd_pp

    def test_two_vertex_cycle(self):
        """0(Even,p=2) -> 1(Odd,p=1) -> 0. Max priority 2 is even -> Even wins."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        r = solve_stochastic_parity(g)
        assert r.win_even_as == {0, 1}

    def test_even_choice_wins(self):
        """Even at 0 chooses between p=2 cycle (win) and p=1 cycle (lose)."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 0)
        g.add_vertex(1, VertexType.EVEN, 2)  # even priority, good
        g.add_vertex(2, VertexType.EVEN, 1)  # odd priority, bad
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        r = solve_stochastic_parity(g)
        assert 0 in r.win_even_as
        assert r.strategy_even_as.get(0) == 1

    def test_odd_forces_loss(self):
        """Odd at 0 can force going to p=1 (bad for Even)."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.ODD, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_vertex(2, VertexType.EVEN, 1)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        r = solve_stochastic_parity(g)
        assert 0 in r.win_odd_as

    def test_partition_all_vertices(self):
        """Winning regions partition all vertices."""
        g = StochasticParityGame()
        for i in range(5):
            g.add_vertex(i, VertexType.EVEN if i % 2 == 0 else VertexType.ODD, i)
        g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(2, 3)
        g.add_edge(3, 4); g.add_edge(4, 0)
        r = solve_stochastic_parity(g)
        assert r.win_even_as | r.win_odd_as == g.vertices
        assert r.win_even_as & r.win_odd_as == set()


# ---- Section 4: Simple Stochastic Games ----

class TestSimpleStochastic:
    def test_random_to_even_priority(self):
        """RANDOM vertex goes to even-priority absorbing state with prob 1."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 1)
        r = solve_stochastic_parity(g)
        assert 0 in r.win_even_as
        assert 0 in r.win_even_pp

    def test_random_splits_good_bad(self):
        """RANDOM vertex goes to even-priority (p=0.5) or odd-priority (p=0.5).
        Almost-sure: Even loses (can't guarantee even priority forever).
        Positive-prob: Even wins (positive chance of even priority).
        """
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)  # absorbing, even prio
        g.add_vertex(2, VertexType.EVEN, 1)  # absorbing, odd prio
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 2, 0.5)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        r = solve_stochastic_parity(g)
        # Almost-sure: RANDOM can go to bad state, so Even can't guarantee
        assert 0 in r.win_odd_as
        # Positive-prob: RANDOM has p=0.5 to reach good state
        assert 0 in r.win_even_pp

    def test_random_all_good(self):
        """RANDOM vertex: all successors lead to even-priority states."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_vertex(2, VertexType.EVEN, 4)
        g.add_edge(0, 1, 0.3)
        g.add_edge(0, 2, 0.7)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        r = solve_stochastic_parity(g)
        assert 0 in r.win_even_as
        assert 0 in r.win_even_pp

    def test_random_cycle_with_good_priority(self):
        """RANDOM in a cycle: 0(R,p=0) -> 1(E,p=2) -> 0. Max prio is 2 (even) -> Even wins."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 0)
        r = solve_stochastic_parity(g)
        assert r.win_even_as == {0, 1}

    def test_even_bypasses_random(self):
        """Even at 0 can choose: go through RANDOM (risky) or directly to good state."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 0)
        g.add_vertex(1, VertexType.RANDOM, 0)
        g.add_vertex(2, VertexType.EVEN, 2)  # good absorbing
        g.add_vertex(3, VertexType.EVEN, 1)  # bad absorbing
        g.add_edge(0, 1)  # risky path
        g.add_edge(0, 2)  # safe path
        g.add_edge(1, 2, 0.5)
        g.add_edge(1, 3, 0.5)
        g.add_edge(2, 2)
        g.add_edge(3, 3)
        r = solve_stochastic_parity(g)
        # Even at 0 picks safe path to 2
        assert 0 in r.win_even_as
        assert r.strategy_even_as.get(0) == 2


# ---- Section 5: Almost-Sure vs Positive-Prob Distinction ----

class TestASvsPP:
    def test_as_subset_of_pp(self):
        """Almost-sure winning is always a subset of positive-prob winning."""
        g = make_simple_stochastic(4, random_vertex=2, prob_even=0.5)
        r = solve_stochastic_parity(g)
        assert r.win_even_as <= r.win_even_pp

    def test_strict_gap(self):
        """Game where almost-sure and positive-prob differ."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_vertex(2, VertexType.EVEN, 1)
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 2, 0.5)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        r = solve_stochastic_parity(g)
        # AS: Even loses at 0 (random can go bad)
        # PP: Even wins at 0 (random can go good)
        assert 0 not in r.win_even_as
        assert 0 in r.win_even_pp
        # Gap exists
        assert r.win_even_pp - r.win_even_as != set()

    def test_no_gap_deterministic(self):
        """When there are no RANDOM vertices, AS == PP."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        r = solve_stochastic_parity(g)
        assert r.win_even_as == r.win_even_pp
        assert r.win_odd_as == r.win_odd_pp

    def test_chain_with_random(self):
        """Chain: 0(E,p=0) -> 1(R,p=0) -> {2(E,p=2), 3(E,p=1)}"""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 0)
        g.add_vertex(1, VertexType.RANDOM, 0)
        g.add_vertex(2, VertexType.EVEN, 2)
        g.add_vertex(3, VertexType.EVEN, 1)
        g.add_edge(0, 1)
        g.add_edge(1, 2, 0.5)
        g.add_edge(1, 3, 0.5)
        g.add_edge(2, 2)
        g.add_edge(3, 3)
        r = solve_stochastic_parity(g)
        # Even is forced through random: AS=lose, PP=win
        assert 0 in r.win_odd_as
        assert 0 in r.win_even_pp


# ---- Section 6: Buchi Stochastic Games ----

class TestBuchiStochastic:
    def test_buchi_simple(self):
        """Buchi: accepting={1}, Even controls 0, RANDOM at 1."""
        g = make_buchi_stochastic(
            states=3,
            accepting={1},
            even_states={0},
            random_states={1},
            transitions=[(0, 1), (1, 0, ), (1, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        g.add_edge(2, 2)  # absorbing bad state
        r = solve_stochastic_parity(g)
        # RANDOM at 1 can go to absorbing 2 (prio 1, odd) -> AS: bad
        # But also goes back to 0 -> eventually escapes to 2 AS -> Odd wins AS
        # PP: positive chance of staying in 0-1 loop? No, eventually exits.
        # Actually with prob 1 the chain 0->1->0->1... eventually goes to 2.
        # So even PP: Odd wins at 0.
        # Wait -- Buchi requires visiting 1 infinitely often. Once at 2, never visits 1.
        # So Odd wins.
        assert 0 in r.win_odd_as

    def test_buchi_deterministic_accepting_cycle(self):
        """Even can force staying in accepting cycle."""
        g = make_buchi_stochastic(
            states=2,
            accepting={0},
            even_states={0, 1},
            random_states=set(),
            transitions=[(0, 1), (1, 0)],
        )
        r = solve_stochastic_parity(g)
        assert r.win_even_as == {0, 1}


# ---- Section 7: Reachability Stochastic Games ----

class TestReachabilityStochastic:
    def test_reachability_certain(self):
        """Even can certainly reach target."""
        g = make_reachability_stochastic(
            states=3, target={2},
            even_states={0, 1}, random_states=set(),
            transitions=[(0, 1), (1, 2)],
        )
        r = solve_stochastic_parity(g)
        assert 0 in r.win_even_as

    def test_reachability_random_blocks(self):
        """RANDOM vertex between Even and target."""
        g = make_reachability_stochastic(
            states=4, target={3},
            even_states={0}, random_states={1},
            transitions=[(0, 1), (1, 2), (1, 3)],
            probs={1: {2: 0.5, 3: 0.5}},
        )
        g.add_edge(2, 2)  # absorbing non-target
        r = solve_stochastic_parity(g)
        # AS: can't guarantee reaching target (random might go to 2)
        # PP: positive probability of reaching target
        assert 0 in r.win_odd_as
        assert 0 in r.win_even_pp


# ---- Section 8: Safety Stochastic Games ----

class TestSafetyStochastic:
    def test_safety_no_bad(self):
        """No bad states -> Even wins trivially."""
        g = make_safety_stochastic(
            states=2, bad=set(),
            even_states={0, 1}, random_states=set(),
            transitions=[(0, 1), (1, 0)],
        )
        r = solve_stochastic_parity(g)
        assert r.win_even_as == {0, 1}

    def test_safety_random_can_reach_bad(self):
        """RANDOM can go to bad state."""
        g = make_safety_stochastic(
            states=3, bad={2},
            even_states={0}, random_states={1},
            transitions=[(0, 1), (1, 0), (1, 2)],
            probs={1: {0: 0.5, 2: 0.5}},
        )
        g.add_edge(2, 2)  # absorbing bad
        r = solve_stochastic_parity(g)
        # In a cycle 0->1->0->1..., random eventually reaches 2 a.s.
        # AS: Even loses
        assert 0 in r.win_odd_as


# ---- Section 9: Simulation ----

class TestSimulation:
    def test_simulate_deterministic(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        trace = simulate_play(g, 0, {0: 1, 1: 0}, {1: 0}, steps=6)
        assert len(trace) == 6
        assert trace[0] == (0, 2)
        assert trace[1] == (1, 1)

    def test_simulate_random(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_vertex(2, VertexType.EVEN, 1)
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 2, 0.5)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        trace = simulate_play(g, 0, {}, {}, steps=5)
        assert len(trace) == 5
        # After first step, should be in either 1 or 2 forever
        assert trace[0][0] == 0
        assert trace[1][0] in {1, 2}


# ---- Section 10: Strategy Verification ----

class TestStrategyVerification:
    def test_verify_valid_strategy(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        result = verify_strategy(g, {0: 1}, {0, 1}, mode='almost_sure')
        assert result['valid']

    def test_verify_strategy_escapes_win_region(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_vertex(2, VertexType.EVEN, 1)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 0)
        g.add_edge(2, 2)
        result = verify_strategy(g, {0: 2}, {0, 1}, mode='almost_sure')
        assert not result['valid']  # Strategy goes to 2 which is outside win region


# ---- Section 11: Game Constructors ----

class TestGameConstructors:
    def test_make_game(self):
        g = make_game(
            vertices=[(0, 'even', 2), (1, 'odd', 1), (2, 'random', 0)],
            edges=[(0, 1), (1, 2), (2, 0)],
            probs={2: {0: 1.0}},
        )
        assert len(g.vertices) == 3
        assert g.vertex_type[2] == VertexType.RANDOM
        assert g.get_prob(2, 0) == 1.0

    def test_make_simple_stochastic(self):
        g = make_simple_stochastic(4, random_vertex=2, prob_even=0.7)
        assert len(g.vertices) == 4
        assert g.vertex_type[2] == VertexType.RANDOM
        # Random vertex has self-loop and forward edge
        assert 3 in g.successors(2)
        assert 2 in g.successors(2)

    def test_make_buchi_stochastic(self):
        g = make_buchi_stochastic(
            states=3, accepting={1},
            even_states={0}, random_states={2},
            transitions=[(0, 1), (1, 2), (2, 0)],
            probs={2: {0: 1.0}},
        )
        assert g.priority[1] == 2  # accepting
        assert g.priority[0] == 1  # non-accepting
        assert g.vertex_type[2] == VertexType.RANDOM

    def test_make_reachability_stochastic(self):
        g = make_reachability_stochastic(
            states=3, target={2},
            even_states={0}, random_states=set(),
            transitions=[(0, 1), (1, 2)],
        )
        assert g.priority[2] == 2  # target
        assert 2 in g.successors(2)  # self-loop on target

    def test_make_safety_stochastic(self):
        g = make_safety_stochastic(
            states=3, bad={2},
            even_states={0, 1}, random_states=set(),
            transitions=[(0, 1), (1, 2), (2, 2)],
        )
        assert g.priority[2] == 1  # bad
        assert g.priority[0] == 0  # safe


# ---- Section 12: Comparison with Deterministic ----

class TestComparison:
    def test_compare_no_random(self):
        """Without RANDOM vertices, all three should agree."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_vertex(1, VertexType.ODD, 1)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        comp = compare_with_deterministic(g)
        assert comp['deterministic']['win_even'] == comp['stochastic_as']['win_even']
        assert comp['deterministic']['win_even'] == comp['stochastic_pp']['win_even']
        assert comp['as_subset_pp']

    def test_compare_with_random(self):
        """With RANDOM: det >= pp >= as for Even."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_vertex(2, VertexType.EVEN, 1)
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 2, 0.5)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        comp = compare_with_deterministic(g)
        assert comp['as_subset_pp']
        # Deterministic (RANDOM=EVEN): Even picks 1 at vertex 0, wins {0, 1}
        # Vertex 2 has self-loop with odd priority -> Odd wins 2
        assert comp['deterministic']['win_even'] == {0, 1}
        # PP matches deterministic (RANDOM=EVEN is same semantics)
        assert comp['stochastic_pp']['win_even'] == {0, 1}
        # AS: RANDOM can go to 2 (bad), so Even loses at 0
        assert 0 not in comp['stochastic_as']['win_even']
        assert 1 in comp['stochastic_as']['win_even']

    def test_pp_matches_deterministic_for_random_as_even(self):
        """Positive-prob treats RANDOM like EVEN, should match deterministic."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)
        g.add_vertex(2, VertexType.EVEN, 1)
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 2, 0.5)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        comp = compare_with_deterministic(g)
        assert comp['pp_matches_det']


# ---- Section 13: Statistics ----

class TestStatistics:
    def test_statistics(self):
        g = make_simple_stochastic(5, random_vertex=3, prob_even=0.5)
        stats = stochastic_parity_statistics(g)
        assert stats['vertices'] == 5
        assert stats['random_vertices'] == 1
        assert stats['even_vertices'] + stats['odd_vertices'] + stats['random_vertices'] == 5

    def test_statistics_empty(self):
        g = StochasticParityGame()
        stats = stochastic_parity_statistics(g)
        assert stats['vertices'] == 0


# ---- Section 14: Batch Solve ----

class TestBatchSolve:
    def test_batch_solve(self):
        g1 = StochasticParityGame()
        g1.add_vertex(0, VertexType.EVEN, 2)
        g1.add_edge(0, 0)

        g2 = StochasticParityGame()
        g2.add_vertex(0, VertexType.ODD, 1)
        g2.add_edge(0, 0)

        results = batch_solve([('g1', g1), ('g2', g2)])
        assert 0 in results['g1'].win_even_as
        assert 0 in results['g2'].win_odd_as


# ---- Section 15: Complex Multi-Random Games ----

class TestComplexGames:
    def test_two_random_chain(self):
        """Two RANDOM vertices in sequence."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.RANDOM, 0)
        g.add_vertex(2, VertexType.EVEN, 2)  # good absorbing
        g.add_vertex(3, VertexType.EVEN, 1)  # bad absorbing
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 3, 0.5)
        g.add_edge(1, 2, 0.5)
        g.add_edge(1, 3, 0.5)
        g.add_edge(2, 2)
        g.add_edge(3, 3)
        r = solve_stochastic_parity(g)
        # AS: both randoms can go bad -> Even loses
        assert 0 in r.win_odd_as
        # PP: chain of positive prob -> Even can win
        assert 0 in r.win_even_pp

    def test_mixed_ownership_cycle(self):
        """Cycle: Even -> Random -> Odd -> Even with mixed priorities."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 4)   # high even prio
        g.add_vertex(1, VertexType.RANDOM, 0)
        g.add_vertex(2, VertexType.ODD, 3)    # high odd prio
        g.add_edge(0, 1)
        g.add_edge(1, 2, 1.0)
        g.add_edge(2, 0)
        r = solve_stochastic_parity(g)
        # Cycle hits priority 4 (even) and 3 (odd). Max is 4 -> Even wins.
        assert r.win_even_as == {0, 1, 2}

    def test_diamond_random(self):
        """Diamond: 0(E) -> {1(R), 2(R)} -> {3(good), 4(bad)}."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 0)
        g.add_vertex(1, VertexType.RANDOM, 0)
        g.add_vertex(2, VertexType.RANDOM, 0)
        g.add_vertex(3, VertexType.EVEN, 2)
        g.add_vertex(4, VertexType.EVEN, 1)
        # R1 -> good(p=0.9) or bad(p=0.1)
        g.add_edge(0, 1); g.add_edge(0, 2)
        g.add_edge(1, 3, 0.9); g.add_edge(1, 4, 0.1)
        # R2 -> good(p=1.0)
        g.add_edge(2, 3, 1.0)
        g.add_edge(3, 3); g.add_edge(4, 4)
        r = solve_stochastic_parity(g)
        # AS: Even at 0 picks R2 (certain good) -> wins
        assert 0 in r.win_even_as
        assert r.strategy_even_as.get(0) == 2

    def test_large_game_partition(self):
        """Larger game: verify partition property."""
        g = StochasticParityGame()
        n = 10
        for i in range(n):
            if i == 5:
                vtype = VertexType.RANDOM
            elif i % 2 == 0:
                vtype = VertexType.EVEN
            else:
                vtype = VertexType.ODD
            g.add_vertex(i, vtype, i % 4)

        for i in range(n):
            g.add_edge(i, (i + 1) % n)
            if i == 5:
                g.probabilities[5] = {6: 1.0}

        r = solve_stochastic_parity(g)
        assert r.win_even_as | r.win_odd_as == g.vertices
        assert r.win_even_as & r.win_odd_as == set()
        assert r.win_even_pp | r.win_odd_pp == g.vertices

    def test_nested_random_choices(self):
        """RANDOM -> EVEN choice -> RANDOM pattern."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 0)
        g.add_vertex(2, VertexType.RANDOM, 0)
        g.add_vertex(3, VertexType.EVEN, 2)  # good
        g.add_vertex(4, VertexType.EVEN, 1)  # bad
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2)
        g.add_edge(1, 3)  # Even can bypass R2
        g.add_edge(2, 3, 0.5)
        g.add_edge(2, 4, 0.5)
        g.add_edge(3, 3)
        g.add_edge(4, 4)
        r = solve_stochastic_parity(g)
        # Even at 1 chooses 3 (safe), so 0 wins AS
        assert 0 in r.win_even_as


# ---- Section 16: Edge Cases ----

class TestEdgeCases:
    def test_empty_game(self):
        g = StochasticParityGame()
        r = solve_stochastic_parity(g)
        assert r.win_even_as == set()
        assert r.win_odd_as == set()

    def test_single_random_self_loop(self):
        """RANDOM vertex with self-loop only."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 2)
        g.add_edge(0, 0, 1.0)
        r = solve_stochastic_parity(g)
        # Priority 2 (even), self-loop -> Even wins
        assert 0 in r.win_even_as

    def test_dead_end_even(self):
        """Even vertex with no successors -> Even loses (dead end)."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        # No edges
        r = solve_stochastic_parity(g)
        assert 0 in r.win_odd_as

    def test_dead_end_odd(self):
        """Odd vertex with no successors -> Odd loses (dead end)."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.ODD, 1)
        r = solve_stochastic_parity(g)
        assert 0 in r.win_even_as

    def test_zero_prob_edge_ignored(self):
        """Zero-probability edges should be ignored."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.RANDOM, 0)
        g.add_vertex(1, VertexType.EVEN, 2)  # good
        g.add_vertex(2, VertexType.EVEN, 1)  # bad
        g.add_edge(0, 1, 1.0)
        g.add_edge(0, 2, 0.0)  # zero prob
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        r = solve_stochastic_parity(g)
        # Zero prob edge to bad state is ignored -> Even wins AS
        assert 0 in r.win_even_as

    def test_priority_zero(self):
        """Priority 0 is even -> good for Even."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 0)
        g.add_edge(0, 0)
        r = solve_stochastic_parity(g)
        assert 0 in r.win_even_as

    def test_high_priority_dominates(self):
        """Higher priority dominates lower in the parity condition."""
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 1)  # odd prio
        g.add_vertex(1, VertexType.EVEN, 4)  # even prio, higher
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        r = solve_stochastic_parity(g)
        # Max infinite-often priority is 4 (even) -> Even wins
        assert r.win_even_as == {0, 1}


# ---- Section 17: solve_almost_sure / solve_positive_prob API ----

class TestSolverAPIs:
    def test_solve_almost_sure_only(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_edge(0, 0)
        r = solve_almost_sure(g)
        assert 0 in r.win_even_as
        # PP fields should be empty
        assert r.win_even_pp == set()

    def test_solve_positive_prob_only(self):
        g = StochasticParityGame()
        g.add_vertex(0, VertexType.EVEN, 2)
        g.add_edge(0, 0)
        r = solve_positive_prob(g)
        assert 0 in r.win_even_pp
        # AS fields should be empty
        assert r.win_even_as == set()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
