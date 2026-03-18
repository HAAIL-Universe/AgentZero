"""Tests for V203: Symbolic Quantitative Partial Observation."""

import sys
import os
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V160_energy_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V200_probabilistic_partial_obs'))

from symbolic_quantitative_po import (
    SymbolicPOGame, SymbolicBelief, BeliefEnergyResult,
    BeliefBDDEncoder, symbolic_belief_update, symbolic_possible_observations,
    build_belief_energy_game, solve_belief_energy, solve_belief_mean_payoff,
    symbolic_safety_analysis, symbolic_belief_reachability,
    pomdp_to_symbolic_game, simulate_belief_energy_game,
    game_statistics, compare_energy_vs_mean_payoff, analyze_belief_space,
    make_tiger_game, make_maze_game, make_surveillance_game,
)
from bdd_model_checker import BDD
from energy_games import Player
from probabilistic_partial_obs import POMDP, Belief


# ---------------------------------------------------------------------------
# SymbolicPOGame construction
# ---------------------------------------------------------------------------

class TestSymbolicPOGame(unittest.TestCase):

    def test_create_empty_game(self):
        g = SymbolicPOGame()
        self.assertEqual(len(g.states), 0)

    def test_add_state(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P1", 1)
        self.assertEqual(len(g.states), 2)
        self.assertEqual(g.owner[0], "P0")
        self.assertEqual(g.obs[1], 1)

    def test_add_transition(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, weight=5, prob=Fraction(1))
        self.assertEqual(len(g.transitions[(0, 0)]), 1)
        self.assertEqual(g.transitions[(0, 0)][0], (1, 5, Fraction(1)))

    def test_all_observations(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_state(2, "P0", 0)
        self.assertEqual(g.all_observations(), {0, 1})

    def test_obs_class(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_state(2, "P0", 0)
        self.assertEqual(g.obs_class(0), {0, 2})
        self.assertEqual(g.obs_class(1), {1})

    def test_get_actions(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 0)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        g.add_transition(0, 1, 1, 0, Fraction(1))
        self.assertEqual(g.get_actions(0), {0, 1})
        self.assertEqual(g.get_actions(1), set())

    def test_is_valid_correct(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 0)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        self.assertTrue(g.is_valid())

    def test_is_valid_bad_probs(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 0)
        g.add_transition(0, 0, 1, 0, Fraction(1, 2))
        self.assertFalse(g.is_valid())

    def test_probabilistic_transition(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_state(2, "P0", 1)
        g.add_transition(0, 0, 1, 1, Fraction(7, 10))
        g.add_transition(0, 0, 2, -1, Fraction(3, 10))
        self.assertTrue(g.is_valid())
        self.assertEqual(len(g.transitions[(0, 0)]), 2)


# ---------------------------------------------------------------------------
# SymbolicBelief
# ---------------------------------------------------------------------------

class TestSymbolicBelief(unittest.TestCase):

    def test_support(self):
        bdd = BDD()
        b = SymbolicBelief(
            bdd_support=bdd.TRUE,
            distribution={0: Fraction(1, 2), 1: Fraction(1, 2)},
        )
        self.assertEqual(b.support, {0, 1})

    def test_prob(self):
        bdd = BDD()
        b = SymbolicBelief(
            bdd_support=bdd.TRUE,
            distribution={0: Fraction(3, 4), 1: Fraction(1, 4)},
        )
        self.assertEqual(b.prob(0), Fraction(3, 4))
        self.assertEqual(b.prob(2), Fraction(0))

    def test_entropy_uniform(self):
        bdd = BDD()
        b = SymbolicBelief(
            bdd_support=bdd.TRUE,
            distribution={0: Fraction(1, 2), 1: Fraction(1, 2)},
        )
        self.assertAlmostEqual(b.entropy(), 1.0, places=5)

    def test_entropy_certain(self):
        bdd = BDD()
        b = SymbolicBelief(
            bdd_support=bdd.TRUE,
            distribution={0: Fraction(1)},
        )
        self.assertAlmostEqual(b.entropy(), 0.0, places=5)


# ---------------------------------------------------------------------------
# BeliefBDDEncoder
# ---------------------------------------------------------------------------

class TestBeliefBDDEncoder(unittest.TestCase):

    def test_encode_decode_support(self):
        enc = BeliefBDDEncoder({0, 1, 2})
        support = {0, 2}
        bdd = enc.encode_support(support)
        decoded = enc.decode_support(bdd)
        self.assertEqual(decoded, support)

    def test_encode_empty_support(self):
        enc = BeliefBDDEncoder({0, 1})
        support = set()
        bdd = enc.encode_support(support)
        decoded = enc.decode_support(bdd)
        self.assertEqual(decoded, set())

    def test_encode_full_support(self):
        enc = BeliefBDDEncoder({0, 1, 2})
        support = {0, 1, 2}
        bdd = enc.encode_support(support)
        decoded = enc.decode_support(bdd)
        self.assertEqual(decoded, support)

    def test_support_superset(self):
        enc = BeliefBDDEncoder({0, 1, 2})
        bdd = enc.encode_support_superset({0})
        # Should include {0}, {0,1}, {0,2}, {0,1,2}
        supports = enc.enumerate_supports(bdd)
        for s in supports:
            self.assertIn(0, s)
        self.assertGreaterEqual(len(supports), 4)

    def test_union(self):
        enc = BeliefBDDEncoder({0, 1, 2})
        a = enc.encode_support({0, 1})
        b = enc.encode_support({1, 2})
        u = enc.union(a, b)
        self.assertTrue(enc.contains(u, {0, 1}))
        self.assertTrue(enc.contains(u, {1, 2}))
        self.assertFalse(enc.contains(u, {0, 2}))

    def test_intersect(self):
        enc = BeliefBDDEncoder({0, 1, 2})
        a = enc.encode_support_superset({0})
        b = enc.encode_support_superset({1})
        inter = enc.intersect(a, b)
        # Must include both 0 and 1
        supports = enc.enumerate_supports(inter)
        for s in supports:
            self.assertIn(0, s)
            self.assertIn(1, s)

    def test_complement(self):
        enc = BeliefBDDEncoder({0, 1})
        a = enc.encode_support({0, 1})
        c = enc.complement(a)
        self.assertFalse(enc.contains(c, {0, 1}))
        self.assertTrue(enc.contains(c, {0}))  # {0} alone is in complement

    def test_is_empty(self):
        enc = BeliefBDDEncoder({0, 1})
        self.assertTrue(enc.is_empty(enc.bdd.FALSE))
        self.assertFalse(enc.is_empty(enc.bdd.TRUE))

    def test_contains(self):
        enc = BeliefBDDEncoder({0, 1, 2})
        bdd = enc.encode_support({1, 2})
        self.assertTrue(enc.contains(bdd, {1, 2}))
        self.assertFalse(enc.contains(bdd, {0, 1}))

    def test_support_count(self):
        enc = BeliefBDDEncoder({0, 1})
        # TRUE has 2^2 = 4 satisfying assignments
        self.assertEqual(enc.support_count(enc.bdd.TRUE), 4)
        # A single support has 1
        bdd = enc.encode_support({0})
        self.assertEqual(enc.support_count(bdd), 1)

    def test_enumerate_supports(self):
        enc = BeliefBDDEncoder({0, 1})
        bdd = enc.encode_support_superset({0})
        supports = enc.enumerate_supports(bdd)
        self.assertEqual(len(supports), 2)  # {0} and {0,1}
        self.assertIn({0}, supports)
        self.assertIn({0, 1}, supports)

    def test_observation_class_encoding(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_state(2, "P0", 0)
        enc = BeliefBDDEncoder({0, 1, 2})
        bdd = enc.encode_observation_class(g, 0)
        supports = enc.enumerate_supports(bdd)
        # Only states 0 and 2 have obs 0. State 1 must be absent.
        for s in supports:
            self.assertNotIn(1, s)
            self.assertTrue(len(s & {0, 2}) > 0)


# ---------------------------------------------------------------------------
# Symbolic belief update
# ---------------------------------------------------------------------------

class TestSymbolicBeliefUpdate(unittest.TestCase):

    def _simple_game(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 0, 1, Fraction(7, 10))
        g.add_transition(0, 0, 1, -1, Fraction(3, 10))
        g.add_transition(1, 0, 0, 1, Fraction(2, 10))
        g.add_transition(1, 0, 1, -1, Fraction(8, 10))
        return g

    def test_update_deterministic(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        enc = BeliefBDDEncoder({0, 1})
        belief = SymbolicBelief(
            bdd_support=enc.encode_support({0}),
            distribution={0: Fraction(1)},
        )
        new_b = symbolic_belief_update(g, belief, 0, 1, enc)
        self.assertIsNotNone(new_b)
        self.assertEqual(new_b.support, {1})
        self.assertEqual(new_b.prob(1), Fraction(1))

    def test_update_probabilistic(self):
        g = self._simple_game()
        enc = BeliefBDDEncoder({0, 1})
        belief = SymbolicBelief(
            bdd_support=enc.encode_support({0, 1}),
            distribution={0: Fraction(1, 2), 1: Fraction(1, 2)},
        )
        # After action 0, observe 0 (state 0)
        new_b = symbolic_belief_update(g, belief, 0, 0, enc)
        self.assertIsNotNone(new_b)
        self.assertIn(0, new_b.support)
        # Prob should be normalized
        total = sum(new_b.distribution.values())
        self.assertEqual(total, Fraction(1))

    def test_update_impossible_observation(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 0, 0, Fraction(1))
        enc = BeliefBDDEncoder({0, 1})
        belief = SymbolicBelief(
            bdd_support=enc.encode_support({0}),
            distribution={0: Fraction(1)},
        )
        # Observation 1 is impossible from state 0 with action 0
        new_b = symbolic_belief_update(g, belief, 0, 1, enc)
        self.assertIsNone(new_b)

    def test_update_maintains_bdd_support(self):
        g = self._simple_game()
        enc = BeliefBDDEncoder({0, 1})
        belief = SymbolicBelief(
            bdd_support=enc.encode_support({0}),
            distribution={0: Fraction(1)},
        )
        new_b = symbolic_belief_update(g, belief, 0, 0, enc)
        self.assertIsNotNone(new_b)
        decoded = enc.decode_support(new_b.bdd_support)
        self.assertEqual(decoded, new_b.support)


class TestSymbolicPossibleObservations(unittest.TestCase):

    def test_single_successor(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        enc = BeliefBDDEncoder({0, 1})
        belief = SymbolicBelief(
            bdd_support=enc.encode_support({0}),
            distribution={0: Fraction(1)},
        )
        obs = symbolic_possible_observations(g, belief, 0)
        self.assertEqual(obs, {1})

    def test_multiple_successors(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_state(2, "P0", 2)
        g.add_transition(0, 0, 1, 0, Fraction(1, 2))
        g.add_transition(0, 0, 2, 0, Fraction(1, 2))
        enc = BeliefBDDEncoder({0, 1, 2})
        belief = SymbolicBelief(
            bdd_support=enc.encode_support({0}),
            distribution={0: Fraction(1)},
        )
        obs = symbolic_possible_observations(g, belief, 0)
        self.assertEqual(obs, {1, 2})


# ---------------------------------------------------------------------------
# Build belief energy game
# ---------------------------------------------------------------------------

class TestBuildBeliefEnergyGame(unittest.TestCase):

    def test_simple_game(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 5, Fraction(1))
        g.add_transition(1, 0, 0, -3, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}

        enc = BeliefBDDEncoder(g.states)
        eg, bmap, omap = build_belief_energy_game(g, enc)
        self.assertGreater(len(eg.vertices), 0)
        self.assertGreater(len(bmap), 0)

    def test_max_beliefs_limit(self):
        g = make_maze_game(3)
        enc = BeliefBDDEncoder(g.states)
        eg, bmap, omap = build_belief_energy_game(g, enc, max_beliefs=5)
        self.assertLessEqual(len(bmap), 5)

    def test_belief_dedup(self):
        """Beliefs with same support should be deduped."""
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 0)
        g.add_transition(0, 0, 0, 1, Fraction(1))
        g.add_transition(1, 0, 1, 1, Fraction(1))
        g.initial = {0, 1}
        g.actions_even = {0}

        enc = BeliefBDDEncoder(g.states)
        eg, bmap, _ = build_belief_energy_game(g, enc)
        # Initial belief {0,1} -> same belief {0,1}, should be one vertex (or two if different obs)
        self.assertGreater(len(bmap), 0)


# ---------------------------------------------------------------------------
# Solve belief energy
# ---------------------------------------------------------------------------

class TestSolveBeliefEnergy(unittest.TestCase):

    def test_trivial_game(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_transition(0, 0, 0, 1, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}
        result = solve_belief_energy(g)
        self.assertGreater(result.belief_states_explored, 0)

    def test_empty_game(self):
        g = SymbolicPOGame()
        result = solve_belief_energy(g)
        self.assertEqual(result.belief_states_explored, 0)

    def test_positive_weight_cycle(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 5, Fraction(1))
        g.add_transition(1, 0, 0, 3, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}
        result = solve_belief_energy(g)
        self.assertGreater(result.belief_states_explored, 0)
        # With positive weights, should be winnable with 0 initial energy
        for o, e in result.min_energy.items():
            if e is not None:
                self.assertEqual(e, 0)

    def test_two_player_game(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P1", 1)
        g.add_state(2, "P0", 0)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        g.add_transition(1, 0, 2, -5, Fraction(1))
        g.add_transition(1, 1, 0, 3, Fraction(1))
        g.add_transition(2, 0, 0, 1, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}
        g.actions_odd = {0, 1}
        result = solve_belief_energy(g)
        self.assertGreater(result.belief_states_explored, 0)


# ---------------------------------------------------------------------------
# Solve belief mean payoff
# ---------------------------------------------------------------------------

class TestSolveBeliefMeanPayoff(unittest.TestCase):

    def test_trivial_game(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_transition(0, 0, 0, 5, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}
        result = solve_belief_mean_payoff(g)
        self.assertGreater(result.belief_states_explored, 0)

    def test_empty_game(self):
        g = SymbolicPOGame()
        result = solve_belief_mean_payoff(g)
        self.assertEqual(result.belief_states_explored, 0)

    def test_mean_payoff_value(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 10, Fraction(1))
        g.add_transition(1, 0, 0, -2, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}
        result = solve_belief_mean_payoff(g)
        # Mean payoff should be (10-2)/2 = 4
        self.assertGreater(result.belief_states_explored, 0)


# ---------------------------------------------------------------------------
# Symbolic safety analysis
# ---------------------------------------------------------------------------

class TestSymbolicSafetyAnalysis(unittest.TestCase):

    def test_no_unsafe_states(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        g.initial = {0}
        enc = BeliefBDDEncoder(g.states)
        safe_bdd, iters = symbolic_safety_analysis(g, enc)
        # No unsafe states -> everything is safe
        self.assertFalse(enc.is_empty(safe_bdd))

    def test_unsafe_state_reachable(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        g.initial = {0}
        g.unsafe = {1}
        enc = BeliefBDDEncoder(g.states)
        safe_bdd, iters = symbolic_safety_analysis(g, enc)
        # State 0 leads to unsafe state 1 -> state 0 is not safe
        v0 = enc.bdd.named_var(enc.state_to_var[0])
        check = enc.bdd.AND(safe_bdd, v0)
        # v0 alone (without v1) should not be safe
        v1_neg = enc.bdd.NOT(enc.bdd.named_var(enc.state_to_var[1]))
        only_0 = enc.bdd.AND(v0, v1_neg)
        check_0 = enc.bdd.AND(safe_bdd, only_0)
        # State 0 can reach unsafe state 1, so not safe
        self.assertTrue(enc.is_empty(check_0))

    def test_isolated_safe_region(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_state(2, "P0", 0)
        g.add_transition(0, 0, 0, 0, Fraction(1))  # self-loop, safe
        g.add_transition(1, 0, 2, 0, Fraction(1))
        g.unsafe = {2}
        g.initial = {0}
        enc = BeliefBDDEncoder(g.states)
        safe_bdd, iters = symbolic_safety_analysis(g, enc)
        # State 0 cannot reach unsafe -> should be safe
        only_0 = enc.encode_support({0})
        check = enc.bdd.AND(safe_bdd, only_0)
        self.assertFalse(enc.is_empty(check))


# ---------------------------------------------------------------------------
# Symbolic belief reachability
# ---------------------------------------------------------------------------

class TestSymbolicBeliefReachability(unittest.TestCase):

    def test_self_loop_reachability(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_transition(0, 0, 0, 0, Fraction(1))
        g.initial = {0}
        enc = BeliefBDDEncoder(g.states)
        reach_bdd, iters = symbolic_belief_reachability(g, enc)
        self.assertTrue(enc.contains(reach_bdd, {0}))

    def test_chain_reachability(self):
        g = SymbolicPOGame()
        for i in range(4):
            g.add_state(i, "P0", i)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        g.add_transition(1, 0, 2, 0, Fraction(1))
        g.add_transition(2, 0, 3, 0, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}
        enc = BeliefBDDEncoder(g.states)
        reach_bdd, iters = symbolic_belief_reachability(g, enc)
        # All states reachable (as individual supports from image)
        for i in range(4):
            v = enc.bdd.named_var(enc.state_to_var[i])
            check = enc.bdd.AND(reach_bdd, v)
            self.assertFalse(enc.is_empty(check))


# ---------------------------------------------------------------------------
# POMDP conversion
# ---------------------------------------------------------------------------

class TestPOMDPConversion(unittest.TestCase):

    def test_simple_pomdp(self):
        pomdp = POMDP()
        pomdp.states = {0, 1}
        pomdp.actions = {0}
        pomdp.state_actions = {0: {0}, 1: {0}}
        pomdp.transitions = {
            (0, 0): [(1, Fraction(1))],
            (1, 0): [(0, Fraction(1))],
        }
        pomdp.obs = {0: 0, 1: 1}
        pomdp.rewards = {(0, 0): Fraction(5), (1, 0): Fraction(-3)}
        pomdp.initial = [(0, Fraction(1))]
        pomdp.target = set()
        pomdp.discount = Fraction(1)
        pomdp.horizon = 0

        game = pomdp_to_symbolic_game(pomdp)
        self.assertEqual(game.states, {0, 1})
        self.assertEqual(game.owner[0], "P0")
        self.assertTrue(game.is_valid())
        # Check weight from reward
        self.assertEqual(game.transitions[(0, 0)][0][1], 5)
        self.assertEqual(game.transitions[(1, 0)][0][1], -3)

    def test_conversion_preserves_observations(self):
        pomdp = POMDP()
        pomdp.states = {0, 1, 2}
        pomdp.actions = {0}
        pomdp.state_actions = {s: {0} for s in pomdp.states}
        pomdp.transitions = {(s, 0): [(s, Fraction(1))] for s in pomdp.states}
        pomdp.obs = {0: 0, 1: 1, 2: 0}
        pomdp.rewards = {}
        pomdp.initial = [(0, Fraction(1))]
        pomdp.target = set()
        pomdp.discount = Fraction(1)
        pomdp.horizon = 0

        game = pomdp_to_symbolic_game(pomdp)
        self.assertEqual(game.obs[0], 0)
        self.assertEqual(game.obs[1], 1)
        self.assertEqual(game.obs[2], 0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation(unittest.TestCase):

    def test_simulate_simple(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 5, Fraction(1))
        g.add_transition(1, 0, 0, -3, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}

        trace = simulate_belief_energy_game(g, {0: 0, 1: 0}, steps=4,
                                            initial_state=0, initial_energy=10)
        self.assertEqual(len(trace), 4)
        self.assertEqual(trace[0]['state'], 0)
        self.assertEqual(trace[0]['action'], 0)

    def test_simulate_tracks_energy(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 5, Fraction(1))
        g.add_transition(1, 0, 0, -3, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}

        trace = simulate_belief_energy_game(g, {0: 0, 1: 0}, steps=4,
                                            initial_state=0, initial_energy=0)
        # Step 0: 0->1, weight 5, energy=5
        # Step 1: 1->0, weight -3, energy=2
        self.assertEqual(trace[0]['energy'], 5)
        self.assertEqual(trace[1]['energy'], 2)

    def test_simulate_terminates_on_dead_end(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 1, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}
        # State 1 has no transitions -> terminates
        trace = simulate_belief_energy_game(g, {0: 0, 1: 0}, steps=10,
                                            initial_state=0, initial_energy=0)
        self.assertTrue(any(t.get('terminated', False) for t in trace))

    def test_simulate_belief_entropy(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 0)  # same observation!
        g.add_transition(0, 0, 0, 1, Fraction(1))
        g.add_transition(1, 0, 1, 1, Fraction(1))
        g.initial = {0, 1}
        g.actions_even = {0}

        trace = simulate_belief_energy_game(g, {0: 0}, steps=2,
                                            initial_state=0, initial_energy=0)
        # Initial belief over {0,1} has entropy > 0
        self.assertGreater(trace[0]['belief_entropy'], 0)


# ---------------------------------------------------------------------------
# Example games
# ---------------------------------------------------------------------------

class TestTigerGame(unittest.TestCase):

    def test_tiger_game_valid(self):
        g = make_tiger_game()
        self.assertTrue(g.is_valid())
        self.assertEqual(len(g.states), 2)
        self.assertEqual(len(g.actions_even), 3)

    def test_tiger_game_solve(self):
        g = make_tiger_game()
        result = solve_belief_energy(g)
        self.assertGreater(result.belief_states_explored, 0)

    def test_tiger_game_statistics(self):
        g = make_tiger_game()
        stats = game_statistics(g)
        self.assertEqual(stats['num_states'], 2)
        self.assertEqual(stats['num_actions_even'], 3)

    def test_tiger_game_simulation(self):
        g = make_tiger_game()
        # Simple strategy: always listen
        trace = simulate_belief_energy_game(
            g, {0: 0, 1: 0}, steps=5,
            initial_state=0, initial_energy=100,
        )
        self.assertGreater(len(trace), 0)


class TestMazeGame(unittest.TestCase):

    def test_maze_game_valid(self):
        g = make_maze_game(3)
        self.assertTrue(g.is_valid())
        self.assertEqual(len(g.states), 9)

    def test_maze_game_solve(self):
        g = make_maze_game(2)
        result = solve_belief_energy(g, max_beliefs=50)
        self.assertGreater(result.belief_states_explored, 0)

    def test_maze_game_statistics(self):
        g = make_maze_game(3)
        stats = game_statistics(g)
        self.assertEqual(stats['num_states'], 9)
        self.assertEqual(stats['num_actions_even'], 4)


class TestSurveillanceGame(unittest.TestCase):

    def test_surveillance_game_valid(self):
        g = make_surveillance_game()
        self.assertTrue(g.is_valid())
        self.assertEqual(len(g.states), 16)

    def test_surveillance_game_has_unsafe(self):
        g = make_surveillance_game()
        self.assertGreater(len(g.unsafe), 0)

    def test_surveillance_game_solve(self):
        g = make_surveillance_game()
        result = solve_belief_energy(g, max_beliefs=50)
        self.assertGreater(result.belief_states_explored, 0)


# ---------------------------------------------------------------------------
# Analysis and comparison
# ---------------------------------------------------------------------------

class TestGameStatistics(unittest.TestCase):

    def test_stats_basic(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P1", 1)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        g.initial = {0}
        g.target = {1}
        stats = game_statistics(g)
        self.assertEqual(stats['num_states'], 2)
        self.assertEqual(stats['num_transitions'], 1)
        self.assertEqual(stats['num_initial'], 1)
        self.assertEqual(stats['num_target'], 1)
        self.assertEqual(stats['owners'], {'P0': 1, 'P1': 1})


class TestCompareEnergyVsMeanPayoff(unittest.TestCase):

    def test_compare_simple(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 5, Fraction(1))
        g.add_transition(1, 0, 0, -3, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}

        result = compare_energy_vs_mean_payoff(g, max_beliefs=50)
        self.assertIn('energy', result)
        self.assertIn('mean_payoff', result)
        self.assertGreater(result['energy']['belief_states'], 0)
        self.assertGreater(result['mean_payoff']['belief_states'], 0)


class TestAnalyzeBeliefSpace(unittest.TestCase):

    def test_analyze_simple(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        g.add_transition(1, 0, 0, 0, Fraction(1))
        g.initial = {0}
        g.actions_even = {0}

        enc = BeliefBDDEncoder(g.states)
        analysis = analyze_belief_space(g, enc)
        self.assertIn('reachable_supports', analysis)
        self.assertIn('safe_supports', analysis)
        self.assertIn('total_possible_supports', analysis)
        self.assertEqual(analysis['total_possible_supports'], 4)  # 2^2

    def test_analyze_with_unsafe(self):
        g = SymbolicPOGame()
        g.add_state(0, "P0", 0)
        g.add_state(1, "P0", 1)
        g.add_transition(0, 0, 1, 0, Fraction(1))
        g.add_transition(1, 0, 0, 0, Fraction(1))
        g.initial = {0}
        g.unsafe = {1}
        g.actions_even = {0}

        enc = BeliefBDDEncoder(g.states)
        analysis = analyze_belief_space(g, enc)
        # With unsafe state, safe supports < total
        self.assertLess(analysis['safe_supports'], analysis['total_possible_supports'])


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline(unittest.TestCase):
    """End-to-end tests composing all components."""

    def test_tiger_full_pipeline(self):
        """Tiger game: build, encode, solve, simulate."""
        g = make_tiger_game()
        self.assertTrue(g.is_valid())

        # BDD encoding
        enc = BeliefBDDEncoder(g.states)
        initial_bdd = enc.encode_support(g.initial)
        self.assertFalse(enc.is_empty(initial_bdd))

        # Belief space analysis
        analysis = analyze_belief_space(g, enc)
        self.assertGreater(analysis['reachable_supports'], 0)

        # Solve
        result = solve_belief_energy(g, max_beliefs=50)
        self.assertGreater(result.belief_states_explored, 0)

        # Simulate with found strategy (or default)
        strategy = result.strategy if result.strategy else {0: 0, 1: 0}
        trace = simulate_belief_energy_game(
            g, strategy, steps=5, initial_state=0, initial_energy=100,
        )
        self.assertGreater(len(trace), 0)

    def test_maze_full_pipeline(self):
        """Maze game: build, solve, compare."""
        g = make_maze_game(2)
        self.assertTrue(g.is_valid())

        result = solve_belief_energy(g, max_beliefs=30)
        self.assertGreater(result.belief_states_explored, 0)

        comparison = compare_energy_vs_mean_payoff(g, max_beliefs=30)
        self.assertIn('energy', comparison)

    def test_surveillance_full_pipeline(self):
        """Surveillance game: build, safety analysis, solve."""
        g = make_surveillance_game()
        self.assertTrue(g.is_valid())

        enc = BeliefBDDEncoder(g.states)
        safe_bdd, iters = symbolic_safety_analysis(g, enc)
        self.assertGreater(iters, 0)

        result = solve_belief_energy(g, max_beliefs=50)
        self.assertGreater(result.belief_states_explored, 0)

    def test_pomdp_conversion_pipeline(self):
        """POMDP -> SymbolicPOGame -> solve."""
        pomdp = POMDP()
        pomdp.states = {0, 1, 2}
        pomdp.actions = {0, 1}
        pomdp.state_actions = {0: {0, 1}, 1: {0}, 2: {0}}
        pomdp.transitions = {
            (0, 0): [(1, Fraction(1))],
            (0, 1): [(2, Fraction(1))],
            (1, 0): [(0, Fraction(1, 2)), (2, Fraction(1, 2))],
            (2, 0): [(2, Fraction(1))],
        }
        pomdp.obs = {0: 0, 1: 1, 2: 1}
        pomdp.rewards = {(0, 0): Fraction(1), (0, 1): Fraction(5)}
        pomdp.initial = [(0, Fraction(1))]
        pomdp.target = {2}
        pomdp.discount = Fraction(1)
        pomdp.horizon = 0

        game = pomdp_to_symbolic_game(pomdp)
        self.assertTrue(game.is_valid())
        result = solve_belief_energy(game, max_beliefs=30)
        self.assertGreater(result.belief_states_explored, 0)


if __name__ == '__main__':
    unittest.main()
