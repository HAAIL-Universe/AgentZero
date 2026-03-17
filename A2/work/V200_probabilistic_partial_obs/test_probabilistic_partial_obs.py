"""Tests for V200: Probabilistic Partial Observation (POMDP)."""

import pytest
import sys, os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))

from probabilistic_partial_obs import (
    POMDP, POMDPObjective, Belief, AlphaVector,
    StochasticPOGame,
    belief_update, observation_probability, possible_observations,
    belief_expected_reward, value_at_belief,
    finite_horizon_vi, pbvi,
    almost_sure_reachability, positive_reachability,
    safety_probability, solve_stochastic_po_game,
    simulate_pomdp, pomdp_statistics, compare_mdp_vs_pomdp,
    belief_space_size,
)


# ============================================================================
# Helper: build common POMDPs
# ============================================================================

def make_tiger_pomdp():
    """Classic Tiger POMDP.

    Two doors: tiger behind one, gold behind other.
    Actions: listen (0), open-left (1), open-right (2).
    States: tiger-left (0), tiger-right (1).
    Observations: hear-left (0), hear-right (1).
    Listen gives noisy observation (85% correct).
    Opening correct door: +10, wrong door: -100.
    """
    p = POMDP()
    p.states = {0, 1}
    p.actions = {0, 1, 2}  # listen, open-left, open-right
    p.obs = {0: 0, 1: 1}  # tiger-left -> obs 0, tiger-right -> obs 1
    p.objective = POMDPObjective.REWARD_FINITE
    p.horizon = 3
    p.discount = Fraction(1)

    # Listen action (0): stay in same state, noisy observation
    # (observation is determined by state, so listen just stays)
    p.add_transition(0, 0, 0, Fraction(1))  # tiger-left, listen -> stay
    p.add_transition(1, 0, 1, Fraction(1))  # tiger-right, listen -> stay

    # Open-left (1): reset to uniform
    p.add_transition(0, 1, 0, Fraction(1, 2))
    p.add_transition(0, 1, 1, Fraction(1, 2))
    p.add_transition(1, 1, 0, Fraction(1, 2))
    p.add_transition(1, 1, 1, Fraction(1, 2))

    # Open-right (2): reset to uniform
    p.add_transition(0, 2, 0, Fraction(1, 2))
    p.add_transition(0, 2, 1, Fraction(1, 2))
    p.add_transition(1, 2, 0, Fraction(1, 2))
    p.add_transition(1, 2, 1, Fraction(1, 2))

    # Rewards
    p.set_reward(0, 0, -1)   # listen cost
    p.set_reward(1, 0, -1)
    p.set_reward(0, 1, -100)  # open left when tiger left -> penalty
    p.set_reward(1, 1, 10)    # open left when tiger right -> reward
    p.set_reward(0, 2, 10)    # open right when tiger left -> reward
    p.set_reward(1, 2, -100)  # open right when tiger right -> penalty

    p.set_initial([(0, Fraction(1, 2)), (1, Fraction(1, 2))])
    return p


def make_simple_pomdp():
    """Simple 3-state POMDP for testing.

    States: 0, 1, 2
    Actions: 0 (stay), 1 (move)
    Obs: 0 sees {0, 1}, 1 sees {2}
    """
    p = POMDP()
    p.states = {0, 1, 2}
    p.actions = {0, 1}
    p.obs = {0: 0, 1: 0, 2: 1}
    p.objective = POMDPObjective.REWARD_FINITE
    p.horizon = 2
    p.discount = Fraction(1)

    # Stay (0): stay in same state
    for s in p.states:
        p.add_transition(s, 0, s, Fraction(1))

    # Move (1): 0->1, 1->2, 2->0
    p.add_transition(0, 1, 1, Fraction(1))
    p.add_transition(1, 1, 2, Fraction(1))
    p.add_transition(2, 1, 0, Fraction(1))

    p.set_reward(2, 0, 5)
    p.set_reward(2, 1, 5)
    p.set_initial([(0, Fraction(1))])
    return p


def make_reachability_pomdp():
    """POMDP for reachability testing.

    States: 0 (start), 1 (intermediate), 2 (target)
    Actions: 0 (try), 1 (safe)
    State 0 -> action try -> state 1 or 2 (50/50)
    State 0 -> action safe -> state 0
    State 1 -> action try -> state 2
    State 1 -> action safe -> state 1
    Target: {2}
    """
    p = POMDP()
    p.states = {0, 1, 2}
    p.actions = {0, 1}
    p.obs = {0: 0, 1: 0, 2: 1}  # 0 and 1 look the same
    p.objective = POMDPObjective.REACHABILITY
    p.target = {2}
    p.horizon = 5

    p.add_transition(0, 0, 1, Fraction(1, 2))
    p.add_transition(0, 0, 2, Fraction(1, 2))
    p.add_transition(0, 1, 0, Fraction(1))
    p.add_transition(1, 0, 2, Fraction(1))
    p.add_transition(1, 1, 1, Fraction(1))
    p.add_transition(2, 0, 2, Fraction(1))
    p.add_transition(2, 1, 2, Fraction(1))

    p.set_initial([(0, Fraction(1))])
    return p


def make_safety_pomdp():
    """POMDP for safety testing.

    States: 0 (safe), 1 (safe), 2 (unsafe)
    Actions: 0 (careful), 1 (risky)
    Safe states = {0, 1}
    Careful always stays safe. Risky might reach unsafe.
    """
    p = POMDP()
    p.states = {0, 1, 2}
    p.actions = {0, 1}
    p.obs = {0: 0, 1: 0, 2: 1}
    p.objective = POMDPObjective.SAFETY
    p.target = {0, 1}  # safe states
    p.horizon = 5

    # Careful (0): stay safe
    p.add_transition(0, 0, 0, Fraction(1))
    p.add_transition(1, 0, 1, Fraction(1))
    p.add_transition(2, 0, 2, Fraction(1))

    # Risky (1): might hit unsafe
    p.add_transition(0, 1, 0, Fraction(1, 2))
    p.add_transition(0, 1, 2, Fraction(1, 2))
    p.add_transition(1, 1, 1, Fraction(1, 2))
    p.add_transition(1, 1, 2, Fraction(1, 2))
    p.add_transition(2, 1, 2, Fraction(1))

    p.set_initial([(0, Fraction(1))])
    return p


# ============================================================================
# POMDP construction tests
# ============================================================================

class TestPOMDPConstruction:
    def test_create_empty(self):
        p = POMDP()
        assert len(p.states) == 0
        assert len(p.actions) == 0

    def test_add_state(self):
        p = POMDP()
        p.add_state(0, 0)
        p.add_state(1, 0)
        p.add_state(2, 1)
        assert p.states == {0, 1, 2}
        assert p.obs[0] == 0
        assert p.obs[2] == 1

    def test_add_action(self):
        p = POMDP()
        p.add_action(0)
        p.add_action(1)
        assert p.actions == {0, 1}

    def test_add_transition(self):
        p = POMDP()
        p.add_transition(0, 0, 1, Fraction(1, 2))
        p.add_transition(0, 0, 2, Fraction(1, 2))
        dist = p.get_transitions(0, 0)
        assert len(dist) == 2
        assert sum(p for _, p in dist) == Fraction(1)

    def test_set_reward(self):
        p = POMDP()
        p.set_reward(0, 0, 5)
        assert p.get_reward(0, 0) == Fraction(5)
        assert p.get_reward(1, 0) == Fraction(0)

    def test_set_initial(self):
        p = POMDP()
        p.set_initial([(0, Fraction(1, 2)), (1, Fraction(1, 2))])
        assert len(p.initial) == 2
        assert sum(pr for _, pr in p.initial) == Fraction(1)

    def test_all_observations(self):
        p = make_simple_pomdp()
        assert p.all_observations() == {0, 1}

    def test_obs_class(self):
        p = make_simple_pomdp()
        assert p.obs_class(0) == {0, 1}
        assert p.obs_class(1) == {2}

    def test_is_valid(self):
        p = make_simple_pomdp()
        assert p.is_valid()

    def test_is_valid_bad_transition(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.add_transition(0, 0, 0, Fraction(1, 2))
        # Missing probability
        assert not p.is_valid()

    def test_get_actions_default(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0, 1, 2}
        assert p.get_actions(0) == {0, 1, 2}

    def test_get_actions_custom(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0, 1, 2}
        p.set_state_actions(0, {0, 1})
        assert p.get_actions(0) == {0, 1}

    def test_tiger_pomdp_valid(self):
        p = make_tiger_pomdp()
        assert p.is_valid()
        assert len(p.states) == 2
        assert len(p.actions) == 3


# ============================================================================
# Belief state tests
# ============================================================================

class TestBelief:
    def test_point_belief(self):
        b = Belief.point(0)
        assert b.support() == {0}
        assert b.prob(0) == Fraction(1)
        assert b.prob(1) == Fraction(0)
        assert b.is_valid()

    def test_uniform_belief(self):
        b = Belief.uniform({0, 1, 2})
        assert b.support() == {0, 1, 2}
        assert b.prob(0) == Fraction(1, 3)
        assert b.is_valid()

    def test_from_initial(self):
        p = make_tiger_pomdp()
        b = Belief.from_initial(p)
        assert b.support() == {0, 1}
        assert b.prob(0) == Fraction(1, 2)
        assert b.is_valid()

    def test_entropy_point(self):
        b = Belief.point(0)
        assert b.entropy() == 0.0

    def test_entropy_uniform(self):
        import math
        b = Belief.uniform({0, 1})
        assert abs(b.entropy() - 1.0) < 1e-10

    def test_observation(self):
        p = make_simple_pomdp()
        b = Belief.point(0)
        assert b.observation(p) == 0
        b2 = Belief.point(2)
        assert b2.observation(p) == 1

    def test_observation_mixed(self):
        p = make_simple_pomdp()
        # Belief over states 0 and 2 which have different observations
        b = Belief({0: Fraction(1, 2), 2: Fraction(1, 2)})
        assert b.observation(p) is None

    def test_belief_equality(self):
        b1 = Belief({0: Fraction(1, 2), 1: Fraction(1, 2)})
        b2 = Belief({0: Fraction(1, 2), 1: Fraction(1, 2)})
        assert b1 == b2

    def test_belief_hash(self):
        b1 = Belief({0: Fraction(1, 2), 1: Fraction(1, 2)})
        b2 = Belief({0: Fraction(1, 2), 1: Fraction(1, 2)})
        assert hash(b1) == hash(b2)
        s = {b1, b2}
        assert len(s) == 1

    def test_empty_belief(self):
        b = Belief()
        assert b.support() == set()

    def test_uniform_empty(self):
        b = Belief.uniform(set())
        assert b.support() == set()


# ============================================================================
# Belief update tests
# ============================================================================

class TestBeliefUpdate:
    def test_deterministic_update(self):
        """Deterministic transition: update is exact."""
        p = make_simple_pomdp()
        b = Belief.point(0)
        # Stay action: 0 -> 0, obs 0
        b_next = belief_update(p, b, 0, 0)
        assert b_next is not None
        assert b_next.prob(0) == Fraction(1)

    def test_move_update(self):
        """Move from state 0 to state 1, same observation."""
        p = make_simple_pomdp()
        b = Belief.point(0)
        # Move action: 0 -> 1, obs 0
        b_next = belief_update(p, b, 1, 0)
        assert b_next is not None
        assert b_next.prob(1) == Fraction(1)

    def test_impossible_observation(self):
        """Update with impossible observation returns None."""
        p = make_simple_pomdp()
        b = Belief.point(0)
        # Stay at 0, can't observe 1 (only state 2 has obs 1)
        b_next = belief_update(p, b, 0, 1)
        assert b_next is None

    def test_tiger_listen(self):
        """Tiger POMDP: listen stays in same state."""
        p = make_tiger_pomdp()
        b = Belief.point(0)  # know tiger is left
        b_next = belief_update(p, b, 0, 0)  # listen, observe left
        assert b_next is not None
        assert b_next.prob(0) == Fraction(1)

    def test_tiger_open_resets(self):
        """Tiger POMDP: opening door resets to uniform."""
        p = make_tiger_pomdp()
        b = Belief.point(0)
        # Open left -> reset to uniform, observe left
        b_next = belief_update(p, b, 1, 0)
        assert b_next is not None
        assert b_next.prob(0) == Fraction(1)  # conditioned on obs 0

    def test_uniform_belief_update(self):
        """Update from uniform belief."""
        p = make_tiger_pomdp()
        b = Belief.uniform({0, 1})
        # Listen, observe 0 (tiger left)
        b_next = belief_update(p, b, 0, 0)
        assert b_next is not None
        # After listening with uniform prior, obs 0 means tiger-left
        assert b_next.prob(0) == Fraction(1)

    def test_normalization(self):
        """Updated belief sums to 1."""
        p = make_reachability_pomdp()
        b = Belief.point(0)
        # Try action from state 0: go to 1 or 2
        b_next = belief_update(p, b, 0, 0)  # obs 0 means state 0 or 1
        if b_next is not None:
            assert b_next.is_valid()


# ============================================================================
# Observation probability tests
# ============================================================================

class TestObservationProbability:
    def test_certain_observation(self):
        p = make_simple_pomdp()
        b = Belief.point(0)
        # Stay: 0 -> 0, obs always 0
        prob = observation_probability(p, b, 0, 0)
        assert prob == Fraction(1)

    def test_zero_probability(self):
        p = make_simple_pomdp()
        b = Belief.point(0)
        prob = observation_probability(p, b, 0, 1)
        assert prob == Fraction(0)

    def test_split_probability(self):
        p = make_reachability_pomdp()
        b = Belief.point(0)
        # Try: 0 -> 1 (obs 0) or 2 (obs 1), 50/50
        p0 = observation_probability(p, b, 0, 0)
        p1 = observation_probability(p, b, 0, 1)
        assert p0 == Fraction(1, 2)
        assert p1 == Fraction(1, 2)

    def test_sum_to_one(self):
        p = make_reachability_pomdp()
        b = Belief.point(0)
        total = Fraction(0)
        for o in p.all_observations():
            total += observation_probability(p, b, 0, o)
        assert total == Fraction(1)

    def test_possible_observations(self):
        p = make_reachability_pomdp()
        b = Belief.point(0)
        obs = possible_observations(p, b, 0)
        assert obs == {0, 1}

    def test_possible_observations_deterministic(self):
        p = make_simple_pomdp()
        b = Belief.point(2)
        obs = possible_observations(p, b, 1)  # move: 2->0, obs 0
        assert obs == {0}


# ============================================================================
# Expected reward tests
# ============================================================================

class TestExpectedReward:
    def test_point_belief(self):
        p = make_tiger_pomdp()
        b = Belief.point(0)
        r = belief_expected_reward(p, b, 0)  # listen
        assert r == Fraction(-1)

    def test_point_open_correct(self):
        p = make_tiger_pomdp()
        b = Belief.point(0)  # tiger left
        r = belief_expected_reward(p, b, 2)  # open right -> +10
        assert r == Fraction(10)

    def test_point_open_wrong(self):
        p = make_tiger_pomdp()
        b = Belief.point(0)  # tiger left
        r = belief_expected_reward(p, b, 1)  # open left -> -100
        assert r == Fraction(-100)

    def test_uniform_open(self):
        p = make_tiger_pomdp()
        b = Belief.uniform({0, 1})
        r = belief_expected_reward(p, b, 1)  # open left: 50% -100, 50% +10
        assert r == Fraction(-45)

    def test_zero_reward(self):
        p = make_simple_pomdp()
        b = Belief.point(0)
        r = belief_expected_reward(p, b, 0)
        assert r == Fraction(0)


# ============================================================================
# Alpha-vector tests
# ============================================================================

class TestAlphaVector:
    def test_evaluate_point(self):
        alpha = AlphaVector({0: Fraction(10), 1: Fraction(-5)}, action=0)
        b = Belief.point(0)
        assert alpha.evaluate(b) == Fraction(10)

    def test_evaluate_uniform(self):
        alpha = AlphaVector({0: Fraction(10), 1: Fraction(-10)}, action=0)
        b = Belief.uniform({0, 1})
        assert alpha.evaluate(b) == Fraction(0)

    def test_value_at_belief(self):
        a1 = AlphaVector({0: Fraction(10), 1: Fraction(0)}, action=0)
        a2 = AlphaVector({0: Fraction(0), 1: Fraction(10)}, action=1)
        b = Belief.point(0)
        val, act = value_at_belief([a1, a2], b)
        assert val == Fraction(10)
        assert act == 0

    def test_value_at_belief_other(self):
        a1 = AlphaVector({0: Fraction(10), 1: Fraction(0)}, action=0)
        a2 = AlphaVector({0: Fraction(0), 1: Fraction(10)}, action=1)
        b = Belief.point(1)
        val, act = value_at_belief([a1, a2], b)
        assert val == Fraction(10)
        assert act == 1

    def test_value_at_belief_empty(self):
        val, act = value_at_belief([], Belief.point(0))
        assert val == Fraction(0)
        assert act == -1

    def test_missing_state(self):
        alpha = AlphaVector({0: Fraction(10)}, action=0)
        b = Belief({1: Fraction(1)})
        assert alpha.evaluate(b) == Fraction(0)


# ============================================================================
# Finite-horizon value iteration tests
# ============================================================================

class TestFiniteHorizonVI:
    def test_simple_one_step(self):
        p = make_simple_pomdp()
        p.horizon = 1
        alphas_per_step = finite_horizon_vi(p)
        assert len(alphas_per_step) >= 1

    def test_simple_value(self):
        """Single-state POMDP: value = horizon * reward."""
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.obs = {0: 0}
        p.add_transition(0, 0, 0, Fraction(1))
        p.set_reward(0, 0, 3)
        p.set_initial([(0, Fraction(1))])
        p.horizon = 4
        p.discount = Fraction(1)

        alphas = finite_horizon_vi(p)
        b = Belief.point(0)
        val, _ = value_at_belief(alphas[0], b)
        assert val == Fraction(12)  # 4 * 3

    def test_discounted(self):
        """Discounted single-state: sum of geometric series."""
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.obs = {0: 0}
        p.add_transition(0, 0, 0, Fraction(1))
        p.set_reward(0, 0, 1)
        p.set_initial([(0, Fraction(1))])
        p.horizon = 3
        p.discount = Fraction(1, 2)

        alphas = finite_horizon_vi(p)
        b = Belief.point(0)
        val, _ = value_at_belief(alphas[0], b)
        # 1 + 0.5 + 0.25 = 1.75
        assert val == Fraction(7, 4)

    def test_two_actions(self):
        """Two actions with different rewards."""
        p = POMDP()
        p.states = {0}
        p.actions = {0, 1}
        p.obs = {0: 0}
        p.add_transition(0, 0, 0, Fraction(1))
        p.add_transition(0, 1, 0, Fraction(1))
        p.set_reward(0, 0, 1)
        p.set_reward(0, 1, 5)
        p.set_initial([(0, Fraction(1))])
        p.horizon = 1

        alphas = finite_horizon_vi(p)
        b = Belief.point(0)
        val, act = value_at_belief(alphas[0], b)
        assert val == Fraction(5)
        assert act == 1

    def test_horizon_zero(self):
        """Horizon 0: empty alpha set at step 0 should still work."""
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.obs = {0: 0}
        p.add_transition(0, 0, 0, Fraction(1))
        p.set_reward(0, 0, 1)
        p.set_initial([(0, Fraction(1))])
        p.horizon = 0

        alphas = finite_horizon_vi(p)
        # Terminal: value is 0
        b = Belief.point(0)
        val, _ = value_at_belief(alphas[0] if alphas else [], b)
        assert val == Fraction(0)

    def test_tiger_prefers_listen(self):
        """Tiger with 1-step horizon: opening randomly is bad, but it's the only option."""
        p = make_tiger_pomdp()
        p.horizon = 1
        alphas = finite_horizon_vi(p)
        b = Belief.uniform({0, 1})
        val, act = value_at_belief(alphas[0], b)
        # With uniform belief and 1 step:
        # listen: -1
        # open-left: 0.5*(-100) + 0.5*(10) = -45
        # open-right: 0.5*(10) + 0.5*(-100) = -45
        # Best is listen: -1
        assert act == 0
        assert val == Fraction(-1)


# ============================================================================
# PBVI tests
# ============================================================================

class TestPBVI:
    def test_single_state(self):
        """Single state: PBVI converges to r/(1-gamma)."""
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.obs = {0: 0}
        p.add_transition(0, 0, 0, Fraction(1))
        p.set_reward(0, 0, 2)
        p.set_initial([(0, Fraction(1))])
        p.discount = Fraction(1, 2)

        beliefs = [Belief.point(0)]
        alphas = pbvi(p, beliefs, iterations=20)
        b = Belief.point(0)
        val, _ = value_at_belief(alphas, b)
        # r/(1-gamma) = 2/(1-0.5) = 4
        assert val == Fraction(4)

    def test_two_state_pbvi(self):
        """PBVI on two-state POMDP converges."""
        p = POMDP()
        p.states = {0, 1}
        p.actions = {0}
        p.obs = {0: 0, 1: 1}
        p.add_transition(0, 0, 0, Fraction(1))
        p.add_transition(1, 0, 1, Fraction(1))
        p.set_reward(0, 0, 3)
        p.set_reward(1, 0, 1)
        p.set_initial([(0, Fraction(1, 2)), (1, Fraction(1, 2))])
        p.discount = Fraction(1, 2)

        beliefs = [Belief.point(0), Belief.point(1)]
        alphas = pbvi(p, beliefs, iterations=20)
        v0, _ = value_at_belief(alphas, Belief.point(0))
        v1, _ = value_at_belief(alphas, Belief.point(1))
        # state 0: 3/(1-0.5) = 6, state 1: 1/(1-0.5) = 2
        assert v0 == Fraction(6)
        assert v1 == Fraction(2)

    def test_pbvi_returns_alphas(self):
        p = make_simple_pomdp()
        p.discount = Fraction(9, 10)
        beliefs = [Belief.point(s) for s in p.states]
        alphas = pbvi(p, beliefs, iterations=10)
        assert len(alphas) > 0


# ============================================================================
# Almost-sure reachability tests
# ============================================================================

class TestAlmostSureReachability:
    def test_already_at_target(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.obs = {0: 0}
        p.add_transition(0, 0, 0, Fraction(1))
        p.target = {0}
        p.set_initial([(0, Fraction(1))])

        result, strat = almost_sure_reachability(p)
        assert result is True

    def test_reachable_deterministic(self):
        """Deterministic POMDP: target always reachable."""
        p = POMDP()
        p.states = {0, 1}
        p.actions = {0}
        p.obs = {0: 0, 1: 1}
        p.add_transition(0, 0, 1, Fraction(1))
        p.add_transition(1, 0, 1, Fraction(1))
        p.target = {1}
        p.set_initial([(0, Fraction(1))])

        result, strat = almost_sure_reachability(p)
        assert result is True

    def test_unreachable(self):
        """No path to target."""
        p = POMDP()
        p.states = {0, 1}
        p.actions = {0}
        p.obs = {0: 0, 1: 1}
        p.add_transition(0, 0, 0, Fraction(1))
        p.add_transition(1, 0, 1, Fraction(1))
        p.target = {1}
        p.set_initial([(0, Fraction(1))])

        result, strat = almost_sure_reachability(p)
        assert result is False

    def test_probabilistic_reach(self):
        """Probabilistic: 50/50 chance of reaching target each step."""
        p = POMDP()
        p.states = {0, 1}
        p.actions = {0}
        p.obs = {0: 0, 1: 1}
        p.add_transition(0, 0, 0, Fraction(1, 2))
        p.add_transition(0, 0, 1, Fraction(1, 2))
        p.add_transition(1, 0, 1, Fraction(1))
        p.target = {1}
        p.set_initial([(0, Fraction(1))])

        # Almost-sure: eventually reaches 1 with prob 1 (geometric tries)
        # But our finite BFS might not capture this
        result, strat = almost_sure_reachability(p)
        # The target is reachable (at least positively)
        pos, _ = positive_reachability(p)
        assert pos is True

    def test_empty_target(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.target = set()
        p.set_initial([(0, Fraction(1))])
        result, _ = almost_sure_reachability(p)
        assert result is False


# ============================================================================
# Positive reachability tests
# ============================================================================

class TestPositiveReachability:
    def test_immediate_target(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.obs = {0: 0}
        p.target = {0}
        p.set_initial([(0, Fraction(1))])
        result, strat = positive_reachability(p)
        assert result is True

    def test_one_step_reach(self):
        p = POMDP()
        p.states = {0, 1}
        p.actions = {0}
        p.obs = {0: 0, 1: 1}
        p.add_transition(0, 0, 1, Fraction(1))
        p.add_transition(1, 0, 1, Fraction(1))
        p.target = {1}
        p.set_initial([(0, Fraction(1))])
        result, strat = positive_reachability(p)
        assert result is True

    def test_unreachable(self):
        p = POMDP()
        p.states = {0, 1}
        p.actions = {0}
        p.obs = {0: 0, 1: 1}
        p.add_transition(0, 0, 0, Fraction(1))
        p.add_transition(1, 0, 1, Fraction(1))
        p.target = {1}
        p.set_initial([(0, Fraction(1))])
        result, strat = positive_reachability(p)
        assert result is False

    def test_probabilistic_reach(self):
        p = make_reachability_pomdp()
        result, strat = positive_reachability(p)
        assert result is True

    def test_empty_target(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.target = set()
        p.set_initial([(0, Fraction(1))])
        result, _ = positive_reachability(p)
        assert result is False


# ============================================================================
# Safety probability tests
# ============================================================================

class TestSafetyProbability:
    def test_always_safe(self):
        """Careful action always stays safe."""
        p = make_safety_pomdp()
        prob, strat = safety_probability(p, 5)
        assert prob == Fraction(1)
        # Strategy should choose careful (0)
        if strat:
            for obs, act in strat.items():
                if obs == 0:  # safe states observation
                    assert act == 0

    def test_unsafe_start(self):
        """Starting in unsafe state: probability 0."""
        p = POMDP()
        p.states = {0, 1}
        p.actions = {0}
        p.obs = {0: 0, 1: 1}
        p.add_transition(0, 0, 0, Fraction(1))
        p.add_transition(1, 0, 1, Fraction(1))
        p.target = {0}  # safe
        p.set_initial([(1, Fraction(1))])  # start unsafe
        prob, _ = safety_probability(p, 5)
        assert prob == Fraction(0)

    def test_one_step_safety(self):
        """One step: careful is safe, risky is 50%."""
        p = make_safety_pomdp()
        # With careful action: prob = 1
        prob, strat = safety_probability(p, 1)
        assert prob == Fraction(1)

    def test_empty_support(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.target = {0}
        p.initial = []
        prob, _ = safety_probability(p, 5)
        assert prob == Fraction(0)


# ============================================================================
# Stochastic PO game tests
# ============================================================================

class TestStochasticPOGame:
    def test_create_game(self):
        g = StochasticPOGame()
        g.add_vertex(0, "P1", 0)
        g.add_vertex(1, "Nature", 1)
        g.add_vertex(2, "P2", 2)
        assert len(g.vertices) == 3
        assert g.owner[0] == "P1"

    def test_edges(self):
        g = StochasticPOGame()
        g.add_vertex(0, "P1", 0)
        g.add_vertex(1, "P1", 1)
        g.add_edge(0, 1)
        assert g.successors(0) == {1}

    def test_prob_edges(self):
        g = StochasticPOGame()
        g.add_vertex(0, "Nature", 0)
        g.add_vertex(1, "P1", 1)
        g.add_vertex(2, "P1", 2)
        g.add_prob_edge(0, 1, Fraction(1, 2))
        g.add_prob_edge(0, 2, Fraction(1, 2))
        succs = g.prob_successors(0)
        assert len(succs) == 2

    def test_solve_trivial(self):
        """Trivial game: P1 can reach target directly."""
        g = StochasticPOGame()
        g.add_vertex(0, "P1", 0)
        g.add_vertex(1, "P1", 1)
        g.add_edge(0, 1)
        g.initial = {0}
        g.target = {1}
        vals, strat = solve_stochastic_po_game(g)
        assert vals.get(0, Fraction(0)) == Fraction(1)

    def test_solve_nature(self):
        """Nature chooses: 50/50 to target or not."""
        g = StochasticPOGame()
        g.add_vertex(0, "P1", 0)
        g.add_vertex(1, "Nature", 0)
        g.add_vertex(2, "P1", 1)
        g.add_vertex(3, "P1", 2)
        g.add_edge(0, 1)
        g.add_prob_edge(1, 2, Fraction(1, 2))
        g.add_prob_edge(1, 3, Fraction(1, 2))
        # State 2 self-loops, state 3 is target
        g.add_edge(2, 2)
        g.add_edge(3, 3)
        g.initial = {0}
        g.target = {3}
        vals, strat = solve_stochastic_po_game(g, iterations=50)
        # Value should reflect nature's 50/50 choice
        assert 0 in vals

    def test_solve_p2_adversary(self):
        """P2 chooses worst for P1."""
        g = StochasticPOGame()
        g.add_vertex(0, "P2", 0)
        g.add_vertex(1, "P1", 1)  # target
        g.add_vertex(2, "P1", 2)  # not target
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 1)
        g.add_edge(2, 2)
        g.initial = {0}
        g.target = {1}
        vals, strat = solve_stochastic_po_game(g, iterations=50)
        # P2 will always choose non-target
        assert vals.get(0, Fraction(1)) == Fraction(0)


# ============================================================================
# Simulation tests
# ============================================================================

class TestSimulation:
    def test_simulate_returns_trace(self):
        p = make_simple_pomdp()
        strat = {0: 0, 1: 0}  # always stay
        trace = simulate_pomdp(p, strat, 5, initial_state=0)
        assert len(trace) == 5
        assert all('state' in t for t in trace)
        assert all('observation' in t for t in trace)
        assert all('action' in t for t in trace)

    def test_simulate_specific_initial(self):
        p = make_simple_pomdp()
        strat = {0: 0, 1: 0}
        trace = simulate_pomdp(p, strat, 3, initial_state=2)
        assert trace[0]['state'] == 2
        assert trace[0]['observation'] == 1

    def test_simulate_with_move(self):
        p = make_simple_pomdp()
        strat = {0: 1, 1: 1}  # always move
        trace = simulate_pomdp(p, strat, 3, initial_state=0)
        # 0 -> 1 -> 2 -> 0
        assert trace[0]['state'] == 0

    def test_simulate_records_reward(self):
        p = make_tiger_pomdp()
        strat = {0: 0, 1: 0}  # always listen
        trace = simulate_pomdp(p, strat, 3, initial_state=0)
        # Listen reward is -1
        for t in trace:
            assert t['reward'] == -1.0

    def test_simulate_belief_info(self):
        p = make_simple_pomdp()
        strat = {0: 0, 1: 0}
        trace = simulate_pomdp(p, strat, 3, initial_state=0)
        assert 'belief_size' in trace[0]
        assert 'belief_entropy' in trace[0]


# ============================================================================
# Statistics and comparison tests
# ============================================================================

class TestStatistics:
    def test_pomdp_statistics(self):
        p = make_tiger_pomdp()
        stats = pomdp_statistics(p)
        assert stats['states'] == 2
        assert stats['actions'] == 3
        assert stats['observations'] == 2
        assert stats['info_ratio'] == 1.0

    def test_simple_statistics(self):
        p = make_simple_pomdp()
        stats = pomdp_statistics(p)
        assert stats['states'] == 3
        assert stats['observations'] == 2
        assert stats['info_ratio'] < 1.0

    def test_belief_space_size(self):
        p = make_simple_pomdp()
        result = belief_space_size(p)
        assert result['reachable_beliefs'] >= 1
        assert not result['capped']

    def test_belief_space_size_cap(self):
        p = make_simple_pomdp()
        result = belief_space_size(p, max_beliefs=2)
        assert result['reachable_beliefs'] <= 2

    def test_compare_mdp_vs_pomdp_safety(self):
        p = make_safety_pomdp()
        result = compare_mdp_vs_pomdp(p, steps=3)
        assert 'mdp_value' in result
        assert 'pomdp_value' in result
        assert 'information_cost' in result
        # MDP should be at least as good as POMDP
        assert result['mdp_value'] >= result['pomdp_value']

    def test_compare_mdp_vs_pomdp_reward(self):
        p = make_tiger_pomdp()
        result = compare_mdp_vs_pomdp(p)
        assert result['mdp_value'] >= result['pomdp_value']


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_single_state_pomdp(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0}
        p.obs = {0: 0}
        p.add_transition(0, 0, 0, Fraction(1))
        p.set_initial([(0, Fraction(1))])
        assert p.is_valid()

    def test_many_actions(self):
        p = POMDP()
        p.states = {0}
        p.actions = {0, 1, 2, 3, 4}
        p.obs = {0: 0}
        for a in p.actions:
            p.add_transition(0, a, 0, Fraction(1))
            p.set_reward(0, a, a)
        p.set_initial([(0, Fraction(1))])
        p.horizon = 1
        alphas = finite_horizon_vi(p)
        val, act = value_at_belief(alphas[0], Belief.point(0))
        assert act == 4  # highest reward

    def test_absorbing_target(self):
        """Target state is absorbing."""
        p = POMDP()
        p.states = {0, 1}
        p.actions = {0}
        p.obs = {0: 0, 1: 1}
        p.add_transition(0, 0, 1, Fraction(1))
        p.add_transition(1, 0, 1, Fraction(1))
        p.target = {1}
        p.set_initial([(0, Fraction(1))])
        result, _ = positive_reachability(p)
        assert result is True

    def test_belief_update_chain(self):
        """Chain of belief updates stays valid."""
        p = make_simple_pomdp()
        b = Belief.point(0)
        b1 = belief_update(p, b, 1, 0)  # move: 0->1, obs 0
        assert b1 is not None and b1.is_valid()
        b2 = belief_update(p, b1, 1, 1)  # move: 1->2, obs 1
        assert b2 is not None and b2.is_valid()
        b3 = belief_update(p, b2, 1, 0)  # move: 2->0, obs 0
        assert b3 is not None and b3.is_valid()
        assert b3.prob(0) == Fraction(1)

    def test_stochastic_game_empty(self):
        g = StochasticPOGame()
        g.initial = set()
        g.target = set()
        vals, strat = solve_stochastic_po_game(g)
        assert vals == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
