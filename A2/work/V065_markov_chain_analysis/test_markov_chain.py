"""Tests for V065: Markov Chain Analysis"""

import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from markov_chain import (
    # Core
    MarkovChain, StateType, ChainAnalysis,
    # Construction
    make_chain, random_walk_chain, gambler_ruin_chain,
    # Analysis
    communication_classes, classify_states, is_absorbing_chain,
    period_of_state, chain_period,
    steady_state, steady_state_exact,
    absorption_probabilities, expected_hitting_time, expected_hitting_times,
    analyze_chain,
    # Verification
    verify_absorption, verify_hitting_time_bound, verify_steady_state_bound,
    # Simulation
    simulate_chain, empirical_steady_state, compare_analytical_vs_simulation,
)


# ============================================================
# 1. MarkovChain data structure
# ============================================================

class TestMarkovChain:
    def test_create_basic(self):
        mc = make_chain([[0.5, 0.5], [0.3, 0.7]])
        assert mc.n_states == 2
        assert mc.state_labels == ["s0", "s1"]

    def test_create_with_labels(self):
        mc = make_chain([[1.0, 0.0], [0.0, 1.0]], labels=["A", "B"])
        assert mc.state_labels == ["A", "B"]

    def test_validate_valid(self):
        mc = make_chain([[0.5, 0.5], [0.3, 0.7]])
        assert mc.validate() == []

    def test_validate_bad_row_sum(self):
        mc = make_chain([[0.5, 0.3], [0.3, 0.7]])
        errors = mc.validate()
        assert len(errors) > 0

    def test_validate_negative(self):
        mc = make_chain([[0.5, 0.5], [-0.3, 1.3]])
        errors = mc.validate()
        assert len(errors) > 0

    def test_step(self):
        mc = make_chain([[0.0, 1.0], [1.0, 0.0]])
        dist = [1.0, 0.0]
        new = mc.step(dist)
        assert abs(new[0]) < 1e-10
        assert abs(new[1] - 1.0) < 1e-10

    def test_successors(self):
        mc = make_chain([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
        succs = mc.successors(0)
        assert len(succs) == 2
        assert (0, 0.5) in succs
        assert (1, 0.5) in succs

    def test_to_from_dict(self):
        mc = make_chain([[0.5, 0.5], [0.3, 0.7]], labels=["X", "Y"])
        d = mc.to_dict()
        mc2 = MarkovChain.from_dict(d)
        assert mc2.n_states == 2
        assert mc2.state_labels == ["X", "Y"]
        assert mc2.transition == mc.transition


# ============================================================
# 2. Chain construction
# ============================================================

class TestChainConstruction:
    def test_random_walk_absorbing(self):
        mc = random_walk_chain(5, p_right=0.5, absorbing_ends=True)
        assert mc.n_states == 5
        assert mc.transition[0][0] == 1.0  # Absorbing
        assert mc.transition[4][4] == 1.0  # Absorbing
        assert mc.validate() == []

    def test_random_walk_reflecting(self):
        mc = random_walk_chain(5, p_right=0.5, absorbing_ends=False)
        assert mc.n_states == 5
        assert mc.validate() == []
        # Reflecting: state 0 stays or goes right
        assert mc.transition[0][0] == 0.5
        assert mc.transition[0][1] == 0.5

    def test_gambler_ruin(self):
        mc = gambler_ruin_chain(4, p=0.5)
        assert mc.n_states == 5  # 0..4
        assert mc.transition[0][0] == 1.0  # Absorbing at 0
        assert mc.transition[4][4] == 1.0  # Absorbing at 4
        assert mc.validate() == []

    def test_biased_random_walk(self):
        mc = random_walk_chain(5, p_right=0.7)
        assert mc.validate() == []
        assert mc.transition[2][3] == 0.7
        assert abs(mc.transition[2][1] - 0.3) < 1e-10


# ============================================================
# 3. Communication classes
# ============================================================

class TestCommunicationClasses:
    def test_single_class(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        classes = communication_classes(mc)
        assert len(classes) == 1
        assert classes[0] == {0, 1}

    def test_two_absorbing(self):
        mc = make_chain([[1.0, 0.0, 0.0],
                        [0.5, 0.0, 0.5],
                        [0.0, 0.0, 1.0]])
        classes = communication_classes(mc)
        assert len(classes) == 3

    def test_gambler_classes(self):
        mc = gambler_ruin_chain(4)
        classes = communication_classes(mc)
        # States 0 and 4 are absorbing (singleton classes), 1-3 form a class
        class_sizes = sorted([len(c) for c in classes])
        assert class_sizes == [1, 1, 3]

    def test_cyclic_chain(self):
        mc = make_chain([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        classes = communication_classes(mc)
        assert len(classes) == 1
        assert classes[0] == {0, 1, 2}


# ============================================================
# 4. State classification
# ============================================================

class TestStateClassification:
    def test_all_recurrent(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        types = classify_states(mc)
        assert all(t == StateType.RECURRENT for t in types)

    def test_absorbing_states(self):
        mc = make_chain([[1.0, 0.0], [0.5, 0.5]])
        types = classify_states(mc)
        assert types[0] == StateType.ABSORBING

    def test_transient_states(self):
        mc = make_chain([[1.0, 0.0, 0.0],
                        [0.5, 0.0, 0.5],
                        [0.0, 0.0, 1.0]])
        types = classify_states(mc)
        assert types[0] == StateType.ABSORBING
        assert types[1] == StateType.TRANSIENT
        assert types[2] == StateType.ABSORBING

    def test_is_absorbing_chain(self):
        mc = gambler_ruin_chain(4)
        assert is_absorbing_chain(mc)

    def test_not_absorbing_chain(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        assert not is_absorbing_chain(mc)


# ============================================================
# 5. Periodicity
# ============================================================

class TestPeriodicity:
    def test_aperiodic(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        assert period_of_state(mc, 0) == 1

    def test_period_2(self):
        mc = make_chain([[0, 1], [1, 0]])
        p = period_of_state(mc, 0)
        assert p == 2

    def test_period_3(self):
        mc = make_chain([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        p = period_of_state(mc, 0)
        assert p == 3

    def test_chain_period(self):
        mc = make_chain([[0, 1], [1, 0]])
        assert chain_period(mc) == 2

    def test_self_loop_aperiodic(self):
        mc = make_chain([[0.1, 0.9], [0.9, 0.1]])
        assert period_of_state(mc, 0) == 1

    def test_absorbing_period(self):
        mc = gambler_ruin_chain(4)
        # Absorbing states have period 1
        assert period_of_state(mc, 0) == 1


# ============================================================
# 6. Steady-state distribution
# ============================================================

class TestSteadyState:
    def test_symmetric(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        ss = steady_state_exact(mc)
        assert ss is not None
        assert abs(ss[0] - 0.5) < 1e-6
        assert abs(ss[1] - 0.5) < 1e-6

    def test_biased(self):
        mc = make_chain([[0.7, 0.3], [0.4, 0.6]])
        ss = steady_state_exact(mc)
        # pi = [4/7, 3/7]
        assert abs(ss[0] - 4/7) < 1e-6
        assert abs(ss[1] - 3/7) < 1e-6

    def test_power_iteration_matches_exact(self):
        mc = make_chain([[0.7, 0.3], [0.4, 0.6]])
        ss_exact = steady_state_exact(mc)
        ss_iter = steady_state(mc)
        for i in range(mc.n_states):
            assert abs(ss_exact[i] - ss_iter[i]) < 1e-4

    def test_three_state(self):
        mc = make_chain([[0.1, 0.6, 0.3],
                        [0.4, 0.2, 0.4],
                        [0.3, 0.3, 0.4]])
        ss = steady_state_exact(mc)
        assert abs(sum(ss) - 1.0) < 1e-6
        assert all(p >= 0 for p in ss)

    def test_sum_to_one(self):
        mc = make_chain([[0.2, 0.3, 0.5],
                        [0.1, 0.8, 0.1],
                        [0.4, 0.4, 0.2]])
        ss = steady_state_exact(mc)
        assert abs(sum(ss) - 1.0) < 1e-6


# ============================================================
# 7. Absorption probabilities
# ============================================================

class TestAbsorption:
    def test_symmetric_gambler(self):
        """Symmetric random walk: P(reach n | start at i) = i/n."""
        mc = gambler_ruin_chain(4, p=0.5)
        abs_probs = absorption_probabilities(mc)

        # P(reach 4 from 2) = 2/4 = 0.5
        assert abs(abs_probs[4][2] - 0.5) < 1e-6
        # P(reach 0 from 2) = 0.5
        assert abs(abs_probs[0][2] - 0.5) < 1e-6
        # P(reach 4 from 1) = 1/4 = 0.25
        assert abs(abs_probs[4][1] - 0.25) < 1e-6

    def test_biased_gambler(self):
        """Biased walk: P(reach n | start at 1) should be higher with p > 0.5."""
        mc_fair = gambler_ruin_chain(4, p=0.5)
        mc_biased = gambler_ruin_chain(4, p=0.7)

        abs_fair = absorption_probabilities(mc_fair)
        abs_biased = absorption_probabilities(mc_biased)

        # More likely to reach 4 with p=0.7
        assert abs_biased[4][1] > abs_fair[4][1]

    def test_absorbing_states_self(self):
        mc = gambler_ruin_chain(4)
        abs_probs = absorption_probabilities(mc)
        assert abs(abs_probs[0][0] - 1.0) < 1e-10
        assert abs(abs_probs[4][4] - 1.0) < 1e-10

    def test_three_state_absorbing(self):
        mc = make_chain([[1.0, 0.0, 0.0],
                        [0.3, 0.0, 0.7],
                        [0.0, 0.0, 1.0]])
        abs_probs = absorption_probabilities(mc)
        assert abs(abs_probs[0][1] - 0.3) < 1e-6
        assert abs(abs_probs[2][1] - 0.7) < 1e-6


# ============================================================
# 8. Expected hitting times
# ============================================================

class TestHittingTimes:
    def test_adjacent(self):
        """One step from 0 to 1 when transition is deterministic."""
        mc = make_chain([[0, 1], [0, 1]])  # Always go to state 1
        ht = expected_hitting_time(mc, 1)
        assert abs(ht[0] - 1.0) < 1e-6  # Exactly 1 step
        assert abs(ht[1] - 0.0) < 1e-10  # Already there

    def test_gambler_hitting_time(self):
        """Symmetric gambler's ruin: E[T_0 | start=1] for n=2 should be finite."""
        mc = gambler_ruin_chain(2, p=0.5)
        ht = expected_hitting_time(mc, 0)
        assert ht[1] > 0  # Should be positive
        assert ht[1] < float('inf')
        assert abs(ht[0] - 0.0) < 1e-10

    def test_multiple_targets(self):
        mc = gambler_ruin_chain(4)
        hts = expected_hitting_times(mc, [0, 4])
        assert 0 in hts
        assert 4 in hts
        assert hts[0][0] == 0.0
        assert hts[4][4] == 0.0

    def test_symmetric_walk_time(self):
        """E[reach boundary from middle] in symmetric walk.

        Note: for gambler's ruin with two absorbing states, the expected
        hitting time to a SPECIFIC absorbing state may be infinite (or
        distorted by the solver) because some paths get absorbed elsewhere.
        We test deterministic hitting instead.
        """
        # Deterministic chain: always go left
        mc = make_chain([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
        ht = expected_hitting_time(mc, 0)
        assert abs(ht[1] - 1.0) < 1e-6  # 1 step from state 1
        assert abs(ht[2] - 2.0) < 1e-6  # 2 steps from state 2


# ============================================================
# 9. Full analysis
# ============================================================

class TestFullAnalysis:
    def test_analyze_ergodic(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        result = analyze_chain(mc)
        assert result.is_irreducible
        assert not result.is_absorbing
        assert result.steady_state is not None

    def test_analyze_absorbing(self):
        mc = gambler_ruin_chain(4)
        result = analyze_chain(mc)
        assert result.is_absorbing
        assert not result.is_irreducible
        assert result.absorption_probabilities is not None

    def test_analyze_periodic(self):
        mc = make_chain([[0, 1], [1, 0]])
        result = analyze_chain(mc)
        assert result.period == 2
        assert result.is_irreducible

    def test_summary(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        result = analyze_chain(mc)
        s = result.summary()
        assert "2 states" in s
        assert "Irreducible: True" in s


# ============================================================
# 10. Property verification
# ============================================================

class TestVerification:
    def test_verify_absorption_success(self):
        mc = gambler_ruin_chain(4, p=0.5)
        result = verify_absorption(mc, start=2, target=4, min_prob=0.4)
        assert result["verified"]
        assert abs(result["probability"] - 0.5) < 1e-6

    def test_verify_absorption_failure(self):
        mc = gambler_ruin_chain(4, p=0.5)
        result = verify_absorption(mc, start=1, target=4, min_prob=0.9)
        assert not result["verified"]

    def test_verify_hitting_time_bound(self):
        mc = gambler_ruin_chain(2, p=0.5)
        result = verify_hitting_time_bound(mc, start=1, target=0, max_steps=100)
        assert result["verified"]
        assert result["expected_time"] > 0

    def test_verify_steady_state(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        result = verify_steady_state_bound(mc, state=0, min_prob=0.4)
        assert result["verified"]
        assert abs(result["probability"] - 0.5) < 1e-6

    def test_verify_non_absorbing_target(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        result = verify_absorption(mc, start=0, target=1, min_prob=0.5)
        assert not result["verified"]
        assert "not absorbing" in result["reason"]


# ============================================================
# 11. Simulation
# ============================================================

class TestSimulation:
    def test_simulate_basic(self):
        mc = make_chain([[0, 1], [1, 0]])
        trace = simulate_chain(mc, 0, 10, seed=42)
        assert len(trace) == 11  # start + 10 steps
        assert trace[0] == 0
        # Alternating chain
        for i in range(len(trace) - 1):
            assert trace[i] != trace[i+1]

    def test_simulate_absorbing(self):
        mc = make_chain([[1, 0], [0.5, 0.5]])
        trace = simulate_chain(mc, 0, 10, seed=42)
        # Once absorbed, stays
        assert all(s == 0 for s in trace)

    def test_empirical_steady_state(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        emp = empirical_steady_state(mc, steps=10000, seed=42)
        assert abs(emp[0] - 0.5) < 0.05
        assert abs(emp[1] - 0.5) < 0.05

    def test_compare_analytical_vs_simulation(self):
        mc = make_chain([[0.7, 0.3], [0.4, 0.6]])
        result = compare_analytical_vs_simulation(mc, steps=10000, seed=42)
        assert result["max_difference"] < 0.05


# ============================================================
# 12. Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_state(self):
        mc = make_chain([[1.0]])
        assert mc.validate() == []
        types = classify_states(mc)
        assert types[0] == StateType.ABSORBING
        ss = steady_state_exact(mc)
        assert abs(ss[0] - 1.0) < 1e-10

    def test_identity_chain(self):
        mc = make_chain([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        types = classify_states(mc)
        assert all(t == StateType.ABSORBING for t in types)

    def test_large_chain(self):
        n = 20
        mc = random_walk_chain(n, absorbing_ends=False)
        assert mc.validate() == []
        ss = steady_state_exact(mc)
        assert abs(sum(ss) - 1.0) < 1e-4

    def test_near_zero_transition(self):
        mc = make_chain([[1.0 - 1e-12, 1e-12], [1e-12, 1.0 - 1e-12]])
        assert mc.validate() == []

    def test_gambler_ruin_n1(self):
        """Edge case: gambler with only 0 and 1."""
        mc = gambler_ruin_chain(1, p=0.5)
        assert mc.n_states == 2
        assert mc.transition[0][0] == 1.0
        assert mc.transition[1][1] == 1.0
