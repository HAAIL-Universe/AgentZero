"""Tests for V150: Weak Probabilistic Bisimulation."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V148_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))

from weak_probabilistic_bisimulation import (
    TAU, WeakBisimVerdict, WeakBisimResult, LabeledProbTS,
    make_labeled_prob_ts, compute_weak_bisimulation, check_weakly_bisimilar,
    compute_branching_bisimulation, check_branching_bisimilar,
    weak_bisimulation_quotient, branching_bisimulation_quotient,
    check_cross_weak_bisimulation, compute_weak_simulation,
    detect_divergence, compute_divergence_sensitive_bisimulation,
    lmc_to_prob_ts, prob_ts_to_lmc, compute_weak_bisimulation_distance,
    compare_strong_vs_weak, minimize_weak, minimize_branching,
    weak_bisimulation_summary,
    _tau_closure, _weak_transition, _stable_states,
)
from pctl_model_check import make_labeled_mc


# ---- Helper: Build simple systems ----

def simple_tau_chain():
    """s0 --tau--> s1 --a--> s2. s0 and s1 should be weakly bisimilar if same labels."""
    return make_labeled_prob_ts(
        n_states=3,
        actions={
            0: {TAU: [(1, 1.0)]},
            1: {"a": [(2, 1.0)]},
            2: {},
        },
        state_labels={0: {"start"}, 1: {"start"}, 2: {"end"}},
    )


def two_state_tau():
    """s0 --tau--> s1, both labeled same. Weakly bisimilar."""
    return make_labeled_prob_ts(
        n_states=2,
        actions={
            0: {TAU: [(1, 1.0)]},
            1: {},
        },
        state_labels={0: {"x"}, 1: {"x"}},
    )


def diamond_system():
    """
    s0 --a--> s1 (p=0.5), s0 --a--> s2 (p=0.5)
    s1 --tau--> s3, s2 --tau--> s3
    s1, s2 have same labels -> weakly bisimilar
    """
    return make_labeled_prob_ts(
        n_states=4,
        actions={
            0: {"a": [(1, 0.5), (2, 0.5)]},
            1: {TAU: [(3, 1.0)]},
            2: {TAU: [(3, 1.0)]},
            3: {},
        },
        state_labels={0: {"init"}, 1: {"mid"}, 2: {"mid"}, 3: {"done"}},
    )


def distinct_system():
    """Two states with different labels -- never bisimilar."""
    return make_labeled_prob_ts(
        n_states=2,
        actions={
            0: {"a": [(1, 1.0)]},
            1: {},
        },
        state_labels={0: {"x"}, 1: {"y"}},
    )


def probabilistic_tau():
    """s0 --tau--> s1 (p=0.5) or s2 (p=0.5). s1, s2 have different labels."""
    return make_labeled_prob_ts(
        n_states=3,
        actions={
            0: {TAU: [(1, 0.5), (2, 0.5)]},
            1: {},
            2: {},
        },
        state_labels={0: {"x"}, 1: {"a"}, 2: {"b"}},
    )


# ---- Tests: LabeledProbTS construction ----

class TestConstruction:
    def test_basic_construction(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {"a": [(1, 1.0)]}, 1: {}},
            state_labels={0: {"x"}, 1: {"y"}},
        )
        assert ts.n_states == 2
        assert ts.observable_actions(0) == {"a"}
        assert ts.observable_actions(1) == set()

    def test_tau_detection(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {TAU: [(1, 1.0)]}, 1: {}},
            state_labels={0: set(), 1: set()},
        )
        assert ts.has_tau(0)
        assert not ts.has_tau(1)
        assert ts.tau_successors(0) == [(1, 1.0)]
        assert ts.tau_successors(1) == []

    def test_mixed_actions(self):
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {TAU: [(1, 1.0)], "a": [(2, 1.0)]},
                1: {},
                2: {},
            },
            state_labels={0: set(), 1: set(), 2: set()},
        )
        assert ts.has_tau(0)
        assert ts.observable_actions(0) == {"a"}

    def test_probability_validation(self):
        with pytest.raises(ValueError):
            make_labeled_prob_ts(
                n_states=2,
                actions={0: {"a": [(1, 0.5)]}},  # doesn't sum to 1
                state_labels={0: set(), 1: set()},
            )

    def test_default_labels(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {}, 1: {}},
            state_labels={},
        )
        assert ts.state_labels.get(0) == set()

    def test_state_names_auto(self):
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={},
            state_labels={},
        )
        assert ts.state_names == ["s0", "s1", "s2"]

    def test_action_distribution(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {"a": [(1, 1.0)]}},
            state_labels={0: set(), 1: set()},
        )
        assert ts.action_distribution(0, "a") == [(1, 1.0)]
        assert ts.action_distribution(0, "b") == []
        assert ts.action_distribution(1, "a") == []


# ---- Tests: Tau Closure ----

class TestTauClosure:
    def test_no_tau(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {"a": [(1, 1.0)]}, 1: {}},
            state_labels={0: set(), 1: set()},
        )
        closure = _tau_closure(ts, 0)
        assert abs(closure.get(0, 0.0) - 1.0) < 1e-9

    def test_single_tau_step(self):
        ts = two_state_tau()
        closure = _tau_closure(ts, 0)
        assert abs(closure.get(1, 0.0) - 1.0) < 1e-9

    def test_chain_tau(self):
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {TAU: [(1, 1.0)]},
                1: {TAU: [(2, 1.0)]},
                2: {},
            },
            state_labels={0: set(), 1: set(), 2: set()},
        )
        closure = _tau_closure(ts, 0)
        assert abs(closure.get(2, 0.0) - 1.0) < 1e-9

    def test_probabilistic_tau(self):
        ts = probabilistic_tau()
        closure = _tau_closure(ts, 0)
        assert abs(closure.get(1, 0.0) - 0.5) < 1e-9
        assert abs(closure.get(2, 0.0) - 0.5) < 1e-9

    def test_stable_state_closure(self):
        ts = distinct_system()
        closure = _tau_closure(ts, 1)
        assert abs(closure.get(1, 0.0) - 1.0) < 1e-9


# ---- Tests: Weak Transition ----

class TestWeakTransition:
    def test_observable_no_tau(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {"a": [(1, 1.0)]}, 1: {}},
            state_labels={0: set(), 1: set()},
        )
        wt = _weak_transition(ts, 0, "a")
        assert abs(wt.get(1, 0.0) - 1.0) < 1e-9

    def test_tau_then_observable(self):
        ts = simple_tau_chain()
        wt = _weak_transition(ts, 0, "a")
        assert abs(wt.get(2, 0.0) - 1.0) < 1e-9

    def test_no_action_available(self):
        ts = simple_tau_chain()
        wt = _weak_transition(ts, 2, "a")
        assert len(wt) == 0

    def test_observable_then_tau(self):
        """s0 --a--> s1 --tau--> s2"""
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {"a": [(1, 1.0)]},
                1: {TAU: [(2, 1.0)]},
                2: {},
            },
            state_labels={0: set(), 1: set(), 2: set()},
        )
        wt = _weak_transition(ts, 0, "a")
        assert abs(wt.get(2, 0.0) - 1.0) < 1e-9

    def test_weak_tau_is_closure(self):
        ts = two_state_tau()
        wt = _weak_transition(ts, 0, TAU)
        assert abs(wt.get(1, 0.0) - 1.0) < 1e-9


# ---- Tests: Stable States ----

class TestStableStates:
    def test_stable_detection(self):
        ts = simple_tau_chain()
        stable = _stable_states(ts)
        assert 1 in stable  # s1 has action 'a', not tau
        assert 2 in stable  # s2 has no actions
        assert 0 not in stable  # s0 has tau


# ---- Tests: Weak Bisimulation ----

class TestWeakBisimulation:
    def test_tau_chain_collapses(self):
        """s0 --tau--> s1 with same labels should be weakly bisimilar."""
        ts = two_state_tau()
        result = compute_weak_bisimulation(ts)
        assert result.statistics["num_blocks"] == 1
        assert result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR

    def test_different_labels_separate(self):
        ts = distinct_system()
        result = compute_weak_bisimulation(ts)
        assert result.statistics["num_blocks"] == 2

    def test_simple_tau_chain(self):
        """s0 --tau--> s1 --a--> s2: s0 and s1 same labels, both can do a weakly."""
        ts = simple_tau_chain()
        result = compute_weak_bisimulation(ts)
        # s0 and s1 should collapse (both have label "start", both can weakly do "a" -> s2)
        found_merged = False
        for block in result.partition:
            if 0 in block and 1 in block:
                found_merged = True
        assert found_merged

    def test_diamond_collapses_mid(self):
        ts = diamond_system()
        result = compute_weak_bisimulation(ts)
        # s1, s2 both labeled "mid", both tau -> s3: should collapse
        found_mid_block = False
        for block in result.partition:
            if 1 in block and 2 in block:
                found_mid_block = True
        assert found_mid_block

    def test_no_tau_is_strong(self):
        """System with no tau transitions: weak == strong bisimulation."""
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {"a": [(1, 0.5), (2, 0.5)]},
                1: {"b": [(0, 1.0)]},
                2: {"b": [(0, 1.0)]},
            },
            state_labels={0: {"x"}, 1: {"y"}, 2: {"y"}},
        )
        result = compute_weak_bisimulation(ts)
        # s1 and s2 should be bisimilar (same labels, same actions)
        found_block = False
        for block in result.partition:
            if 1 in block and 2 in block:
                found_block = True
        assert found_block

    def test_singleton_system(self):
        ts = make_labeled_prob_ts(
            n_states=1,
            actions={0: {}},
            state_labels={0: {"x"}},
        )
        result = compute_weak_bisimulation(ts)
        assert result.statistics["num_blocks"] == 1

    def test_three_state_tau_chain(self):
        """s0 --tau--> s1 --tau--> s2, all same labels."""
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {TAU: [(1, 1.0)]},
                1: {TAU: [(2, 1.0)]},
                2: {},
            },
            state_labels={0: {"x"}, 1: {"x"}, 2: {"x"}},
        )
        result = compute_weak_bisimulation(ts)
        # All three should collapse
        assert result.statistics["num_blocks"] == 1

    def test_probabilistic_tau_different_labels(self):
        """s0 --tau--> s1 (0.5) or s2 (0.5), s1 != s2 labels."""
        ts = probabilistic_tau()
        result = compute_weak_bisimulation(ts)
        # s0 has label "x", s1 has "a", s2 has "b" -- all different, 3 blocks
        assert result.statistics["num_blocks"] == 3

    def test_parallel_tau_same_target(self):
        """Two states with tau to same target, same labels -> bisimilar."""
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {TAU: [(2, 1.0)]},
                1: {TAU: [(2, 1.0)]},
                2: {},
            },
            state_labels={0: {"x"}, 1: {"x"}, 2: {"x"}},
        )
        result = compute_weak_bisimulation(ts)
        assert result.statistics["num_blocks"] == 1


# ---- Tests: Check Weakly Bisimilar ----

class TestCheckWeaklyBisimilar:
    def test_bisimilar_pair(self):
        ts = two_state_tau()
        result = check_weakly_bisimilar(ts, 0, 1)
        assert result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR

    def test_not_bisimilar_pair(self):
        ts = distinct_system()
        result = check_weakly_bisimilar(ts, 0, 1)
        assert result.verdict == WeakBisimVerdict.NOT_WEAKLY_BISIMILAR
        assert result.witness is not None

    def test_label_mismatch_witness(self):
        ts = distinct_system()
        result = check_weakly_bisimilar(ts, 0, 1)
        assert "Label mismatch" in result.witness


# ---- Tests: Branching Bisimulation ----

class TestBranchingBisimulation:
    def test_tau_chain_same_labels(self):
        ts = two_state_tau()
        result = compute_branching_bisimulation(ts)
        assert result.statistics["num_blocks"] == 1
        assert result.verdict == WeakBisimVerdict.BRANCHING_BISIMILAR

    def test_different_labels(self):
        ts = distinct_system()
        result = compute_branching_bisimulation(ts)
        assert result.statistics["num_blocks"] == 2

    def test_branching_finer_than_weak(self):
        """Branching bisim is finer than weak bisim (more blocks or equal)."""
        ts = diamond_system()
        weak = compute_weak_bisimulation(ts)
        branching = compute_branching_bisimulation(ts)
        assert branching.statistics["num_blocks"] >= weak.statistics["num_blocks"]

    def test_stuttering_tau(self):
        """Tau transitions within the same block are stuttering (ignored)."""
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={
                0: {TAU: [(1, 1.0)]},
                1: {TAU: [(0, 1.0)]},
            },
            state_labels={0: {"x"}, 1: {"x"}},
        )
        result = compute_branching_bisimulation(ts)
        # Both have same label, tau between them is stuttering -> bisimilar
        assert result.statistics["num_blocks"] == 1

    def test_check_branching_bisimilar(self):
        ts = two_state_tau()
        result = check_branching_bisimilar(ts, 0, 1)
        assert result.verdict == WeakBisimVerdict.BRANCHING_BISIMILAR

    def test_check_not_branching_bisimilar(self):
        ts = distinct_system()
        result = check_branching_bisimilar(ts, 0, 1)
        assert result.verdict == WeakBisimVerdict.NOT_BRANCHING_BISIMILAR

    def test_no_tau_system(self):
        """With no tau, branching bisim == strong bisim."""
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={
                0: {"a": [(1, 1.0)]},
                1: {"a": [(0, 1.0)]},
            },
            state_labels={0: {"x"}, 1: {"x"}},
        )
        result = compute_branching_bisimulation(ts)
        assert result.statistics["num_blocks"] == 1


# ---- Tests: Quotient Construction ----

class TestQuotient:
    def test_weak_quotient_reduces(self):
        ts = two_state_tau()
        quotient = weak_bisimulation_quotient(ts)
        assert quotient.n_states == 1

    def test_branching_quotient_reduces(self):
        ts = two_state_tau()
        quotient = branching_bisimulation_quotient(ts)
        assert quotient.n_states == 1

    def test_quotient_preserves_labels(self):
        ts = diamond_system()
        quotient = weak_bisimulation_quotient(ts)
        # Should have init, mid (collapsed), done = 3 blocks
        label_sets = [quotient.state_labels[s] for s in range(quotient.n_states)]
        assert {"init"} in label_sets
        assert {"done"} in label_sets

    def test_no_reduction_distinct(self):
        ts = distinct_system()
        quotient = weak_bisimulation_quotient(ts)
        assert quotient.n_states == 2

    def test_full_collapse(self):
        """All same labels, connected by tau: collapses to 1 state."""
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {TAU: [(1, 1.0)]},
                1: {TAU: [(2, 1.0)]},
                2: {},
            },
            state_labels={0: {"x"}, 1: {"x"}, 2: {"x"}},
        )
        quotient = weak_bisimulation_quotient(ts)
        assert quotient.n_states == 1

    def test_quotient_names(self):
        ts = diamond_system()
        quotient = weak_bisimulation_quotient(ts)
        # Check that merged blocks get composite names
        for name in quotient.state_names:
            assert isinstance(name, str)


# ---- Tests: Cross-System Weak Bisimulation ----

class TestCrossSystem:
    def test_identical_systems(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {"a": [(1, 1.0)]}, 1: {}},
            state_labels={0: {"x"}, 1: {"y"}},
        )
        result = check_cross_weak_bisimulation(ts, ts)
        assert result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR
        assert result.statistics["cross_pairs"] > 0

    def test_different_systems(self):
        ts1 = make_labeled_prob_ts(
            n_states=1,
            actions={0: {}},
            state_labels={0: {"x"}},
        )
        ts2 = make_labeled_prob_ts(
            n_states=1,
            actions={0: {}},
            state_labels={0: {"y"}},
        )
        result = check_cross_weak_bisimulation(ts1, ts2)
        assert result.verdict == WeakBisimVerdict.NOT_WEAKLY_BISIMILAR

    def test_tau_abstraction_cross(self):
        """System with tau should be cross-bisimilar to system without."""
        ts1 = make_labeled_prob_ts(
            n_states=2,
            actions={0: {TAU: [(1, 1.0)]}, 1: {"a": [(1, 1.0)]}},
            state_labels={0: {"x"}, 1: {"x"}},
        )
        ts2 = make_labeled_prob_ts(
            n_states=1,
            actions={0: {"a": [(0, 1.0)]}},
            state_labels={0: {"x"}},
        )
        result = check_cross_weak_bisimulation(ts1, ts2)
        assert result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR


# ---- Tests: Weak Simulation ----

class TestWeakSimulation:
    def test_self_simulation(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {"a": [(1, 1.0)]}, 1: {}},
            state_labels={0: {"x"}, 1: {"y"}},
        )
        result = compute_weak_simulation(ts)
        # Every state simulates itself
        assert result.statistics["relation_size"] >= ts.n_states

    def test_same_label_simulation(self):
        ts = two_state_tau()
        result = compute_weak_simulation(ts)
        # Both have same label; s0 -> s1 via tau, s1 is stable
        assert result.statistics["relation_size"] >= 2


# ---- Tests: Divergence Detection ----

class TestDivergence:
    def test_no_divergence(self):
        ts = distinct_system()
        div = detect_divergence(ts)
        assert not any(div.values())

    def test_tau_self_loop_divergent(self):
        ts = make_labeled_prob_ts(
            n_states=1,
            actions={0: {TAU: [(0, 1.0)]}},
            state_labels={0: {"x"}},
        )
        div = detect_divergence(ts)
        assert div[0]

    def test_tau_cycle_divergent(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={
                0: {TAU: [(1, 1.0)]},
                1: {TAU: [(0, 1.0)]},
            },
            state_labels={0: {"x"}, 1: {"x"}},
        )
        div = detect_divergence(ts)
        assert div[0]
        assert div[1]

    def test_tau_chain_not_divergent(self):
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {TAU: [(1, 1.0)]},
                1: {TAU: [(2, 1.0)]},
                2: {},
            },
            state_labels={0: set(), 1: set(), 2: set()},
        )
        div = detect_divergence(ts)
        assert not div[2]  # s2 has no tau

    def test_divergence_sensitive_bisim(self):
        """Divergent and convergent states with same label should be separated."""
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {TAU: [(0, 1.0)]},  # divergent
                1: {},  # convergent
                2: {TAU: [(2, 1.0)]},  # divergent
            },
            state_labels={0: {"x"}, 1: {"x"}, 2: {"x"}},
        )
        result = compute_divergence_sensitive_bisimulation(ts)
        # s0 and s2 are divergent, s1 is convergent -- s0,s2 in one block, s1 alone
        assert result.statistics["num_blocks"] == 2

    def test_divergence_sensitive_all_convergent(self):
        ts = two_state_tau()
        result = compute_divergence_sensitive_bisimulation(ts)
        # No divergence, so same as weak bisim
        assert result.statistics["num_blocks"] == 1


# ---- Tests: Conversion Utilities ----

class TestConversion:
    def test_lmc_to_prob_ts(self):
        matrix = [[0.0, 1.0], [0.5, 0.5]]
        labels = {0: {"a"}, 1: {"b"}}
        lmc = make_labeled_mc(matrix, labels)
        ts = lmc_to_prob_ts(lmc)
        assert ts.n_states == 2
        assert "step" in ts.actions[0]

    def test_prob_ts_to_lmc(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {"a": [(1, 1.0)]}, 1: {"a": [(0, 1.0)]}},
            state_labels={0: {"x"}, 1: {"y"}},
        )
        lmc = prob_ts_to_lmc(ts)
        assert lmc.mc.n_states == 2

    def test_roundtrip_labels(self):
        matrix = [[0.0, 1.0], [1.0, 0.0]]
        labels = {0: {"start"}, 1: {"end"}}
        lmc = make_labeled_mc(matrix, labels)
        ts = lmc_to_prob_ts(lmc)
        assert ts.state_labels[0] == {"start"}
        assert ts.state_labels[1] == {"end"}

    def test_deadlock_self_loop(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {"a": [(1, 1.0)]}, 1: {}},
            state_labels={0: set(), 1: set()},
        )
        lmc = prob_ts_to_lmc(ts)
        # State 1 has no actions -> should get self-loop
        assert abs(lmc.mc.transition[1][1] - 1.0) < 1e-9


# ---- Tests: Weak Bisimulation Distance ----

class TestDistance:
    def test_same_states_zero_distance(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={
                0: {"a": [(0, 1.0)]},
                1: {"a": [(1, 1.0)]},
            },
            state_labels={0: {"x"}, 1: {"x"}},
        )
        result = compute_weak_bisimulation_distance(ts)
        assert result.distances[0][1] < 0.01

    def test_different_labels_max_distance(self):
        ts = distinct_system()
        result = compute_weak_bisimulation_distance(ts)
        assert result.distances[0][1] == 1.0

    def test_bisimilar_pairs_found(self):
        ts = two_state_tau()
        result = compute_weak_bisimulation_distance(ts)
        assert len(result.bisimilar_pairs) >= 1

    def test_distance_symmetry(self):
        ts = diamond_system()
        result = compute_weak_bisimulation_distance(ts)
        for i in range(ts.n_states):
            for j in range(ts.n_states):
                assert abs(result.distances[i][j] - result.distances[j][i]) < 1e-9

    def test_distance_triangle_inequality(self):
        """d(s,u) <= d(s,t) + d(t,u) for all s,t,u."""
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {"a": [(1, 0.7), (2, 0.3)]},
                1: {"a": [(0, 0.5), (2, 0.5)]},
                2: {"a": [(2, 1.0)]},
            },
            state_labels={0: {"x"}, 1: {"x"}, 2: {"x"}},
        )
        result = compute_weak_bisimulation_distance(ts)
        d = result.distances
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    assert d[i][k] <= d[i][j] + d[j][k] + 1e-8


# ---- Tests: Compare Strong vs Weak ----

class TestComparison:
    def test_hierarchy_holds(self):
        """strong >= branching >= weak (in number of blocks)."""
        ts = diamond_system()
        comp = compare_strong_vs_weak(ts)
        assert comp["hierarchy"]["strong_leq_branching"]
        assert comp["hierarchy"]["branching_leq_weak"]

    def test_no_tau_all_equal(self):
        """Without tau transitions, all three bisimulations coincide."""
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={
                0: {"a": [(1, 1.0)]},
                1: {"a": [(0, 1.0)]},
            },
            state_labels={0: {"x"}, 1: {"x"}},
        )
        comp = compare_strong_vs_weak(ts)
        # All should be 1 block
        assert comp["strong"]["num_blocks"] == comp["weak"]["num_blocks"]

    def test_tau_makes_weak_coarser(self):
        """Tau transitions make weak bisim coarser than strong."""
        ts = simple_tau_chain()
        comp = compare_strong_vs_weak(ts)
        assert comp["weak"]["num_blocks"] <= comp["strong"]["num_blocks"]


# ---- Tests: Minimize ----

class TestMinimize:
    def test_minimize_weak(self):
        ts = diamond_system()
        quotient, result = minimize_weak(ts)
        assert quotient.n_states <= ts.n_states
        assert result.quotient is not None

    def test_minimize_branching(self):
        ts = diamond_system()
        quotient, result = minimize_branching(ts)
        assert quotient.n_states <= ts.n_states

    def test_minimize_idempotent(self):
        """Minimizing a minimal system gives the same system."""
        ts = distinct_system()
        q1, _ = minimize_weak(ts)
        # q1 should already be minimal
        q2, _ = minimize_weak(q1)
        assert q1.n_states == q2.n_states


# ---- Tests: Summary ----

class TestSummary:
    def test_summary_output(self):
        ts = diamond_system()
        summary = weak_bisimulation_summary(ts)
        assert "Weak Probabilistic Bisimulation Analysis" in summary
        assert "Block" in summary

    def test_summary_has_divergence_info(self):
        ts = make_labeled_prob_ts(
            n_states=2,
            actions={0: {TAU: [(0, 1.0)]}, 1: {}},
            state_labels={0: {"x"}, 1: {"x"}},
        )
        summary = weak_bisimulation_summary(ts)
        assert "Divergent" in summary


# ---- Tests: Edge Cases ----

class TestEdgeCases:
    def test_empty_action_state(self):
        ts = make_labeled_prob_ts(
            n_states=1,
            actions={0: {}},
            state_labels={0: set()},
        )
        result = compute_weak_bisimulation(ts)
        assert result.statistics["num_blocks"] == 1

    def test_multiple_observable_actions(self):
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {"a": [(1, 1.0)], "b": [(2, 1.0)]},
                1: {},
                2: {},
            },
            state_labels={0: {"x"}, 1: {"y"}, 2: {"z"}},
        )
        result = compute_weak_bisimulation(ts)
        assert result.statistics["num_blocks"] == 3

    def test_self_loop_observable(self):
        ts = make_labeled_prob_ts(
            n_states=1,
            actions={0: {"a": [(0, 1.0)]}},
            state_labels={0: {"x"}},
        )
        result = compute_weak_bisimulation(ts)
        assert result.statistics["num_blocks"] == 1

    def test_many_states_same_label(self):
        n = 10
        actions = {}
        for i in range(n - 1):
            actions[i] = {TAU: [(i + 1, 1.0)]}
        actions[n - 1] = {}
        ts = make_labeled_prob_ts(
            n_states=n,
            actions=actions,
            state_labels={i: {"x"} for i in range(n)},
        )
        result = compute_weak_bisimulation(ts)
        assert result.statistics["num_blocks"] == 1

    def test_actions_not_in_all_states(self):
        """Action 'a' available in s0 but not s1 -- should distinguish if same label."""
        ts = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {"a": [(2, 1.0)]},
                1: {},
                2: {},
            },
            state_labels={0: {"x"}, 1: {"x"}, 2: {"y"}},
        )
        result = compute_weak_bisimulation(ts)
        # s0 can do 'a', s1 cannot -> different weak behavior
        found_separate = True
        for block in result.partition:
            if 0 in block and 1 in block:
                found_separate = False
        assert found_separate


# ---- Tests: Complex Scenarios ----

class TestComplexScenarios:
    def test_buffer_system(self):
        """Model a 1-slot buffer: empty --put--> full --get--> empty, with tau internal."""
        ts = make_labeled_prob_ts(
            n_states=4,
            actions={
                0: {"put": [(1, 1.0)]},       # empty
                1: {TAU: [(2, 1.0)]},          # buffering (internal)
                2: {"get": [(3, 1.0)]},        # full
                3: {TAU: [(0, 1.0)]},          # draining (internal)
            },
            state_labels={0: {"empty"}, 1: {"mid"}, 2: {"full"}, 3: {"mid"}},
        )
        result = compute_weak_bisimulation(ts)
        # s1 and s3 have same label "mid" but different weak behavior
        # s1 can weakly do "get" (via tau to s2 then get), s3 can weakly do "put" (via tau to s0 then put)
        # So they should be in different blocks
        assert result.statistics["num_blocks"] >= 3

    def test_concurrent_coin_flips(self):
        """Two paths with same probabilistic outcome."""
        ts = make_labeled_prob_ts(
            n_states=5,
            actions={
                0: {TAU: [(1, 1.0)]},
                1: {"flip": [(3, 0.5), (4, 0.5)]},
                2: {"flip": [(3, 0.5), (4, 0.5)]},
                3: {},
                4: {},
            },
            state_labels={0: {"ready"}, 1: {"ready"}, 2: {"ready"}, 3: {"heads"}, 4: {"tails"}},
        )
        result = compute_weak_bisimulation(ts)
        # s0, s1, s2 all have label "ready" and same weak "flip" distribution
        found_block = False
        for block in result.partition:
            if 0 in block and 1 in block and 2 in block:
                found_block = True
        assert found_block

    def test_protocol_with_retransmit(self):
        """Send, internal ack, done. Vs. send, done directly. Weakly same if labels match."""
        ts1 = make_labeled_prob_ts(
            n_states=3,
            actions={
                0: {"send": [(1, 1.0)]},
                1: {TAU: [(2, 1.0)]},  # internal ack
                2: {},
            },
            state_labels={0: {"idle"}, 1: {"busy"}, 2: {"done"}},
        )
        ts2 = make_labeled_prob_ts(
            n_states=2,
            actions={
                0: {"send": [(1, 1.0)]},
                1: {},
            },
            state_labels={0: {"idle"}, 1: {"done"}},
        )
        result = check_cross_weak_bisimulation(ts1, ts2)
        # s0 (ts1) and s0 (ts2) both can "send" -> "done" (weakly)
        # But ts1 s0 goes through "busy" label, ts2 goes directly to "done"
        # Since "busy" != "done", the intermediate state in ts1 is different
        # Cross bisim checks partition: ts1.s0=idle, ts2.s0=idle -> same block?
        # ts1.s0 send -> s1(busy), ts2.s0 send -> s1(done)
        # Weak: ts1.s0 =send=> tau* reaches s2(done), ts2.s0 =send=> s1(done)
        # Both reach "done" weakly via send. s0s should match.
        assert result.statistics["cross_pairs"] >= 1

    def test_nondeterministic_tau_choices(self):
        """s0 has two tau-reachable observable behaviors."""
        ts = make_labeled_prob_ts(
            n_states=4,
            actions={
                0: {TAU: [(1, 0.5), (2, 0.5)]},
                1: {"a": [(3, 1.0)]},
                2: {"b": [(3, 1.0)]},
                3: {},
            },
            state_labels={0: {"start"}, 1: {"mid"}, 2: {"mid"}, 3: {"end"}},
        )
        result = compute_weak_bisimulation(ts)
        # s1 and s2 have same label but different observable actions
        # s1 can do "a", s2 can do "b" -- they are NOT weakly bisimilar
        for block in result.partition:
            assert not (1 in block and 2 in block), "s1 and s2 should be separate"


# ---- Tests: Partition Properties ----

class TestPartitionProperties:
    def test_partition_covers_all_states(self):
        ts = diamond_system()
        result = compute_weak_bisimulation(ts)
        all_states = set()
        for block in result.partition:
            all_states |= block
        assert all_states == set(range(ts.n_states))

    def test_partition_blocks_disjoint(self):
        ts = diamond_system()
        result = compute_weak_bisimulation(ts)
        for i, b1 in enumerate(result.partition):
            for j, b2 in enumerate(result.partition):
                if i != j:
                    assert b1.isdisjoint(b2)

    def test_same_labels_within_block(self):
        ts = diamond_system()
        result = compute_weak_bisimulation(ts)
        for block in result.partition:
            labels = [frozenset(ts.state_labels.get(s, set())) for s in block]
            assert len(set(labels)) == 1, f"Mixed labels in block: {labels}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
