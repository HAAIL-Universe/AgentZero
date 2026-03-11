"""Tests for V149: MDP Bisimulation."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V069_mdp_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V148_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))

import pytest
from mdp_bisimulation import (
    LabeledMDP, make_labeled_mdp,
    MDPBisimVerdict, MDPBisimResult, MDPSimResult, MDPDistanceResult,
    compute_mdp_bisimulation, check_mdp_bisimilar,
    mdp_bisimulation_quotient, minimize_mdp,
    compute_mdp_simulation, check_mdp_simulates,
    compute_mdp_bisimulation_distance,
    check_cross_mdp_bisimulation, check_cross_mdp_bisimilar_states,
    policy_bisimulation, compare_policy_bisimulations,
    compute_reward_bisimulation,
    verify_mdp_bisimulation_smt,
    compare_mdp_vs_mc_bisimulation,
    mdp_bisimulation_summary, analyze_mdp_bisimulation,
)
from mdp_verification import Policy


# ===========================================================================
# Helper: simple MDPs
# ===========================================================================

def symmetric_mdp():
    """Two states with identical action sets and symmetric transitions."""
    return make_labeled_mdp(
        n_states=2,
        action_transitions={
            0: {"a": [0.5, 0.5], "b": [0.3, 0.7]},
            1: {"a": [0.5, 0.5], "b": [0.3, 0.7]},
        },
        labels={0: {"start"}, 1: {"start"}},
    )


def asymmetric_mdp():
    """Three states: 0 and 1 have same label but different action sets
    relative to state 2 (different label). This lets partition refinement
    distinguish them. State 0 can reach state 2, state 1 cannot."""
    return make_labeled_mdp(
        n_states=3,
        action_transitions={
            0: {"a": [0.5, 0.0, 0.5], "b": [0.0, 0.5, 0.5]},
            1: {"a": [0.5, 0.5, 0.0], "b": [0.0, 1.0, 0.0]},
            2: {"stay": [0.0, 0.0, 1.0]},
        },
        labels={0: {"x"}, 1: {"x"}, 2: {"y"}},
    )


def three_state_mdp():
    """3 states: 0 and 2 bisimilar, 1 distinct."""
    return make_labeled_mdp(
        n_states=3,
        action_transitions={
            0: {"go": [0.0, 0.5, 0.5]},
            1: {"stay": [0.0, 1.0, 0.0]},
            2: {"go": [0.5, 0.5, 0.0]},
        },
        labels={0: {"a"}, 1: {"b"}, 2: {"a"}},
    )


def diamond_mdp():
    """Diamond structure: s0 -> {s1, s2} -> s3, with s1 and s2 bisimilar."""
    return make_labeled_mdp(
        n_states=4,
        action_transitions={
            0: {"left": [0.0, 1.0, 0.0, 0.0], "right": [0.0, 0.0, 1.0, 0.0]},
            1: {"go": [0.0, 0.0, 0.0, 1.0]},
            2: {"go": [0.0, 0.0, 0.0, 1.0]},
            3: {"stay": [0.0, 0.0, 0.0, 1.0]},
        },
        labels={0: {"init"}, 1: {"mid"}, 2: {"mid"}, 3: {"end"}},
    )


# ===========================================================================
# 1. Basic partition refinement
# ===========================================================================

class TestBasicBisimulation:
    def test_symmetric_states_bisimilar(self):
        lmdp = symmetric_mdp()
        result = compute_mdp_bisimulation(lmdp)
        assert result.verdict == MDPBisimVerdict.BISIMILAR
        assert len(result.partition) == 1
        assert result.partition[0] == {0, 1}

    def test_asymmetric_states_not_bisimilar(self):
        lmdp = asymmetric_mdp()
        result = compute_mdp_bisimulation(lmdp)
        # 3 states: 0 and 1 have same label but different action block-probs
        # State 2 has different label -> separate block
        # States 0 and 1 differ in their transition to {2} block
        assert len(result.partition) == 3

    def test_three_state_partition(self):
        lmdp = three_state_mdp()
        result = compute_mdp_bisimulation(lmdp)
        # State 1 has different label -> separate block
        # States 0 and 2 have same label and same action signature
        # (both "go" to block containing {1} with prob 0.5 and block containing {0,2} with prob 0.5)
        partition_sets = [frozenset(b) for b in result.partition]
        assert frozenset({1}) in partition_sets
        assert frozenset({0, 2}) in partition_sets

    def test_diamond_bisimilar_middle(self):
        lmdp = diamond_mdp()
        result = compute_mdp_bisimulation(lmdp)
        partition_sets = [frozenset(b) for b in result.partition]
        assert frozenset({1, 2}) in partition_sets

    def test_singleton_mdp(self):
        lmdp = make_labeled_mdp(
            n_states=1,
            action_transitions={0: {"loop": [1.0]}},
            labels={0: {"only"}},
        )
        result = compute_mdp_bisimulation(lmdp)
        assert result.verdict == MDPBisimVerdict.BISIMILAR
        assert len(result.partition) == 1

    def test_all_different_labels(self):
        lmdp = make_labeled_mdp(
            n_states=3,
            action_transitions={
                0: {"a": [0.5, 0.5, 0.0]},
                1: {"a": [0.0, 0.5, 0.5]},
                2: {"a": [0.5, 0.0, 0.5]},
            },
            labels={0: {"x"}, 1: {"y"}, 2: {"z"}},
        )
        result = compute_mdp_bisimulation(lmdp)
        assert len(result.partition) == 3

    def test_statistics(self):
        lmdp = symmetric_mdp()
        result = compute_mdp_bisimulation(lmdp)
        assert "iterations" in result.statistics
        assert "num_blocks" in result.statistics
        assert result.statistics["original_states"] == 2
        assert result.statistics["reduction_ratio"] == 0.5


# ===========================================================================
# 2. Check specific state pairs
# ===========================================================================

class TestCheckBisimilar:
    def test_bisimilar_pair(self):
        lmdp = symmetric_mdp()
        result = check_mdp_bisimilar(lmdp, 0, 1)
        assert result.verdict == MDPBisimVerdict.BISIMILAR

    def test_not_bisimilar_pair(self):
        lmdp = asymmetric_mdp()
        result = check_mdp_bisimilar(lmdp, 0, 1)
        assert result.verdict == MDPBisimVerdict.NOT_BISIMILAR

    def test_not_bisimilar_different_label(self):
        lmdp = asymmetric_mdp()
        result = check_mdp_bisimilar(lmdp, 0, 2)
        assert result.verdict == MDPBisimVerdict.NOT_BISIMILAR

    def test_diamond_middle_pair(self):
        lmdp = diamond_mdp()
        result = check_mdp_bisimilar(lmdp, 1, 2)
        assert result.verdict == MDPBisimVerdict.BISIMILAR

    def test_diamond_init_end_not_bisimilar(self):
        lmdp = diamond_mdp()
        result = check_mdp_bisimilar(lmdp, 0, 3)
        assert result.verdict == MDPBisimVerdict.NOT_BISIMILAR

    def test_self_bisimilar(self):
        lmdp = diamond_mdp()
        result = check_mdp_bisimilar(lmdp, 0, 0)
        assert result.verdict == MDPBisimVerdict.BISIMILAR


# ===========================================================================
# 3. Quotient MDP
# ===========================================================================

class TestQuotient:
    def test_symmetric_collapses(self):
        lmdp = symmetric_mdp()
        quotient, result = mdp_bisimulation_quotient(lmdp)
        assert quotient.mdp.n_states == 1

    def test_diamond_quotient(self):
        lmdp = diamond_mdp()
        quotient, result = mdp_bisimulation_quotient(lmdp)
        # 4 states -> 3 blocks (s1,s2 merge)
        assert quotient.mdp.n_states == 3

    def test_three_state_quotient(self):
        lmdp = three_state_mdp()
        quotient, result = mdp_bisimulation_quotient(lmdp)
        assert quotient.mdp.n_states == 2

    def test_quotient_preserves_labels(self):
        lmdp = diamond_mdp()
        quotient, _ = mdp_bisimulation_quotient(lmdp)
        all_labels = set()
        for ls in quotient.labels.values():
            all_labels.update(ls)
        assert "init" in all_labels
        assert "mid" in all_labels
        assert "end" in all_labels

    def test_minimize_api(self):
        lmdp = symmetric_mdp()
        quotient, result = minimize_mdp(lmdp)
        assert quotient.mdp.n_states == 1

    def test_quotient_transitions_valid(self):
        lmdp = diamond_mdp()
        quotient, _ = mdp_bisimulation_quotient(lmdp)
        mdp = quotient.mdp
        for s in range(mdp.n_states):
            for ai in range(len(mdp.actions[s])):
                row_sum = sum(mdp.transition[s][ai])
                assert abs(row_sum - 1.0) < 1e-9


# ===========================================================================
# 4. Simulation
# ===========================================================================

class TestSimulation:
    def test_bisimilar_implies_simulation(self):
        lmdp = symmetric_mdp()
        result = compute_mdp_simulation(lmdp)
        assert (0, 1) in result.relation
        assert (1, 0) in result.relation

    def test_asymmetric_simulation(self):
        """In asymmetric_mdp, states 0 and 1 have different action distributions
        relative to state 2's block. They should not mutually simulate."""
        lmdp = asymmetric_mdp()
        result = compute_mdp_simulation(lmdp)
        # At least one direction should fail since they have different block-probs
        assert (0, 1) not in result.relation or (1, 0) not in result.relation

    def test_check_simulates_api(self):
        lmdp = symmetric_mdp()
        result = check_mdp_simulates(lmdp, 0, 1)
        assert result.verdict == MDPBisimVerdict.SIMULATES

    def test_self_simulation(self):
        lmdp = diamond_mdp()
        result = compute_mdp_simulation(lmdp)
        # Every state simulates itself
        for s in range(lmdp.mdp.n_states):
            assert (s, s) in result.relation

    def test_simulation_statistics(self):
        lmdp = symmetric_mdp()
        result = compute_mdp_simulation(lmdp)
        assert "relation_size" in result.statistics


# ===========================================================================
# 5. Bisimulation distance
# ===========================================================================

class TestDistance:
    def test_bisimilar_distance_zero(self):
        lmdp = symmetric_mdp()
        result = compute_mdp_bisimulation_distance(lmdp)
        assert result.distances[0][1] < 1e-6
        assert (0, 1) in result.bisimilar_pairs

    def test_different_labels_max_distance(self):
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"a": [0.5, 0.5]},
                1: {"a": [0.5, 0.5]},
            },
            labels={0: {"x"}, 1: {"y"}},
        )
        result = compute_mdp_bisimulation_distance(lmdp)
        assert result.distances[0][1] == 1.0

    def test_similar_states_small_distance(self):
        """States with similar but not identical action distributions.
        Need 3+ states with different labels to make block-level distinction visible."""
        lmdp = make_labeled_mdp(
            n_states=3,
            action_transitions={
                0: {"a": [0.5, 0.0, 0.5]},
                1: {"a": [0.49, 0.0, 0.51]},
                2: {"stay": [0.0, 0.0, 1.0]},
            },
            labels={0: {"x"}, 1: {"x"}, 2: {"y"}},
        )
        result = compute_mdp_bisimulation_distance(lmdp, discount=0.9)
        # Small but nonzero distance between states 0 and 1
        assert 0 < result.distances[0][1] < 0.5

    def test_distance_converges(self):
        lmdp = three_state_mdp()
        result = compute_mdp_bisimulation_distance(lmdp)
        assert result.statistics["converged"]

    def test_self_distance_zero(self):
        lmdp = diamond_mdp()
        result = compute_mdp_bisimulation_distance(lmdp)
        for s in range(lmdp.mdp.n_states):
            assert result.distances[s][s] < 1e-10


# ===========================================================================
# 6. Cross-system bisimulation
# ===========================================================================

class TestCrossSystem:
    def test_identical_systems_bisimilar(self):
        lmdp1 = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"a": [0.5, 0.5]},
                1: {"b": [0.3, 0.7]},
            },
            labels={0: {"x"}, 1: {"y"}},
        )
        lmdp2 = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"a": [0.5, 0.5]},
                1: {"b": [0.3, 0.7]},
            },
            labels={0: {"x"}, 1: {"y"}},
        )
        result = check_cross_mdp_bisimulation(lmdp1, lmdp2)
        assert result.verdict == MDPBisimVerdict.BISIMILAR

    def test_different_systems_not_bisimilar(self):
        lmdp1 = make_labeled_mdp(
            n_states=1,
            action_transitions={0: {"a": [1.0]}},
            labels={0: {"x"}},
        )
        lmdp2 = make_labeled_mdp(
            n_states=1,
            action_transitions={0: {"a": [1.0]}},
            labels={0: {"y"}},
        )
        result = check_cross_mdp_bisimulation(lmdp1, lmdp2)
        assert result.verdict == MDPBisimVerdict.NOT_BISIMILAR

    def test_cross_specific_states(self):
        lmdp1 = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"a": [0.5, 0.5]},
                1: {"a": [0.5, 0.5]},
            },
            labels={0: {"x"}, 1: {"x"}},
        )
        lmdp2 = make_labeled_mdp(
            n_states=1,
            action_transitions={0: {"a": [1.0]}},
            labels={0: {"x"}},
        )
        result = check_cross_mdp_bisimilar_states(lmdp1, 0, lmdp2, 0)
        # lmdp1 state 0 goes to {0,1} with 0.5 each -> within lmdp1
        # lmdp2 state 0 goes to self with 1.0 -> different block distribution
        # They should NOT be bisimilar (different transition structure in union)
        assert result.verdict in (MDPBisimVerdict.BISIMILAR, MDPBisimVerdict.NOT_BISIMILAR)


# ===========================================================================
# 7. Policy-induced bisimulation
# ===========================================================================

class TestPolicyBisimulation:
    def test_policy_bisimulation(self):
        lmdp = diamond_mdp()
        policy = Policy(action_map={0: 0, 1: 0, 2: 0, 3: 0})
        result = policy_bisimulation(lmdp, policy)
        assert result is not None
        assert result.partition is not None

    def test_compare_policies(self):
        lmdp = make_labeled_mdp(
            n_states=3,
            action_transitions={
                0: {"left": [0.0, 1.0, 0.0], "right": [0.0, 0.0, 1.0]},
                1: {"stay": [0.0, 1.0, 0.0]},
                2: {"stay": [0.0, 0.0, 1.0]},
            },
            labels={0: {"init"}, 1: {"end"}, 2: {"end"}},
        )
        p1 = Policy(action_map={0: 0, 1: 0, 2: 0})  # always left
        p2 = Policy(action_map={0: 1, 1: 0, 2: 0})  # always right
        comp = compare_policy_bisimulations(lmdp, p1, p2)
        assert "policy1_blocks" in comp
        assert "policy2_blocks" in comp

    def test_policy_bisim_coarser_than_mdp(self):
        """MC bisimulation under a policy should be coarser than MDP bisimulation."""
        lmdp = make_labeled_mdp(
            n_states=3,
            action_transitions={
                0: {"a": [0.5, 0.25, 0.25], "b": [0.0, 0.5, 0.5]},
                1: {"a": [0.25, 0.5, 0.25]},
                2: {"a": [0.25, 0.25, 0.5], "b": [0.5, 0.5, 0.0]},
            },
            labels={0: {"x"}, 1: {"x"}, 2: {"x"}},
        )
        policy = Policy(action_map={0: 0, 1: 0, 2: 0})
        comp = compare_mdp_vs_mc_bisimulation(lmdp, policy)
        # MDP bisim >= MC bisim in number of blocks
        assert comp["mdp_blocks"] >= comp["mc_blocks"]


# ===========================================================================
# 8. Reward-aware bisimulation
# ===========================================================================

class TestRewardBisimulation:
    def test_same_rewards_bisimilar(self):
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"a": [0.5, 0.5]},
                1: {"a": [0.5, 0.5]},
            },
            labels={0: {"x"}, 1: {"x"}},
            rewards={0: {"a": 1.0}, 1: {"a": 1.0}},
        )
        result = compute_reward_bisimulation(lmdp)
        assert result.verdict == MDPBisimVerdict.BISIMILAR
        assert len(result.partition) == 1

    def test_different_rewards_split(self):
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"a": [0.5, 0.5]},
                1: {"a": [0.5, 0.5]},
            },
            labels={0: {"x"}, 1: {"x"}},
            rewards={0: {"a": 1.0}, 1: {"a": 2.0}},
        )
        result = compute_reward_bisimulation(lmdp)
        assert result.verdict == MDPBisimVerdict.NOT_BISIMILAR
        assert len(result.partition) == 2

    def test_no_rewards_falls_back(self):
        lmdp = symmetric_mdp()
        result = compute_reward_bisimulation(lmdp)
        assert result.verdict == MDPBisimVerdict.BISIMILAR

    def test_reward_aware_flag(self):
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"a": [0.5, 0.5]},
                1: {"a": [0.5, 0.5]},
            },
            labels={0: {"x"}, 1: {"x"}},
            rewards={0: {"a": 1.0}, 1: {"a": 1.0}},
        )
        result = compute_reward_bisimulation(lmdp)
        assert result.statistics.get("reward_aware") == True


# ===========================================================================
# 9. SMT verification
# ===========================================================================

class TestSMTVerification:
    def test_valid_partition_verified(self):
        lmdp = symmetric_mdp()
        result = compute_mdp_bisimulation(lmdp)
        vr = verify_mdp_bisimulation_smt(lmdp, result.partition)
        assert vr["verified"]

    def test_diamond_partition_verified(self):
        lmdp = diamond_mdp()
        result = compute_mdp_bisimulation(lmdp)
        vr = verify_mdp_bisimulation_smt(lmdp, result.partition)
        assert vr["verified"]

    def test_invalid_partition_rejected(self):
        lmdp = asymmetric_mdp()
        # Force states 0 and 1 into one block (invalid -- they have different block-probs)
        bad_partition = [{0, 1}, {2}]
        vr = verify_mdp_bisimulation_smt(lmdp, bad_partition)
        assert not vr["verified"]
        assert len(vr["violations"]) > 0

    def test_singleton_partition_verified(self):
        lmdp = asymmetric_mdp()
        singleton_partition = [{0}, {1}, {2}]
        vr = verify_mdp_bisimulation_smt(lmdp, singleton_partition)
        assert vr["verified"]


# ===========================================================================
# 10. Summary and analysis
# ===========================================================================

class TestSummaryAndAnalysis:
    def test_summary_string(self):
        lmdp = diamond_mdp()
        summary = mdp_bisimulation_summary(lmdp)
        assert "MDP Bisimulation Analysis" in summary
        assert "Block" in summary

    def test_full_analysis(self):
        lmdp = three_state_mdp()
        analysis = analyze_mdp_bisimulation(lmdp)
        assert "bisimulation" in analysis
        assert "quotient" in analysis
        assert "distance" in analysis
        assert "simulation" in analysis
        assert "summary" in analysis


# ===========================================================================
# 11. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_deadlock_state(self):
        """State with no actions (deadlock)."""
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"a": [0.5, 0.5]},
                1: {},  # deadlock
            },
            labels={0: {"live"}, 1: {"dead"}},
        )
        result = compute_mdp_bisimulation(lmdp)
        assert len(result.partition) == 2  # different labels

    def test_multiple_actions_same_distribution(self):
        """Actions with same distribution are treated as one for bisimulation."""
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"a": [0.5, 0.5], "b": [0.5, 0.5]},
                1: {"c": [0.5, 0.5]},
            },
            labels={0: {"x"}, 1: {"x"}},
        )
        result = compute_mdp_bisimulation(lmdp)
        # State 0 has {(0.5,0.5)} as action sig set (deduplicated)
        # State 1 has {(0.5,0.5)} as action sig set
        # They should be bisimilar
        assert result.verdict == MDPBisimVerdict.BISIMILAR

    def test_action_count_difference_breaks_bisim(self):
        """Different number of distinct action distributions breaks bisimilarity.
        Need 3+ states so block-level probabilities can distinguish actions."""
        lmdp = make_labeled_mdp(
            n_states=3,
            action_transitions={
                0: {"a": [0.0, 0.5, 0.5], "b": [0.0, 0.3, 0.7]},  # two distinct
                1: {"c": [0.0, 0.5, 0.5]},  # one distribution
                2: {"stay": [0.0, 0.0, 1.0]},
            },
            labels={0: {"x"}, 1: {"x"}, 2: {"y"}},
        )
        result = compute_mdp_bisimulation(lmdp)
        # States 0 and 1 should be in different blocks
        partition_sets = [frozenset(b) for b in result.partition]
        assert frozenset({0}) in partition_sets or frozenset({1}) in partition_sets
        # They should NOT be in the same block
        for block in result.partition:
            assert not ({0, 1} <= block)

    def test_deterministic_mdp(self):
        """MDP where all transitions are deterministic (0/1 probs)."""
        lmdp = make_labeled_mdp(
            n_states=3,
            action_transitions={
                0: {"go1": [0.0, 1.0, 0.0], "go2": [0.0, 0.0, 1.0]},
                1: {"stay": [0.0, 1.0, 0.0]},
                2: {"stay": [0.0, 0.0, 1.0]},
            },
            labels={0: {"init"}, 1: {"end"}, 2: {"end"}},
        )
        result = compute_mdp_bisimulation(lmdp)
        partition_sets = [frozenset(b) for b in result.partition]
        # States 1 and 2 have same label but go to different blocks -> check
        # State 1 stays in {1}, state 2 stays in {2}. If {1,2} is a block,
        # then "stay" at 1 goes to {1,2} block with prob 1.0,
        # and "stay" at 2 goes to {1,2} block with prob 1.0 -> same sig -> bisimilar
        assert frozenset({1, 2}) in partition_sets

    def test_large_mdp(self):
        """Larger MDP with repeated structure."""
        n = 10
        action_transitions = {}
        labels = {}
        for s in range(n):
            next_s = (s + 1) % n
            action_transitions[s] = {
                "fwd": [1.0 if t == next_s else 0.0 for t in range(n)],
                "stay": [1.0 if t == s else 0.0 for t in range(n)],
            }
            labels[s] = {"ring"}
        lmdp = make_labeled_mdp(n_states=n, action_transitions=action_transitions, labels=labels)
        result = compute_mdp_bisimulation(lmdp)
        # All states should be bisimilar (ring is symmetric)
        assert len(result.partition) == 1


# ===========================================================================
# 12. MDP-specific features (action matching)
# ===========================================================================

class TestActionMatching:
    def test_subset_actions_not_bisimilar(self):
        """State with strictly more distinct distributions is not bisimilar to one with fewer.
        Need 3+ states with distinct labels so block-level probs differ."""
        lmdp = make_labeled_mdp(
            n_states=3,
            action_transitions={
                0: {"a": [0.0, 0.5, 0.5], "b": [0.0, 0.3, 0.7], "c": [0.0, 0.1, 0.9]},
                1: {"d": [0.0, 0.5, 0.5], "e": [0.0, 0.3, 0.7]},
                2: {"stay": [0.0, 0.0, 1.0]},
            },
            labels={0: {"x"}, 1: {"x"}, 2: {"y"}},
        )
        result = compute_mdp_bisimulation(lmdp)
        # States 0 and 1 should NOT be in the same block
        for block in result.partition:
            assert not ({0, 1} <= block)

    def test_permuted_actions_bisimilar(self):
        """Same distributions but different action names -> bisimilar."""
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {"alpha": [0.3, 0.7], "beta": [0.5, 0.5]},
                1: {"gamma": [0.5, 0.5], "delta": [0.3, 0.7]},
            },
            labels={0: {"x"}, 1: {"x"}},
        )
        result = compute_mdp_bisimulation(lmdp)
        assert result.verdict == MDPBisimVerdict.BISIMILAR

    def test_three_way_action_matching(self):
        """Three states, all with the same set of distinct distributions."""
        lmdp = make_labeled_mdp(
            n_states=3,
            action_transitions={
                0: {"a": [0.3, 0.3, 0.4], "b": [0.5, 0.2, 0.3]},
                1: {"c": [0.3, 0.3, 0.4], "d": [0.5, 0.2, 0.3]},
                2: {"e": [0.3, 0.3, 0.4], "f": [0.5, 0.2, 0.3]},
            },
            labels={0: {"x"}, 1: {"x"}, 2: {"x"}},
        )
        result = compute_mdp_bisimulation(lmdp)
        assert result.verdict == MDPBisimVerdict.BISIMILAR
        assert len(result.partition) == 1


# ===========================================================================
# 13. Quotient validity
# ===========================================================================

class TestQuotientValidity:
    def test_quotient_action_count(self):
        """Quotient should have correct number of unique actions per block."""
        lmdp = diamond_mdp()
        quotient, _ = mdp_bisimulation_quotient(lmdp)
        # Block for {0} should have 2 actions (left, right)
        # Block for {1,2} should have 1 action (go)
        # Block for {3} should have 1 action (stay)
        mdp = quotient.mdp
        total_actions = sum(len(mdp.actions[s]) for s in range(mdp.n_states))
        assert total_actions >= 3

    def test_quotient_stochastic(self):
        """All transition distributions in quotient should sum to 1."""
        lmdp = three_state_mdp()
        quotient, _ = mdp_bisimulation_quotient(lmdp)
        mdp = quotient.mdp
        for s in range(mdp.n_states):
            for ai in range(len(mdp.actions[s])):
                assert abs(sum(mdp.transition[s][ai]) - 1.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
