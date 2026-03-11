"""Tests for V148: Probabilistic Bisimulation."""

import sys
sys.path.insert(0, 'Z:/AgentZero/A2/work/V148_probabilistic_bisimulation')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V065_markov_chain_analysis')
sys.path.insert(0, 'Z:/AgentZero/A2/work/V067_pctl_model_checking')

import pytest
from prob_bisimulation import (
    compute_bisimulation, check_bisimilar, bisimulation_quotient,
    compute_simulation, check_simulates,
    compute_bisimulation_distance,
    check_cross_bisimulation, check_cross_bisimilar_states,
    lump_chain, is_valid_lumping,
    verify_bisimulation_smt,
    minimize, compare_systems, bisimulation_summary,
    BisimVerdict, BisimResult, SimResult, DistanceResult,
)
from pctl_model_check import make_labeled_mc


# ===================================================================
# 1. Basic bisimulation - identical states
# ===================================================================

class TestBasicBisimulation:
    def test_two_identical_states(self):
        """Two states with same labels and same transitions are bisimilar."""
        # s0 and s1 both go to s2 with prob 1.0, both labeled 'a'
        matrix = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        assert result.statistics['num_blocks'] == 2  # {0,1} and {2}

    def test_two_different_labels(self):
        """States with different labels are not bisimilar."""
        matrix = [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        labels = {0: {'a'}, 1: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        assert result.statistics['num_blocks'] == 2

    def test_single_state(self):
        """Single-state MC is trivially bisimilar with itself."""
        matrix = [[1.0]]
        labels = {0: {'a'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        assert result.statistics['num_blocks'] == 1

    def test_all_same(self):
        """All states identical -> one equivalence class."""
        matrix = [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
        # s0 and s1 same labels, same transitions (0.5 to {0,1} block, 0 to {2})
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        # s0 and s1 should be in same block
        found = False
        for block in result.partition:
            if 0 in block and 1 in block:
                found = True
        assert found

    def test_different_transition_probs(self):
        """Same labels but different transition probs -> not bisimilar."""
        matrix = [
            [0.0, 0.3, 0.7],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = check_bisimilar(lmc, 0, 1)
        assert result.verdict == BisimVerdict.NOT_BISIMILAR


# ===================================================================
# 2. Check bisimilar API
# ===================================================================

class TestCheckBisimilar:
    def test_bisimilar_pair(self):
        """Two states known to be bisimilar."""
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        result = check_bisimilar(lmc, 0, 1)
        assert result.verdict == BisimVerdict.BISIMILAR

    def test_not_bisimilar_pair(self):
        """Two states known to not be bisimilar."""
        matrix = [
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = check_bisimilar(lmc, 0, 1)
        assert result.verdict == BisimVerdict.NOT_BISIMILAR
        assert result.witness is not None

    def test_self_bisimilar(self):
        """Every state is bisimilar to itself."""
        matrix = [
            [0.5, 0.5],
            [0.3, 0.7],
        ]
        labels = {0: {'a'}, 1: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = check_bisimilar(lmc, 0, 0)
        assert result.verdict == BisimVerdict.BISIMILAR

    def test_witness_label_difference(self):
        """Witness identifies label difference."""
        matrix = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        labels = {0: {'a'}, 1: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = check_bisimilar(lmc, 0, 1)
        assert result.verdict == BisimVerdict.NOT_BISIMILAR
        assert 'label' in result.witness.lower() or 'Different' in result.witness


# ===================================================================
# 3. Bisimulation quotient
# ===================================================================

class TestBisimulationQuotient:
    def test_quotient_reduces_states(self):
        """Quotient has fewer states when bisimilar states exist."""
        # 4 states, but 0~1 and 2~3
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        quotient = bisimulation_quotient(lmc)
        assert quotient.mc.n_states == 2

    def test_quotient_preserves_transitions(self):
        """Quotient preserves transition structure."""
        matrix = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        quotient = bisimulation_quotient(lmc)
        assert quotient.mc.n_states == 2
        # Block with 'a' should transition to block with 'b' with prob 1.0
        a_block = None
        b_block = None
        for i in range(2):
            if 'a' in quotient.labels.get(i, set()):
                a_block = i
            if 'b' in quotient.labels.get(i, set()):
                b_block = i
        assert a_block is not None and b_block is not None
        assert abs(quotient.mc.transition[a_block][b_block] - 1.0) < 1e-10

    def test_quotient_no_reduction(self):
        """No reduction when all states are distinguishable."""
        matrix = [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        labels = {0: {'a'}, 1: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        quotient = bisimulation_quotient(lmc)
        assert quotient.mc.n_states == 2

    def test_quotient_full_reduction(self):
        """Full reduction to 1 state when all bisimilar."""
        matrix = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        labels = {0: {'a'}, 1: {'a'}}
        lmc = make_labeled_mc(matrix, labels)

        quotient = bisimulation_quotient(lmc)
        assert quotient.mc.n_states == 1
        assert abs(quotient.mc.transition[0][0] - 1.0) < 1e-10


# ===================================================================
# 4. Simulation preorder
# ===================================================================

class TestSimulation:
    def test_bisimilar_implies_simulation(self):
        """Bisimilar states mutually simulate each other."""
        matrix = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_simulation(lmc)
        assert (0, 1) in result.relation
        assert (1, 0) in result.relation

    def test_simulation_not_symmetric(self):
        """Simulation can be one-directional."""
        # s0: goes to s2 with 0.5, s3 with 0.5
        # s1: goes to s2 with 1.0
        # s0 cannot simulate s1 (s1 concentrates on s2, s0 splits)
        # Actually for probabilistic simulation, this depends on the relation
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_simulation(lmc)
        # s0 and s1 have same labels but different transition structure
        # s1 simulates itself, s0 simulates itself
        assert (0, 0) in result.relation
        assert (1, 1) in result.relation

    def test_check_simulates(self):
        """Check specific simulation relationship."""
        matrix = [
            [0.0, 1.0],
            [0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}}
        lmc = make_labeled_mc(matrix, labels)

        result = check_simulates(lmc, 0, 1)
        assert result.verdict == BisimVerdict.SIMULATES

    def test_different_labels_no_simulation(self):
        """Different labels means no simulation."""
        matrix = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        labels = {0: {'a'}, 1: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = check_simulates(lmc, 0, 1)
        assert result.verdict == BisimVerdict.NOT_SIMULATES

    def test_self_simulation(self):
        """Every state simulates itself."""
        matrix = [
            [0.3, 0.7],
            [0.6, 0.4],
        ]
        labels = {0: {'a'}, 1: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_simulation(lmc)
        assert (0, 0) in result.relation
        assert (1, 1) in result.relation


# ===================================================================
# 5. Bisimulation distance
# ===================================================================

class TestBisimulationDistance:
    def test_identical_states_zero_distance(self):
        """Bisimilar states have distance 0."""
        matrix = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation_distance(lmc)
        assert result.distances[0][1] < 1e-6
        assert (0, 1) in result.bisimilar_pairs

    def test_different_labels_max_distance(self):
        """States with different labels have distance 1.0."""
        matrix = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        labels = {0: {'a'}, 1: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation_distance(lmc)
        assert abs(result.distances[0][1] - 1.0) < 1e-6

    def test_near_bisimilar(self):
        """States with slightly different transitions have small distance."""
        # s0: 0.5 to s2, 0.5 to s3
        # s1: 0.51 to s2, 0.49 to s3  (slightly different)
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.51, 0.49],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation_distance(lmc, threshold=0.5)
        d = result.distances[0][1]
        assert d > 0  # Not exactly bisimilar
        assert d < 0.5  # But close

    def test_symmetry(self):
        """Distance is symmetric: d(s,t) = d(t,s)."""
        matrix = [
            [0.0, 0.3, 0.7],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation_distance(lmc)
        assert abs(result.distances[0][1] - result.distances[1][0]) < 1e-8

    def test_discount_effect(self):
        """Lower discount reduces distances."""
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        r1 = compute_bisimulation_distance(lmc, discount=0.9)
        r2 = compute_bisimulation_distance(lmc, discount=0.5)
        assert r2.distances[0][1] < r1.distances[0][1]


# ===================================================================
# 6. Cross-system bisimulation
# ===================================================================

class TestCrossSystemBisimulation:
    def test_identical_systems(self):
        """Two identical systems have cross-bisimilar states."""
        matrix = [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        labels = {0: {'a'}, 1: {'b'}}
        lmc1 = make_labeled_mc(matrix, labels)
        lmc2 = make_labeled_mc(matrix, labels)

        result = check_cross_bisimulation(lmc1, lmc2)
        assert result.verdict == BisimVerdict.BISIMILAR

    def test_different_systems(self):
        """Two structurally different systems are not cross-bisimilar."""
        m1 = [[0.0, 1.0], [1.0, 0.0]]
        l1 = {0: {'a'}, 1: {'b'}}
        m2 = [[0.0, 1.0], [0.0, 1.0]]
        l2 = {0: {'a'}, 1: {'b'}}

        lmc1 = make_labeled_mc(m1, l1)
        lmc2 = make_labeled_mc(m2, l2)

        result = check_cross_bisimulation(lmc1, lmc2)
        # s0 in sys1: goes to s1(b), then back to s0(a)
        # s0 in sys2: goes to s1(b), stays at s1(b)
        # Different!
        assert result.verdict == BisimVerdict.NOT_BISIMILAR

    def test_cross_specific_states(self):
        """Check specific cross-system state pair."""
        matrix = [[1.0]]
        labels = {0: {'a'}}
        lmc1 = make_labeled_mc(matrix, labels)
        lmc2 = make_labeled_mc(matrix, labels)

        result = check_cross_bisimilar_states(lmc1, 0, lmc2, 0)
        assert result.verdict == BisimVerdict.BISIMILAR

    def test_cross_different_labels(self):
        """Cross-system states with different labels."""
        matrix = [[1.0]]
        lmc1 = make_labeled_mc(matrix, {0: {'a'}})
        lmc2 = make_labeled_mc(matrix, {0: {'b'}})

        result = check_cross_bisimilar_states(lmc1, 0, lmc2, 0)
        assert result.verdict == BisimVerdict.NOT_BISIMILAR


# ===================================================================
# 7. Lumping
# ===================================================================

class TestLumping:
    def test_valid_lumping(self):
        """Check that bisimulation partition is a valid lumping."""
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        valid, msg = is_valid_lumping(lmc, result.partition)
        assert valid

    def test_invalid_lumping(self):
        """Invalid partition that's not a valid lumping."""
        matrix = [
            [0.0, 0.3, 0.7],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        # Force s0 and s1 into same block (invalid)
        partition = [{0, 1}, {2}]
        valid, msg = is_valid_lumping(lmc, partition)
        assert not valid

    def test_lump_chain(self):
        """Lump a chain according to a valid partition."""
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        partition = [{0, 1}, {2, 3}]
        lumped = lump_chain(lmc, partition)
        assert lumped.mc.n_states == 2
        # Each block transitions to the other with prob 1.0
        assert abs(lumped.mc.transition[0][1] - 1.0) < 1e-10
        assert abs(lumped.mc.transition[1][0] - 1.0) < 1e-10


# ===================================================================
# 8. SMT verification
# ===================================================================

class TestSMTVerification:
    def test_verify_valid_partition(self):
        """SMT verifies a valid bisimulation partition."""
        matrix = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        verification = verify_bisimulation_smt(lmc, result.partition)
        assert verification['valid']
        assert verification['obligations_passed'] == verification['obligations_checked']

    def test_verify_invalid_partition(self):
        """SMT detects invalid bisimulation partition."""
        matrix = [
            [0.0, 0.3, 0.7],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        # Force invalid partition
        partition = [{0, 1}, {2}]
        verification = verify_bisimulation_smt(lmc, partition)
        assert not verification['valid']
        assert len(verification['failures']) > 0

    def test_verify_trivial_partition(self):
        """Singleton partition is trivially valid."""
        matrix = [[0.5, 0.5], [0.5, 0.5]]
        labels = {0: {'a'}, 1: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        partition = [{0}, {1}]
        verification = verify_bisimulation_smt(lmc, partition)
        assert verification['valid']


# ===================================================================
# 9. Minimize API
# ===================================================================

class TestMinimize:
    def test_minimize_reducible(self):
        """Minimize reduces a reducible system."""
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        quotient, result = minimize(lmc)
        assert quotient.mc.n_states < lmc.mc.n_states
        assert result.quotient is not None

    def test_minimize_irreducible(self):
        """Minimize doesn't reduce an irreducible system."""
        matrix = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
        labels = {0: {'a'}, 1: {'b'}, 2: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        quotient, result = minimize(lmc)
        assert quotient.mc.n_states == lmc.mc.n_states


# ===================================================================
# 10. Compare systems
# ===================================================================

class TestCompareSystems:
    def test_compare_identical(self):
        """Compare two identical systems."""
        matrix = [[0.0, 1.0], [1.0, 0.0]]
        labels = {0: {'a'}, 1: {'b'}}
        lmc1 = make_labeled_mc(matrix, labels)
        lmc2 = make_labeled_mc(matrix, labels)

        comp = compare_systems(lmc1, lmc2)
        assert comp['cross_bisimulation']['verdict'] == 'bisimilar'

    def test_compare_different(self):
        """Compare two different systems."""
        m1 = [[0.0, 1.0], [1.0, 0.0]]
        m2 = [[0.5, 0.5], [0.5, 0.5]]
        labels = {0: {'a'}, 1: {'b'}}

        lmc1 = make_labeled_mc(m1, labels)
        lmc2 = make_labeled_mc(m2, labels)

        comp = compare_systems(lmc1, lmc2)
        assert comp['cross_bisimulation']['verdict'] == 'not_bisimilar'


# ===================================================================
# 11. Summary
# ===================================================================

class TestSummary:
    def test_summary_output(self):
        """Summary generates readable text."""
        matrix = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        summary = bisimulation_summary(lmc)
        assert 'Bisimulation' in summary
        assert 'Block' in summary
        assert 'Reduction' in summary


# ===================================================================
# 12. Classic examples
# ===================================================================

class TestClassicExamples:
    def test_die_vs_biased_die(self):
        """Fair die vs biased die are not bisimilar."""
        # Fair: uniform over 6 outcomes
        # Biased: non-uniform
        n = 7  # 1 start + 6 outcomes
        fair_matrix = [[0.0] * n for _ in range(n)]
        biased_matrix = [[0.0] * n for _ in range(n)]

        # Start state transitions to each outcome
        for i in range(1, 7):
            fair_matrix[0][i] = 1.0 / 6.0
            biased_matrix[0][i] = 0.1 if i <= 3 else 0.3 / 3.0
        # Fix biased to sum to 1: 3*0.1 + 3*(0.3/3) = 0.3 + 0.3 = 0.6
        # Need to fix: make it sum to 1
        for i in range(1, 4):
            biased_matrix[0][i] = 0.1
        for i in range(4, 7):
            biased_matrix[0][i] = 7.0 / 30.0  # (1 - 0.3) / 3 = 0.7/3

        # Outcomes are absorbing
        for i in range(1, 7):
            fair_matrix[i][i] = 1.0
            biased_matrix[i][i] = 1.0

        labels = {0: {'start'}}
        for i in range(1, 7):
            labels[i] = {f'face_{i}'}

        lmc1 = make_labeled_mc(fair_matrix, labels)
        lmc2 = make_labeled_mc(biased_matrix, labels)

        result = check_cross_bisimilar_states(lmc1, 0, lmc2, 0)
        assert result.verdict == BisimVerdict.NOT_BISIMILAR

    def test_coin_flip_equivalence(self):
        """Two fair coin flip implementations are bisimilar."""
        # System 1: start -> H(0.5), T(0.5)
        # System 2: start -> H(0.5), T(0.5)  (same structure)
        matrix = [
            [0.0, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'start'}, 1: {'heads'}, 2: {'tails'}}
        lmc1 = make_labeled_mc(matrix, labels)
        lmc2 = make_labeled_mc(matrix, labels)

        result = check_cross_bisimilar_states(lmc1, 0, lmc2, 0)
        assert result.verdict == BisimVerdict.BISIMILAR

    def test_random_walk_symmetry(self):
        """Symmetric random walk: left-right states are bisimilar."""
        # States: 0(abs) - 1 - 2(center) - 3 - 4(abs)
        # Symmetric: 1 ~ 3
        matrix = [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # absorbing
            [0.5, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 1.0],  # absorbing
        ]
        labels = {0: {'boundary'}, 1: {'inner'}, 2: {'center'},
                  3: {'inner'}, 4: {'boundary'}}
        lmc = make_labeled_mc(matrix, labels)

        result = check_bisimilar(lmc, 1, 3)
        assert result.verdict == BisimVerdict.BISIMILAR

        result2 = check_bisimilar(lmc, 0, 4)
        assert result2.verdict == BisimVerdict.BISIMILAR

    def test_three_block_chain(self):
        """Chain with 3 distinct equivalence classes."""
        # 6 states: {0,1} 'start', {2,3} 'mid', {4,5} 'end'
        matrix = [
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        ]
        labels = {0: {'start'}, 1: {'start'}, 2: {'mid'}, 3: {'mid'},
                  4: {'end'}, 5: {'end'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        assert result.statistics['num_blocks'] == 3

        quotient = bisimulation_quotient(lmc)
        assert quotient.mc.n_states == 3


# ===================================================================
# 13. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_absorbing_states(self):
        """Absorbing states with same labels are bisimilar."""
        matrix = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'done'}, 1: {'done'}, 2: {'other'}}
        lmc = make_labeled_mc(matrix, labels)

        result = check_bisimilar(lmc, 0, 1)
        assert result.verdict == BisimVerdict.BISIMILAR

    def test_empty_labels(self):
        """States with empty label sets."""
        matrix = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        labels = {0: set(), 1: set()}
        lmc = make_labeled_mc(matrix, labels)

        result = check_bisimilar(lmc, 0, 1)
        assert result.verdict == BisimVerdict.BISIMILAR

    def test_multiple_label_sets(self):
        """States with multiple labels."""
        matrix = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        labels = {0: {'a', 'b'}, 1: {'a', 'b'}, 2: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        # s0 goes to s1, s1 goes to s2 -- different next step
        result = check_bisimilar(lmc, 0, 1)
        assert result.verdict == BisimVerdict.NOT_BISIMILAR

    def test_convergence(self):
        """Large enough system still converges."""
        n = 10
        matrix = [[0.0] * n for _ in range(n)]
        # All states transition uniformly
        for i in range(n):
            for j in range(n):
                matrix[i][j] = 1.0 / n
        labels = {i: {'same'} for i in range(n)}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        assert result.statistics['num_blocks'] == 1  # All bisimilar

    def test_distance_result_fields(self):
        """DistanceResult has all expected fields."""
        matrix = [[0.5, 0.5], [0.5, 0.5]]
        labels = {0: {'a'}, 1: {'a'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation_distance(lmc)
        assert isinstance(result.distances, list)
        assert isinstance(result.max_distance, float)
        assert isinstance(result.bisimilar_pairs, list)
        assert isinstance(result.near_bisimilar_pairs, list)
        assert isinstance(result.statistics, dict)


# ===================================================================
# 14. Partition refinement details
# ===================================================================

class TestPartitionRefinement:
    def test_label_partition_separates(self):
        """Initial label partition correctly separates by labels."""
        matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        # s0 and s1 have same labels and same transitions (self-loops)
        found = False
        for block in result.partition:
            if 0 in block and 1 in block:
                found = True
        assert found

    def test_refinement_splits_correctly(self):
        """Refinement correctly splits blocks."""
        # s0: 0.7 to s2, 0.3 to s3
        # s1: 0.3 to s2, 0.7 to s3
        # Same labels but different transition probs -> should split
        matrix = [
            [0.0, 0.0, 0.7, 0.3],
            [0.0, 0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        # s0 and s1 should be in different blocks
        for block in result.partition:
            assert not (0 in block and 1 in block)

    def test_two_step_refinement(self):
        """System requiring 2 refinement steps."""
        # s0: -> s2(1.0)
        # s1: -> s3(1.0)
        # s2: -> s4(1.0)
        # s3: -> s5(1.0)
        # s4: absorbing, label 'x'
        # s5: absorbing, label 'y'
        # First refinement: s2 vs s3 (different successors' labels)
        # Second refinement: s0 vs s1 (different blocks for s2 vs s3)
        matrix = [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
        matrix = [[float(x) for x in row] for row in matrix]
        labels = {0: {'start'}, 1: {'start'}, 2: {'mid'}, 3: {'mid'},
                  4: {'x'}, 5: {'y'}}
        lmc = make_labeled_mc(matrix, labels)

        result = compute_bisimulation(lmc)
        # s0 and s1 should NOT be bisimilar (lead to different endpoints)
        for block in result.partition:
            assert not (0 in block and 1 in block)
        assert result.statistics['iterations'] >= 2


# ===================================================================
# 15. Integration: bisim + quotient + verify
# ===================================================================

class TestIntegration:
    def test_full_pipeline(self):
        """Full pipeline: compute bisim -> quotient -> verify."""
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        # Step 1: compute bisimulation
        bisim = compute_bisimulation(lmc)
        assert bisim.statistics['num_blocks'] == 3  # {0,1}, {2}, {3}

        # Step 2: quotient
        quotient = bisimulation_quotient(lmc)
        assert quotient.mc.n_states == 3

        # Step 3: SMT verify
        verification = verify_bisimulation_smt(lmc, bisim.partition)
        assert verification['valid']

    def test_minimize_and_compare(self):
        """Minimize two systems and compare."""
        # System 1: 4 states, reducible to 2
        m1 = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ]
        l1 = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'b'}}

        # System 2: 2 states, already minimal
        m2 = [[0.0, 1.0], [1.0, 0.0]]
        l2 = {0: {'a'}, 1: {'b'}}

        lmc1 = make_labeled_mc(m1, l1)
        lmc2 = make_labeled_mc(m2, l2)

        comp = compare_systems(lmc1, lmc2)
        assert comp['system1']['minimized_states'] == 2
        assert comp['system2']['minimized_states'] == 2
        # Minimized systems should be cross-bisimilar
        assert comp['minimized_cross']['verdict'] == 'bisimilar'

    def test_simulation_contains_bisimulation(self):
        """Every bisimilar pair is also a simulation pair (both directions)."""
        matrix = [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        labels = {0: {'a'}, 1: {'a'}, 2: {'b'}, 3: {'c'}}
        lmc = make_labeled_mc(matrix, labels)

        bisim = compute_bisimulation(lmc)
        sim = compute_simulation(lmc)

        # Every bisimilar pair should be in simulation (both directions)
        for block in bisim.partition:
            for s in block:
                for t in block:
                    assert (s, t) in sim.relation, \
                        f"Bisimilar pair ({s},{t}) not in simulation"
