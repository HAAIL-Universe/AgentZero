"""Tests for V153: Game-based Bisimulation."""

import sys
sys.path.insert(0, 'Z:/AgentZero/A2/work/V153_game_bisimulation')

import pytest
from game_bisimulation import (
    LTS, make_lts, build_bisimulation_game, check_bisimulation_game,
    check_weak_bisimulation_game, check_simulation_game,
    full_bisimulation_game, partition_bisimulation,
    compare_game_vs_partition, make_vending_machine_lts,
    make_bisimilar_pair, bisimulation_game_summary,
    NodeKind, GameNode, BisimGameResult, FullGameResult
)


# ============================================================
# Section 1: LTS Construction
# ============================================================

class TestLTSConstruction:
    def test_make_lts_basic(self):
        lts = make_lts(3, [(0, 'a', 1), (1, 'b', 2)])
        assert lts.n_states == 3
        assert lts.actions == {'a', 'b'}
        assert lts.successors(0, 'a') == {1}
        assert lts.successors(1, 'b') == {2}

    def test_make_lts_with_labels(self):
        lts = make_lts(2, [(0, 'a', 1)], labels={0: ['p'], 1: ['q']})
        assert lts.state_label(0) == frozenset({'p'})
        assert lts.state_label(1) == frozenset({'q'})

    def test_enabled_actions(self):
        lts = make_lts(3, [(0, 'a', 1), (0, 'b', 2)])
        assert lts.enabled_actions(0) == {'a', 'b'}
        assert lts.enabled_actions(1) == set()

    def test_nondeterministic_lts(self):
        lts = make_lts(3, [(0, 'a', 1), (0, 'a', 2)])
        assert lts.successors(0, 'a') == {1, 2}

    def test_empty_lts(self):
        lts = make_lts(1, [])
        assert lts.enabled_actions(0) == set()
        assert lts.successors(0, 'a') == set()

    def test_self_loop(self):
        lts = make_lts(1, [(0, 'a', 0)])
        assert lts.successors(0, 'a') == {0}


# ============================================================
# Section 2: Game Construction
# ============================================================

class TestGameConstruction:
    def test_game_has_nodes(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(2, [(0, 'a', 1)])
        pg, node_map, rev_map = build_bisimulation_game(lts1, lts2, check_labels=False)
        assert len(pg.nodes) > 0

    def test_attacker_nodes_exist(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(2, [(0, 'a', 1)])
        pg, node_map, _ = build_bisimulation_game(lts1, lts2, check_labels=False)
        attacker_nodes = [n for n in node_map.values() if n.kind == NodeKind.ATTACKER]
        assert len(attacker_nodes) > 0

    def test_defender_nodes_exist(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(2, [(0, 'a', 1)])
        pg, node_map, _ = build_bisimulation_game(lts1, lts2, check_labels=False)
        defender_nodes = [n for n in node_map.values() if n.kind == NodeKind.DEFENDER]
        assert len(defender_nodes) > 0

    def test_all_nodes_have_successors(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(2, [(0, 'a', 1)])
        pg, _, _ = build_bisimulation_game(lts1, lts2, check_labels=False)
        for nid in pg.nodes:
            assert nid in pg.successors and len(pg.successors[nid]) > 0, \
                f"Node {nid} has no successors"


# ============================================================
# Section 3: Identical LTS (Bisimilar)
# ============================================================

class TestIdenticalLTS:
    def test_identical_single_action(self):
        lts = make_lts(2, [(0, 'a', 1)])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_identical_two_actions(self):
        lts = make_lts(3, [(0, 'a', 1), (0, 'b', 2)])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_identical_cycle(self):
        lts = make_lts(2, [(0, 'a', 1), (1, 'b', 0)])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_self_bisimilar(self):
        lts = make_lts(3, [(0, 'a', 1), (0, 'a', 2), (1, 'b', 0), (2, 'b', 0)])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_single_state(self):
        lts = make_lts(1, [])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_single_state_self_loop(self):
        lts = make_lts(1, [(0, 'a', 0)])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.bisimilar


# ============================================================
# Section 4: Non-Bisimilar LTS
# ============================================================

class TestNonBisimilar:
    def test_different_actions(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(2, [(0, 'b', 1)])
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert not result.bisimilar

    def test_vending_machines(self):
        """Classic bisimulation counterexample: VM1 and VM2 are NOT bisimilar."""
        lts1, lts2 = make_vending_machine_lts()
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        # VM1: after coin, can do coffee OR tea (nondeterministic)
        # VM2: after coin, can do coffee (from state 1) or tea (from state 2)
        # Attacker picks coin in VM2 -> goes to state 1 (coffee only)
        # Defender must match in VM1 -> goes to state 1 (coffee AND tea)
        # Then attacker plays tea in VM1 state 1, defender can't match from VM2 state 1
        assert not result.bisimilar

    def test_deadlock_vs_action(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(1, [])
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert not result.bisimilar

    def test_different_branching(self):
        # LTS1: 0 -a-> 1 -b-> 2, 0 -a-> 3 -c-> 4
        lts1 = make_lts(5, [(0, 'a', 1), (1, 'b', 2), (0, 'a', 3), (3, 'c', 4)])
        # LTS2: 0 -a-> 1 -b-> 2 -c-> 3 (b and c on same path)
        lts2 = make_lts(4, [(0, 'a', 1), (1, 'b', 2), (1, 'c', 3)])
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert not result.bisimilar

    def test_label_mismatch(self):
        lts1 = make_lts(2, [(0, 'a', 1)], labels={0: ['p']})
        lts2 = make_lts(2, [(0, 'a', 1)], labels={0: ['q']})
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=True)
        assert not result.bisimilar


# ============================================================
# Section 5: Bisimilar but Structurally Different
# ============================================================

class TestBisimilarDifferentStructure:
    def test_different_state_count(self):
        """LTS2 has redundant states but same behavior."""
        lts1, lts2 = make_bisimilar_pair()
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_unfolded_cycle(self):
        # LTS1: 0 -a-> 0 (self-loop)
        lts1 = make_lts(1, [(0, 'a', 0)])
        # LTS2: 0 -a-> 1 -a-> 0 (2-cycle, same observable behavior)
        lts2 = make_lts(2, [(0, 'a', 1), (1, 'a', 0)])
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_merged_branches(self):
        # LTS1: two branches that do the same thing
        lts1 = make_lts(4, [(0, 'a', 1), (0, 'a', 2), (1, 'b', 3), (2, 'b', 3)])
        # LTS2: single branch
        lts2 = make_lts(3, [(0, 'a', 1), (1, 'b', 2)])
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_symmetric_nondeterminism(self):
        # Both have same nondeterministic behavior from state 0
        lts1 = make_lts(3, [(0, 'a', 1), (0, 'a', 2), (1, 'b', 0), (2, 'b', 0)])
        lts2 = make_lts(4, [
            (0, 'a', 1), (0, 'a', 2), (0, 'a', 3),
            (1, 'b', 0), (2, 'b', 0), (3, 'b', 0)
        ])
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.bisimilar


# ============================================================
# Section 6: Game Result Details
# ============================================================

class TestGameResults:
    def test_game_size_nonzero(self):
        lts = make_lts(2, [(0, 'a', 1)])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.game_size > 0

    def test_attacker_wins_set(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(2, [(0, 'b', 1)])
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert (0, 0) in result.attacker_wins

    def test_defender_wins_set(self):
        lts = make_lts(2, [(0, 'a', 1)])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert (0, 0) in result.defender_wins

    def test_distinguishing_play_exists(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(2, [(0, 'b', 1)])
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.distinguishing_play is not None
        assert len(result.distinguishing_play) > 0

    def test_no_distinguishing_play_when_bisimilar(self):
        lts = make_lts(2, [(0, 'a', 1)])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.distinguishing_play is None


# ============================================================
# Section 7: Full Game Analysis
# ============================================================

class TestFullGameAnalysis:
    def test_full_game_bisimilar(self):
        lts = make_lts(2, [(0, 'a', 1), (1, 'b', 0)])
        result = full_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.bisimilar
        assert result.game is not None
        assert result.parity_result is not None

    def test_full_game_not_bisimilar(self):
        lts1, lts2 = make_vending_machine_lts()
        result = full_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert not result.bisimilar
        assert len(result.attacker_pairs) > 0

    def test_full_game_distinguishing_sequence(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(2, [(0, 'b', 1)])
        result = full_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert not result.bisimilar
        assert result.distinguishing_sequence is not None

    def test_full_game_node_map(self):
        lts = make_lts(2, [(0, 'a', 1)])
        result = full_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.node_map is not None
        assert len(result.node_map) > 0


# ============================================================
# Section 8: Weak Bisimulation Game
# ============================================================

class TestWeakBisimulationGame:
    def test_weak_bisimilar_with_tau(self):
        """States differing only by tau prefix are weakly bisimilar."""
        lts = make_lts(3, [
            (0, 'a', 2),
            (1, 'tau', 0),
            (1, 'a', 2),  # also direct a for matching
        ])
        # State 0 and 1: 1 can do tau then a, 0 can do a directly
        # But for weak bisim, 1's tau is invisible
        result = check_weak_bisimulation_game(lts, 0, 1)
        # 0: does a -> 2. 1: does tau -> 0 -> a -> 2. Weakly bisimilar.
        assert result.bisimilar

    def test_weak_not_bisimilar(self):
        """Weak bisimulation still distinguishes observable differences."""
        lts = make_lts(3, [
            (0, 'a', 1),
            (1, 'b', 0),
            (2, 'a', 1),
            (2, 'c', 0),  # different: c instead of b after a
        ])
        # State 0: a->1, then b possible. State 2: a->1 (same), but also c->0
        # Not weakly bisimilar because state 2 enables c but state 0 doesn't
        result = check_weak_bisimulation_game(lts, 0, 2)
        assert not result.bisimilar

    def test_weak_tau_loop_bisimilar(self):
        """Tau loops are invisible in weak bisimulation."""
        lts = make_lts(3, [
            (0, 'a', 2),
            (1, 'tau', 1),  # tau self-loop
            (1, 'a', 2),
        ])
        result = check_weak_bisimulation_game(lts, 0, 1)
        assert result.bisimilar

    def test_weak_game_size(self):
        lts = make_lts(3, [(0, 'a', 1), (1, 'tau', 2), (2, 'b', 0)])
        result = check_weak_bisimulation_game(lts, 0, 0)
        assert result.game_size > 0


# ============================================================
# Section 9: Simulation Game
# ============================================================

class TestSimulationGame:
    def test_simulation_self(self):
        lts = make_lts(2, [(0, 'a', 1)])
        result = check_simulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.bisimilar  # self-simulation always holds

    def test_simulation_more_behaviors(self):
        """LTS2 has strictly more behaviors -- LTS1 is simulated by LTS2."""
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(3, [(0, 'a', 1), (0, 'b', 2)])
        result = check_simulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.bisimilar  # lts2 simulates lts1

    def test_simulation_not_reverse(self):
        """LTS2 can simulate LTS1 but LTS1 cannot simulate LTS2."""
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(3, [(0, 'a', 1), (0, 'b', 2)])
        result = check_simulation_game(lts2, lts1, 0, 0, check_labels=False)
        assert not result.bisimilar  # lts1 CANNOT simulate lts2

    def test_simulation_different_actions(self):
        lts1 = make_lts(2, [(0, 'a', 1)])
        lts2 = make_lts(2, [(0, 'b', 1)])
        result = check_simulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert not result.bisimilar

    def test_simulation_nondeterminism(self):
        """Nondeterministic LTS simulated by deterministic one."""
        lts1 = make_lts(3, [(0, 'a', 1), (0, 'a', 2)])
        lts2 = make_lts(3, [(0, 'a', 1), (0, 'a', 2)])
        result = check_simulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_simulation_label_check(self):
        lts1 = make_lts(2, [(0, 'a', 1)], labels={0: ['p']})
        lts2 = make_lts(2, [(0, 'a', 1)], labels={0: ['q']})
        result = check_simulation_game(lts1, lts2, 0, 0, check_labels=True)
        assert not result.bisimilar


# ============================================================
# Section 10: Partition Refinement Comparison
# ============================================================

class TestPartitionRefinement:
    def test_partition_single_block(self):
        lts = make_lts(2, [(0, 'a', 1), (1, 'a', 0)])
        # Both states have same behavior: a -> other
        partition = partition_bisimulation(lts)
        assert len(partition) == 1  # All in one block

    def test_partition_different_blocks(self):
        lts = make_lts(2, [(0, 'a', 1)])
        partition = partition_bisimulation(lts)
        assert len(partition) == 2  # state 0 has 'a', state 1 doesn't

    def test_partition_three_states(self):
        lts = make_lts(3, [
            (0, 'a', 1), (0, 'a', 2),
            (1, 'b', 0),
            (2, 'b', 0)
        ])
        partition = partition_bisimulation(lts)
        # States 1 and 2 are bisimilar (both do b->0)
        found = False
        for block in partition:
            if 1 in block and 2 in block:
                found = True
        assert found

    def test_compare_game_vs_partition(self):
        lts = make_lts(3, [
            (0, 'a', 1), (0, 'a', 2),
            (1, 'b', 0), (2, 'b', 0)
        ])
        result = compare_game_vs_partition(lts, lts, 1, 2)
        assert result['game_bisimilar'] == result['partition_bisimilar']
        assert result['agree'] == True

    def test_compare_not_bisimilar(self):
        lts = make_lts(2, [(0, 'a', 1)])
        result = compare_game_vs_partition(lts, lts, 0, 1)
        assert result['game_bisimilar'] == False
        assert result['partition_bisimilar'] == False
        assert result['agree'] == True


# ============================================================
# Section 11: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_deadlocked_states_bisimilar(self):
        """Two deadlocked states with same labels are bisimilar."""
        lts1 = make_lts(1, [])
        lts2 = make_lts(1, [])
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_deadlocked_different_labels(self):
        lts1 = make_lts(1, [], labels={0: ['p']})
        lts2 = make_lts(1, [], labels={0: ['q']})
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=True)
        assert not result.bisimilar

    def test_large_nondeterminism(self):
        """Many nondeterministic successors."""
        n = 10
        trans1 = [(0, 'a', i) for i in range(1, n+1)]
        trans2 = [(0, 'a', i) for i in range(1, n+1)]
        lts1 = make_lts(n+1, trans1)
        lts2 = make_lts(n+1, trans2)
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_chain_bisimilar(self):
        """Two identical chains are bisimilar."""
        lts = make_lts(4, [(0, 'a', 1), (1, 'b', 2), (2, 'c', 3)])
        result = check_bisimulation_game(lts, lts, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_chain_suffix_not_bisimilar(self):
        """Different positions in a chain are not bisimilar."""
        lts = make_lts(4, [(0, 'a', 1), (1, 'b', 2), (2, 'c', 3)])
        result = check_bisimulation_game(lts, lts, 0, 1, check_labels=False)
        assert not result.bisimilar


# ============================================================
# Section 12: Summary API
# ============================================================

class TestSummaryAPI:
    def test_summary_bisimilar(self):
        lts = make_lts(2, [(0, 'a', 1)])
        summary = bisimulation_game_summary(lts, lts, 0, 0)
        assert summary['bisimilar'] == True
        assert summary['game_positions'] > 0

    def test_summary_not_bisimilar(self):
        lts1, lts2 = make_vending_machine_lts()
        summary = bisimulation_game_summary(lts1, lts2, 0, 0)
        assert summary['bisimilar'] == False
        assert summary['attacker_winning_pairs'] > 0

    def test_summary_fields(self):
        lts = make_lts(2, [(0, 'a', 1)])
        summary = bisimulation_game_summary(lts, lts, 0, 0)
        assert 'bisimilar' in summary
        assert 'game_positions' in summary
        assert 'attacker_winning_pairs' in summary
        assert 'defender_winning_pairs' in summary


# ============================================================
# Section 13: Classic Examples
# ============================================================

class TestClassicExamples:
    def test_milner_scheduler(self):
        """Simple scheduler-like example."""
        # Process 1: can do a then b, repeatedly
        lts1 = make_lts(2, [(0, 'a', 1), (1, 'b', 0)])
        # Process 2: same behavior, different structure
        lts2 = make_lts(3, [(0, 'a', 1), (1, 'b', 2), (2, 'a', 1)])
        # Not bisimilar: lts2 state 2 does 'a' but not 'b'; lts1 state 0 does 'a' too
        # After a->b->a: lts1 back at 0 (can do a), lts2 at 1 (can do b, not a yet)
        # Actually: lts1: 0-a->1-b->0-a->1...  lts2: 0-a->1-b->2-a->1...
        # lts1 state 0 ~ lts2 state 0: both do a.
        # lts1 state 1 ~ lts2 state 1: both do b.
        # lts2 state 2: does a (same as lts1 state 0).
        # So bisimilar! lts2 state 2 ~ lts1 state 0
        result = check_bisimulation_game(lts1, lts2, 0, 0, check_labels=False)
        assert result.bisimilar

    def test_buffer_one_vs_two(self):
        """1-place buffer vs 2-place buffer (classic non-bisimilar example)."""
        # 1-place: empty -put-> full -get-> empty
        buf1 = make_lts(2, [(0, 'put', 1), (1, 'get', 0)])
        # 2-place: empty -put-> half -put-> full, half -get-> empty, full -get-> half
        buf2 = make_lts(3, [
            (0, 'put', 1), (1, 'put', 2), (1, 'get', 0), (2, 'get', 1)
        ])
        # Not bisimilar: buf2 state 1 can do both put and get
        # buf1 after put: state 1 can only get
        result = check_bisimulation_game(buf1, buf2, 0, 0, check_labels=False)
        assert not result.bisimilar

    def test_deterministic_vs_nondeterministic(self):
        """Deterministic process bisimilar to nondeterministic one."""
        # det: 0 -a-> 1 -b-> 0
        det = make_lts(2, [(0, 'a', 1), (1, 'b', 0)])
        # nondet: 0 -a-> 1, 0 -a-> 2, 1 -b-> 0, 2 -b-> 0
        nondet = make_lts(3, [(0, 'a', 1), (0, 'a', 2), (1, 'b', 0), (2, 'b', 0)])
        result = check_bisimulation_game(det, nondet, 0, 0, check_labels=False)
        assert result.bisimilar


# ============================================================
# Section 14: Game-Partition Agreement
# ============================================================

class TestGamePartitionAgreement:
    def test_agreement_on_cycle(self):
        lts = make_lts(3, [(0, 'a', 1), (1, 'a', 2), (2, 'a', 0)])
        for s1 in range(3):
            for s2 in range(3):
                cmp = compare_game_vs_partition(lts, lts, s1, s2)
                assert cmp['agree'], f"Disagreement on ({s1}, {s2})"

    def test_agreement_on_tree(self):
        lts = make_lts(7, [
            (0, 'a', 1), (0, 'b', 2),
            (1, 'c', 3), (1, 'd', 4),
            (2, 'c', 5), (2, 'd', 6),
        ])
        for s1 in range(7):
            for s2 in range(7):
                cmp = compare_game_vs_partition(lts, lts, s1, s2)
                assert cmp['agree'], f"Disagreement on ({s1}, {s2})"

    def test_agreement_nondeterministic(self):
        lts = make_lts(4, [
            (0, 'a', 1), (0, 'a', 2),
            (1, 'b', 3), (2, 'c', 3)
        ])
        cmp = compare_game_vs_partition(lts, lts, 1, 2)
        assert cmp['agree']
        assert cmp['game_bisimilar'] == False  # 1 does b, 2 does c


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
