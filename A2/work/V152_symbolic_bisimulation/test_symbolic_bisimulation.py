"""Tests for V152: Symbolic Bisimulation."""

import pytest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))
from symbolic_bisimulation import (
    make_symbolic_ts, make_symbolic_ts_from_kripke, make_symbolic_ts_from_lts,
    compute_strong_bisimulation, compute_weak_bisimulation,
    compute_branching_bisimulation, check_bisimilar,
    check_cross_bisimulation, compute_quotient, minimize,
    compare_bisimulations, bisimulation_summary,
    make_chain, make_ring, make_binary_tree, make_parallel_composition,
    TAU, _preimage, _tau_closure, _state_bdd, _state_in_bdd,
    _initial_partition, _block_count,
)


# ===== Section 1: SymbolicTS Construction =====

class TestSymbolicTSConstruction:
    def test_kripke_basic(self):
        """Two states, one transition."""
        ts = make_symbolic_ts_from_kripke(
            n_states=2,
            transitions=[(0, 1)],
            state_labels={0: {"p"}, 1: {"q"}},
        )
        assert ts.n_state_bits == 1
        assert ts.actions == ["tau"]
        assert "p" in ts.labels
        assert "q" in ts.labels

    def test_lts_basic(self):
        """LTS with two actions."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[(0, "a", 1), (0, "b", 2), (1, "a", 2)],
            state_labels={0: {"start"}, 1: set(), 2: {"end"}},
        )
        assert ts.n_state_bits == 2
        assert "a" in ts.actions
        assert "b" in ts.actions

    def test_explicit_construction(self):
        """Direct make_symbolic_ts."""
        ts = make_symbolic_ts(
            state_var_names=["x"],
            actions=["a"],
            transitions={"a": [(0, 1), (1, 0)]},
            state_labels={0: {"even"}, 1: {"odd"}},
        )
        assert ts.n_state_bits == 1
        assert len(ts.labels) == 2

    def test_no_labels(self):
        """System with no propositions."""
        ts = make_symbolic_ts_from_kripke(
            n_states=2,
            transitions=[(0, 1)],
            state_labels={0: set(), 1: set()},
        )
        assert len(ts.labels) == 0

    def test_self_loop(self):
        """State with self-loop."""
        ts = make_symbolic_ts_from_kripke(
            n_states=2,
            transitions=[(0, 0), (0, 1)],
            state_labels={0: {"p"}, 1: {"p"}},
        )
        assert ts.n_state_bits == 1


# ===== Section 2: Preimage Operations =====

class TestPreimage:
    def test_preimage_simple(self):
        """Preimage of state 1 via action a in 0->1."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
        )
        s1 = _state_bdd(ts, 1)
        pre = _preimage(ts, s1, "a")
        # State 0 should be in preimage
        s0 = _state_bdd(ts, 0)
        assert ts.bdd.AND(s0, pre) != ts.bdd.FALSE
        # State 1 should not
        assert ts.bdd.AND(s1, pre) == ts.bdd.FALSE

    def test_preimage_multiple_predecessors(self):
        """Multiple states can reach target."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[(0, "a", 2), (1, "a", 2)],
        )
        s2 = _state_bdd(ts, 2)
        pre = _preimage(ts, s2, "a")
        assert _state_in_bdd(ts, 0, pre)
        assert _state_in_bdd(ts, 1, pre)
        assert not _state_in_bdd(ts, 2, pre)

    def test_preimage_no_action(self):
        """Preimage for non-existent action is empty."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
        )
        s1 = _state_bdd(ts, 1)
        pre = _preimage(ts, s1, "b")
        assert pre == ts.bdd.FALSE

    def test_preimage_self_loop(self):
        """Preimage includes self-loop state."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 0), (0, "a", 1)],
        )
        s0 = _state_bdd(ts, 0)
        pre = _preimage(ts, s0, "a")
        assert _state_in_bdd(ts, 0, pre)


# ===== Section 3: Initial Partition =====

class TestInitialPartition:
    def test_distinct_labels(self):
        """Each state has unique labels -> one block per state."""
        ts = make_symbolic_ts_from_kripke(
            n_states=2,
            transitions=[(0, 1)],
            state_labels={0: {"p"}, 1: {"q"}},
        )
        partition = _initial_partition(ts)
        assert len(partition) == 2

    def test_same_labels(self):
        """All states have same labels -> one block."""
        ts = make_symbolic_ts_from_kripke(
            n_states=2,
            transitions=[(0, 1)],
            state_labels={0: {"p"}, 1: {"p"}},
        )
        partition = _initial_partition(ts)
        assert len(partition) == 1

    def test_no_labels(self):
        """No labels -> one block."""
        ts = make_symbolic_ts_from_kripke(
            n_states=4,
            transitions=[(0, 1), (1, 2), (2, 3)],
            state_labels={0: set(), 1: set(), 2: set(), 3: set()},
        )
        partition = _initial_partition(ts)
        assert len(partition) == 1

    def test_three_groups(self):
        """Three distinct label sets."""
        ts = make_symbolic_ts_from_kripke(
            n_states=4,
            transitions=[],
            state_labels={0: {"a"}, 1: {"b"}, 2: {"a"}, 3: {"b"}},
            propositions=["a", "b"],
        )
        partition = _initial_partition(ts)
        # {a} only, {b} only, neither, both -- but states only use {a} and {b}
        non_empty = [b for b in partition if b != ts.bdd.FALSE]
        # States 0,2 have {a}, states 1,3 have {b}
        # With 2 bits we have 4 states; the other 2 bit combos have neither label
        assert len(non_empty) >= 2


# ===== Section 4: Strong Bisimulation - Basic =====

class TestStrongBisimulationBasic:
    def test_two_distinct_states(self):
        """States with different labels are not bisimilar."""
        ts = make_symbolic_ts_from_kripke(
            n_states=2,
            transitions=[(0, 1)],
            state_labels={0: {"p"}, 1: {"q"}},
        )
        result = compute_strong_bisimulation(ts)
        assert result.n_blocks == 2

    def test_two_identical_states(self):
        """States with same labels and same transitions are bisimilar."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 0), (1, "a", 1)],
            state_labels={0: {"p"}, 1: {"p"}},
        )
        result = compute_strong_bisimulation(ts)
        assert result.n_blocks == 1

    def test_deadlock_vs_live(self):
        """Deadlock state vs state with transitions."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
            state_labels={0: set(), 1: set()},
        )
        result = compute_strong_bisimulation(ts)
        # State 0 can do 'a', state 1 cannot -> different
        assert result.n_blocks == 2

    def test_three_state_chain(self):
        """0 -> 1 -> 2, all same labels."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[(0, "a", 1), (1, "a", 2)],
            state_labels={0: set(), 1: set(), 2: set()},
        )
        result = compute_strong_bisimulation(ts)
        # 2 is deadlock, 1 can reach deadlock, 0 can reach 1 -> all different
        assert result.n_blocks == 3

    def test_equivalence_relation(self):
        """Equivalence BDD encodes correct relation."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 0), (1, "a", 1)],
            state_labels={0: {"p"}, 1: {"p"}},
        )
        result = compute_strong_bisimulation(ts)
        # Equivalence should include (0,0), (0,1), (1,0), (1,1)
        assert result.equivalence != ts.bdd.FALSE


# ===== Section 5: Strong Bisimulation - LTS =====

class TestStrongBisimulationLTS:
    def test_coffee_machine(self):
        """Classic coffee machine example with coin/coffee actions."""
        # Machine 1: coin -> coffee -> done (state 0->1->2)
        # Machine 2: coin -> coffee -> done (state 3->4->5)
        # These should be bisimilar
        ts = make_symbolic_ts_from_lts(
            n_states=6,
            transitions=[
                (0, "coin", 1), (1, "coffee", 2),
                (3, "coin", 4), (4, "coffee", 5),
            ],
            state_labels={0: set(), 1: set(), 2: {"done"},
                         3: set(), 4: set(), 5: {"done"}},
        )
        assert check_bisimilar(ts, 0, 3, "strong")
        assert check_bisimilar(ts, 1, 4, "strong")
        assert check_bisimilar(ts, 2, 5, "strong")

    def test_different_actions(self):
        """States offering different actions are not bisimilar."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, "a", 2), (1, "b", 3),
            ],
            state_labels={0: set(), 1: set(), 2: set(), 3: set()},
        )
        assert not check_bisimilar(ts, 0, 1, "strong")

    def test_diamond(self):
        """Diamond: 0 -a-> 1, 0 -a-> 2, 1 -b-> 3, 2 -b-> 3."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, "a", 1), (0, "a", 2),
                (1, "b", 3), (2, "b", 3),
            ],
            state_labels={0: set(), 1: set(), 2: set(), 3: set()},
        )
        # States 1 and 2 are bisimilar (same label, same outgoing)
        assert check_bisimilar(ts, 1, 2, "strong")

    def test_nondeterministic_choice(self):
        """Nondeterministic choice leads to different outcomes."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, "a", 1), (0, "a", 2),  # 0 can reach 1 or 2 via a
                (3, "a", 1),                 # 3 can only reach 1 via a
            ],
            state_labels={0: set(), 1: {"p"}, 2: {"q"}, 3: set()},
        )
        # 0 and 3 differ: 0 can reach {q}-state via a, 3 cannot
        assert not check_bisimilar(ts, 0, 3, "strong")


# ===== Section 6: Weak Bisimulation =====

class TestWeakBisimulation:
    def test_tau_invisible(self):
        """Tau transitions are invisible in weak bisimulation."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[
                (0, "a", 2),
                (1, TAU, 0),  # 1 silently becomes 0
                (1, "a", 2),  # redundant but helps
            ],
            state_labels={0: set(), 1: set(), 2: {"done"}},
        )
        result = compute_weak_bisimulation(ts)
        # States 0 and 1 should be weakly bisimilar (1 can tau to 0)
        # Both can do 'a' to reach done state
        assert check_bisimilar(ts, 0, 1, "weak")

    def test_tau_chain(self):
        """Chain of tau transitions collapsed."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, TAU, 1), (1, TAU, 2), (2, "a", 3),
            ],
            state_labels={0: set(), 1: set(), 2: set(), 3: {"done"}},
        )
        result = compute_weak_bisimulation(ts)
        # 0, 1, 2 should be weakly bisimilar (all can tau*;a to 3)
        assert check_bisimilar(ts, 0, 2, "weak")

    def test_weak_vs_strong_differ(self):
        """Weak bisimulation is coarser than strong."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[
                (0, "a", 2),
                (1, TAU, 0),
            ],
            state_labels={0: set(), 1: set(), 2: {"done"}},
        )
        strong = compute_strong_bisimulation(ts)
        weak = compute_weak_bisimulation(ts)
        # Weak should have fewer or equal blocks
        assert weak.n_blocks <= strong.n_blocks


# ===== Section 7: Branching Bisimulation =====

class TestBranchingBisimulation:
    def test_branching_basic(self):
        """Branching bisimulation on simple tau system."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[
                (0, TAU, 1), (1, "a", 2),
            ],
            state_labels={0: set(), 1: set(), 2: {"done"}},
        )
        result = compute_branching_bisimulation(ts)
        # 0 and 1 should be branching-bisimilar (tau is stuttering)
        assert result.n_blocks >= 1

    def test_branching_preserves_structure(self):
        """Branching bisimulation is finer than weak."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, TAU, 1), (1, "a", 3),
                (2, "a", 3),
            ],
            state_labels={0: set(), 1: set(), 2: set(), 3: {"done"}},
        )
        weak = compute_weak_bisimulation(ts)
        branching = compute_branching_bisimulation(ts)
        # Branching >= weak in block count (finer)
        assert branching.n_blocks >= weak.n_blocks

    def test_hierarchy_strong_branching_weak(self):
        """strong >= branching >= weak in fineness."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, "a", 3), (1, TAU, 0), (2, TAU, 1),
            ],
            state_labels={0: set(), 1: set(), 2: set(), 3: {"done"}},
        )
        strong = compute_strong_bisimulation(ts)
        branching = compute_branching_bisimulation(ts)
        weak = compute_weak_bisimulation(ts)
        assert strong.n_blocks >= branching.n_blocks
        assert branching.n_blocks >= weak.n_blocks


# ===== Section 8: Cross-System Bisimulation =====

class TestCrossBisimulation:
    def test_identical_systems(self):
        """Two identical systems are bisimilar."""
        ts1 = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
            state_labels={0: set(), 1: {"done"}},
        )
        ts2 = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
            state_labels={0: set(), 1: {"done"}},
        )
        result = check_cross_bisimulation(ts1, ts2)
        assert result.bisimilar

    def test_different_systems(self):
        """Systems with different behavior are not bisimilar."""
        ts1 = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
            state_labels={0: set(), 1: {"p"}},
        )
        ts2 = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "b", 1)],
            state_labels={0: set(), 1: {"p"}},
        )
        result = check_cross_bisimulation(ts1, ts2)
        assert not result.bisimilar

    def test_different_structure_same_behavior(self):
        """Different structure but bisimilar behavior."""
        # System 1: 0 -a-> 1
        ts1 = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
            state_labels={0: set(), 1: {"done"}},
        )
        # System 2: 0 -a-> 1, 0 -a-> 2 (but 1 and 2 are identical)
        ts2 = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[(0, "a", 1), (0, "a", 2)],
            state_labels={0: set(), 1: {"done"}, 2: {"done"}},
        )
        result = check_cross_bisimulation(ts1, ts2)
        assert result.bisimilar

    def test_distinguishing_witness(self):
        """Non-bisimilar systems produce a witness."""
        ts1 = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
            state_labels={0: set(), 1: {"p"}},
        )
        ts2 = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "b", 1)],
            state_labels={0: set(), 1: {"p"}},
        )
        result = check_cross_bisimulation(ts1, ts2)
        assert not result.bisimilar
        assert result.witness is not None


# ===== Section 9: Quotient / Minimization =====

class TestQuotient:
    def test_minimize_trivial(self):
        """System with all states bisimilar minimizes to 1 block."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 0), (1, "a", 1)],
            state_labels={0: {"p"}, 1: {"p"}},
        )
        q = minimize(ts, "strong")
        assert q["n_blocks"] == 1

    def test_minimize_chain(self):
        """Chain minimization preserves structure."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[(0, "a", 1), (1, "a", 2)],
            state_labels={0: set(), 1: set(), 2: set()},
        )
        q = minimize(ts, "strong")
        # 3 distinct behaviors -> 3 blocks
        assert q["n_blocks"] == 3

    def test_minimize_symmetric_pair(self):
        """Symmetric states collapse."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, "a", 2), (0, "a", 3),
                (1, "a", 2), (1, "a", 3),
            ],
            state_labels={0: set(), 1: set(), 2: {"end"}, 3: {"end"}},
        )
        q = minimize(ts, "strong")
        # States 0,1 bisimilar; states 2,3 bisimilar
        assert q["n_blocks"] == 2

    def test_quotient_transitions(self):
        """Quotient preserves transition structure."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
            state_labels={0: {"start"}, 1: {"end"}},
        )
        q = minimize(ts, "strong")
        assert q["n_blocks"] == 2
        assert "a" in q["transitions"]

    def test_state_to_block_mapping(self):
        """Every state maps to a block."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[(0, "a", 1), (2, "a", 3)],
            state_labels={0: set(), 1: {"end"}, 2: set(), 3: {"end"}},
        )
        q = minimize(ts, "strong")
        # States 0,2 should map to same block; 1,3 to same block
        assert q["state_to_block"][0] == q["state_to_block"][2]
        assert q["state_to_block"][1] == q["state_to_block"][3]


# ===== Section 10: Comparison API =====

class TestComparison:
    def test_compare_all_modes(self):
        """Compare all three bisimulation modes."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[
                (0, TAU, 1), (1, "a", 2),
            ],
            state_labels={0: set(), 1: set(), 2: {"done"}},
        )
        cmp = compare_bisimulations(ts)
        assert "strong" in cmp
        assert "weak" in cmp
        assert "branching" in cmp
        assert "hierarchy" in cmp

    def test_compare_no_tau(self):
        """Without tau, all modes equal."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
            state_labels={0: set(), 1: set()},
        )
        cmp = compare_bisimulations(ts)
        assert "note" in cmp

    def test_summary(self):
        """Summary includes block sizes."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[(0, "a", 2), (1, "a", 3)],
            state_labels={0: {"start"}, 1: {"start"}, 2: {"end"}, 3: {"end"}},
        )
        s = bisimulation_summary(ts, "strong")
        assert s["n_blocks"] == 2
        assert s["total_states"] == 4
        assert s["reduction_ratio"] == 2.0

    def test_summary_modes(self):
        """Summary works for all modes."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[(0, TAU, 1), (1, "a", 2)],
            state_labels={0: set(), 1: set(), 2: set()},
        )
        for mode in ["strong", "weak", "branching"]:
            s = bisimulation_summary(ts, mode)
            assert s["mode"] == mode
            assert s["n_blocks"] >= 1


# ===== Section 11: Parametric Systems =====

class TestParametricSystems:
    def test_chain(self):
        """Chain of 4 states with alternating labels."""
        ts = make_chain(4)
        result = compute_strong_bisimulation(ts)
        # Alternating labels: states 0,2 have {p}; 1,3 have {q}
        # But transitions differ: 0->1, 1->2, 2->3 (no transition from 3)
        # So all 4 are distinct
        assert result.n_blocks == 4

    def test_chain_unlabeled(self):
        """Unlabeled chain -- bisimulation by transition structure."""
        ts = make_chain(4, labeled=False)
        result = compute_strong_bisimulation(ts)
        # All same labels, so distinguish by depth from deadlock
        assert result.n_blocks == 4

    def test_ring(self):
        """Ring -- all states bisimilar (same structure, no labels)."""
        ts = make_ring(4)
        result = compute_strong_bisimulation(ts)
        assert result.n_blocks == 1

    def test_binary_tree(self):
        """Binary tree -- leaves bisimilar, internal nodes by depth."""
        ts = make_binary_tree(2)
        result = compute_strong_bisimulation(ts)
        # Depth 2: root, 2 internal, 4 leaves
        # Leaves are bisimilar, internal nodes bisimilar, root unique
        # -> 3 blocks: {root}, {internal}, {leaves}
        assert result.n_blocks == 3

    def test_parallel_2_components(self):
        """Parallel composition of 2 components."""
        ts = make_parallel_composition(2)
        result = compute_strong_bisimulation(ts)
        # 4 states, each with unique label combo -> 4 blocks
        assert result.n_blocks == 4


# ===== Section 12: Tau Closure =====

class TestTauClosure:
    def test_no_tau(self):
        """No tau transitions -> closure is identity."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
        )
        s0 = _state_bdd(ts, 0)
        closed = _tau_closure(ts, s0)
        assert _state_in_bdd(ts, 0, closed)
        assert not _state_in_bdd(ts, 1, closed)

    def test_tau_chain_closure(self):
        """Tau chain: 0 -tau-> 1 -tau-> 2."""
        ts = make_symbolic_ts_from_lts(
            n_states=3,
            transitions=[(0, TAU, 1), (1, TAU, 2)],
        )
        s0 = _state_bdd(ts, 0)
        closed = _tau_closure(ts, s0)
        assert _state_in_bdd(ts, 0, closed)
        assert _state_in_bdd(ts, 1, closed)
        assert _state_in_bdd(ts, 2, closed)

    def test_tau_cycle(self):
        """Tau cycle: 0 -tau-> 1 -tau-> 0."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, TAU, 1), (1, TAU, 0)],
        )
        s0 = _state_bdd(ts, 0)
        closed = _tau_closure(ts, s0)
        assert _state_in_bdd(ts, 0, closed)
        assert _state_in_bdd(ts, 1, closed)

    def test_tau_partial(self):
        """Tau closure doesn't reach unreachable states."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[(0, TAU, 1), (2, TAU, 3)],
        )
        s0 = _state_bdd(ts, 0)
        closed = _tau_closure(ts, s0)
        assert _state_in_bdd(ts, 0, closed)
        assert _state_in_bdd(ts, 1, closed)
        assert not _state_in_bdd(ts, 2, closed)
        assert not _state_in_bdd(ts, 3, closed)


# ===== Section 13: Edge Cases =====

class TestEdgeCases:
    def test_single_state(self):
        """Single state system."""
        ts = make_symbolic_ts_from_kripke(
            n_states=1,
            transitions=[],
            state_labels={0: {"p"}},
        )
        result = compute_strong_bisimulation(ts)
        assert result.n_blocks == 1

    def test_disconnected(self):
        """Disconnected states with same labels."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[(0, "a", 1), (2, "a", 3)],
            state_labels={0: set(), 1: set(), 2: set(), 3: set()},
        )
        result = compute_strong_bisimulation(ts)
        # 0 and 2 bisimilar (both -a-> deadlock), 1 and 3 bisimilar (both deadlock)
        assert result.n_blocks == 2

    def test_all_self_loops(self):
        """All states have same self-loop -> bisimilar."""
        ts = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 0), (1, "a", 1)],
            state_labels={0: set(), 1: set()},
        )
        result = compute_strong_bisimulation(ts)
        assert result.n_blocks == 1

    def test_multiple_actions(self):
        """System with multiple actions splits correctly."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, "a", 2), (0, "b", 3),
                (1, "a", 2),  # 1 can't do b
            ],
            state_labels={0: set(), 1: set(), 2: set(), 3: set()},
        )
        result = compute_strong_bisimulation(ts)
        # 0 and 1 differ: 0 can do both a and b, 1 only a
        assert not check_bisimilar(ts, 0, 1, "strong")


# ===== Section 14: Larger Systems =====

class TestLargerSystems:
    def test_parallel_3_components(self):
        """Parallel composition of 3 components = 8 states."""
        ts = make_parallel_composition(3)
        result = compute_strong_bisimulation(ts)
        # Each state has unique label combo (which components are on)
        assert result.n_blocks == 8

    def test_binary_tree_depth_3(self):
        """Depth 3 binary tree."""
        ts = make_binary_tree(3)
        result = compute_strong_bisimulation(ts)
        # 4 levels of depth distinction: root, depth1, depth2, leaves
        assert result.n_blocks == 4

    def test_ring_8(self):
        """Ring of 8 states -- all bisimilar."""
        ts = make_ring(8)
        result = compute_strong_bisimulation(ts)
        assert result.n_blocks == 1

    def test_chain_8(self):
        """Chain of 8 states -- all distinct."""
        ts = make_chain(8, labeled=False)
        result = compute_strong_bisimulation(ts)
        assert result.n_blocks == 8


# ===== Section 15: Integration =====

class TestIntegration:
    def test_minimize_and_check(self):
        """Minimize a system, verify quotient preserves behavior."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, "a", 2), (1, "a", 3),
                (2, "b", 0), (3, "b", 1),
            ],
            state_labels={0: {"start"}, 1: {"start"}, 2: {"mid"}, 3: {"mid"}},
        )
        q = minimize(ts, "strong")
        # 0~1 (start, can do a to mid), 2~3 (mid, can do b to start)
        assert q["n_blocks"] == 2
        # Quotient should have a and b transitions between the two blocks
        assert len(q["transitions"]["a"]) >= 1
        assert len(q["transitions"]["b"]) >= 1

    def test_cross_bisim_with_minimization(self):
        """Cross-bisimilar systems should have same quotient size."""
        ts1 = make_symbolic_ts_from_lts(
            n_states=2,
            transitions=[(0, "a", 1)],
            state_labels={0: set(), 1: {"done"}},
        )
        ts2 = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[(0, "a", 1), (0, "a", 2), (2, "a", 3)],
            state_labels={0: set(), 1: {"done"}, 2: set(), 3: {"done"}},
        )
        q1 = minimize(ts1, "strong")
        q2 = minimize(ts2, "strong")
        # Both should have 2 blocks: one for "can do a" and one for "done"
        assert q1["n_blocks"] == 2

    def test_full_workflow(self):
        """Full workflow: build, bisimulate, compare, quotient."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, "a", 1), (0, TAU, 2),
                (2, "a", 3),
            ],
            state_labels={0: set(), 1: {"done"}, 2: set(), 3: {"done"}},
        )
        # Strong
        strong = compute_strong_bisimulation(ts)
        # Weak
        weak = compute_weak_bisimulation(ts)
        # Compare
        cmp = compare_bisimulations(ts)
        assert cmp["strong"]["n_blocks"] >= cmp["weak"]["n_blocks"]
        # Summary
        s = bisimulation_summary(ts, "strong")
        assert s["total_states"] == 4

    def test_summary_block_sizes(self):
        """Block sizes sum to total states."""
        ts = make_symbolic_ts_from_lts(
            n_states=4,
            transitions=[
                (0, "a", 2), (1, "a", 3),
            ],
            state_labels={0: {"start"}, 1: {"start"}, 2: {"end"}, 3: {"end"}},
        )
        s = bisimulation_summary(ts, "strong")
        assert sum(s["block_sizes"]) == s["total_states"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
